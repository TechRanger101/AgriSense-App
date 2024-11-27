from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sentinelhub import SentinelHubRequest, SentinelHubCatalog, DataCollection, MimeType, CRS, SHConfig, BBox
from shapely.geometry import shape, Point, mapping, Polygon, MultiPolygon
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.transform import from_bounds
import geopandas as gpd
from .serializers import EndDateSerializer, IndicesSerializer
import json
import pandas as pd
from prophet import Prophet
import random

config = SHConfig()
config.sh_client_id = '9db91b67-1611-42b4-8b62-4b18344e9146'
config.sh_client_secret = 'GR9vjnIhWF4DocX9nycKw7ulfF2aDk6T'

class SentinelDataAvailabilityView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = EndDateSerializer(data=request.data)
        if serializer.is_valid():
            end_date = serializer.validated_data['end_date']
            cloud_coverage = int(request.data.get('cloud_coverage', 100))
            if cloud_coverage > 100:
                return Response({'error': 'Cloud coverage value cannot be greater than 100.'}, status=status.HTTP_400_BAD_REQUEST)
            catalog = SentinelHubCatalog(config=config)
            search_results = catalog.search(
                collection=DataCollection.SENTINEL2_L2A,
                bbox=bbox,
                filter=f"eo:cloud_cover < {cloud_coverage}",
                time=('2023-01-01', end_date),
            )
            dates = {datetime.fromisoformat(result['properties']['datetime'][:-1]).date().isoformat() for result in search_results}
            return JsonResponse(list(dates), safe=False)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Vegetation Health
class NDVIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var ndvi;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB04 = getValue(validValuesB04);
                    var avgB11 = getValue(validValuesB11);
                    
                    // Modified NDVI calculation to include under-canopy plants using B11
                    if (avgB11 > 0.3) { // Threshold to identify tree canopies
                        // Adjust NDVI for plants under tree canopies
                        ndvi = (avgB08 - avgB04) / (avgB08 + avgB04 + avgB11); 
                    } else {
                        // Regular NDVI calculation
                        ndvi = (avgB08 - avgB04) / (avgB08 + avgB04);
                    }
                } else {
                    ndvi = -9999; // No valid data
                }

                return [ndvi];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_ndvi(ndvi_array):
                classified_array = np.zeros_like(ndvi_array, dtype=np.uint8)
                classified_array[(ndvi_array <= 0) & (ndvi_array != -9999)] = 1
                classified_array[(ndvi_array > 0) & (ndvi_array <= 0.1)] = 2
                classified_array[(ndvi_array > 0.1) & (ndvi_array <= 0.2)] = 3
                classified_array[(ndvi_array > 0.2) & (ndvi_array <= 0.4)] = 4
                classified_array[(ndvi_array > 0.4) & (ndvi_array <= 0.5)] = 5
                classified_array[(ndvi_array > 0.5) & (ndvi_array <= 0.6)] = 6
                classified_array[(ndvi_array > 0.6) & (ndvi_array <= 0.7)] = 7
                classified_array[(ndvi_array > 0.7) & (ndvi_array <= 1)] = 8
                classified_array[(ndvi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_ndvi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#COFFEE RIPENESS USING NIR
class NIRView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        date = request.data.get('date')
        if not date:
            return Response({'error': 'Date is required.'}, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
        function setup() {
            return { 
                input: ["B08", "B04", "SCL"],  // B08 for NIR, B04 for Red
                output: { bands: 1, sampleType: "FLOAT32" }, 
                mosaicking: "ORBIT" 
            };
        }

        function validate(sample) {
            var scl = sample.SCL;
            // Exclude cloud and other invalid pixels
            if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                return false; // Exclude clouds, cloud shadows, and water
            }
            return true;
        }

        function evaluatePixel(samples) {
            var validValuesB08 = [], validValuesB04 = [];
            var a = 0;

            for (var i = 0; i < samples.length; i++) {
                var sample = samples[i];
                if (sample.B08 > 0 && sample.B04 > 0) {
                    var isValid = validate(sample);
                    if (isValid) {
                        validValuesB08[a] = sample.B08;  // NIR
                        validValuesB04[a] = sample.B04;  // Red
                        a++;
                    }
                }
            }

            var nirValue = -9999; // Default if no valid data
            if (a > 0) {
                // Compute the average NIR value
                nirValue = validValuesB08.reduce((a, b) => a + b, 0) / a;  // Average NIR
            }

            return [nirValue];
        }
        """

        sentinel_request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
            responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = sentinel_request.get_data()[0]

        # Check if the response is empty (all invalid values)
        if np.all(response == -9999):
            return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

        transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

        def reclassify_nir(nir_array):
            classified_array = np.zeros_like(nir_array, dtype=np.uint8)
            classified_array[(nir_array <= 0.1) & (nir_array != -9999)] = 1  # Unripe
            classified_array[(nir_array > 0.1) & (nir_array <= 0.3)] = 2  # Almost ripe
            classified_array[(nir_array > 0.3) & (nir_array <= 0.5)] = 3  # Ripe
            classified_array[(nir_array > 0.5) & (nir_array <= 0.7)] = 4  # Overripe
            classified_array[(nir_array > 0.7) & (nir_array <= 1)] = 5    # Very overripe
            classified_array[(nir_array == -9999)] = 0  # Set cloudy pixels to 0
            return classified_array

        classified_image = reclassify_nir(response)
        shapes_gen = shapes(classified_image, mask=None, transform=transform)
        geometries = list(shapes_gen)

        features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
        geojson_data = {"type": "FeatureCollection", "features": features}

        geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
        geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
        intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
        intersection_geojson = intersection_df.to_json()

        return JsonResponse(json.loads(intersection_geojson))

# Humidity level
class NDWIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B03", "B08", "B11", "SCL"],  // Added B11 for detecting vegetation under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds, shadows, and invalid pixels
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude cloud and cloud shadow pixels
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB03 = [], validValuesB08 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B03 > 0 && sample.B08 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB03[a] = sample.B03;
                            validValuesB08[a] = sample.B08;
                            validValuesB11[a] = sample.B11;  // B11 added for under-tree vegetation detection
                            a++;
                        }
                    }
                }

                var ndwi;
                if (a > 0) {
                    var avgB03 = getValue(validValuesB03);
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);

                    if (avgB11 > 0.3) {  // Threshold to account for vegetation under tree canopies
                        // Adjust NDWI considering B11 for under-canopy water
                        ndwi = (avgB03 - avgB08) / (avgB03 + avgB08 + avgB11);
                    } else {
                        // Standard NDWI calculation
                        ndwi = (avgB03 - avgB08) / (avgB03 + avgB08);
                    }
                } else {
                    ndwi = -9999; // No valid data
                }

                return [ndwi];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == 0):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_ndwi(ndwi_array):
                classified_array = np.zeros_like(ndwi_array, dtype=np.uint8)
                classified_array[(ndwi_array <= -1) & (ndwi_array != -9999)] = 1
                classified_array[(ndwi_array > -1) & (ndwi_array <= 0)] = 2
                classified_array[(ndwi_array > 0) & (ndwi_array <= 0.1)] = 3
                classified_array[(ndwi_array > 0.1) & (ndwi_array <= 0.2)] = 4
                classified_array[(ndwi_array > 0.2) & (ndwi_array <= 0.3)] = 5
                classified_array[(ndwi_array > 0.3) & (ndwi_array <= 0.4)] = 6
                classified_array[(ndwi_array > 0.4) & (ndwi_array <= 0.5)] = 7
                classified_array[(ndwi_array > 0.5) & (ndwi_array <= 1)] = 8
                classified_array[(ndwi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_ndwi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Plant Moisture
class NDMIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B08", "B11", "B04", "SCL"],  // Added B04 for vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds and invalid pixels, keeping tree canopy (SCL = 4) for analysis
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB11 = [], validValuesB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B11 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB11[a] = sample.B11;
                            validValuesB04[a] = sample.B04;  // Added B04 for better vegetation assessment
                            a++;
                        }
                    }
                }

                var ndmi;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);
                    var avgB04 = getValue(validValuesB04);

                    if (avgB04 > 0.3) { // Threshold to detect vegetation under tree canopies
                        // Adjust NDMI for plants under tree canopies
                        ndmi = (avgB08 - avgB11) / (avgB08 + avgB11 + avgB04); 
                    } else {
                        // Regular NDMI calculation
                        ndmi = (avgB08 - avgB11) / (avgB08 + avgB11);
                    }
                } else {
                    ndmi = -9999; // No valid data
                }

                return [ndmi];
            }
            """
            
            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == 0):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_ndmi(ndmi_array):
                classified_array = np.zeros_like(ndmi_array, dtype=np.uint8)
                classified_array[(ndmi_array <= -1) & (ndmi_array != -9999)] = 1
                classified_array[(ndmi_array > -1) & (ndmi_array <= 0)] = 2
                classified_array[(ndmi_array > 0) & (ndmi_array <= 0.1)] = 3
                classified_array[(ndmi_array > 0.1) & (ndmi_array <= 0.2)] = 4
                classified_array[(ndmi_array > 0.2) & (ndmi_array <= 0.3)] = 5
                classified_array[(ndmi_array > 0.3) & (ndmi_array <= 0.4)] = 6
                classified_array[(ndmi_array > 0.4) & (ndmi_array <= 0.5)] = 7
                classified_array[(ndmi_array > 0.5) & (ndmi_array <= 1)] = 8
                classified_array[(ndmi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_ndmi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Coffee Ripeness
class CRIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)

        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B11", "SCL"],  // Included B11 for detecting vegetation under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB04[a] = sample.B04;
                            a++;
                        }
                    }
                }

                var cri;
                if (a > 0) {
                    cri = getValue(validValuesB04);
                } else {
                    cri = -9999; // No valid data
                }

                return [cri];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_cri(cri_array):
                classified_array = np.zeros_like(cri_array, dtype=np.uint8)
                classified_array[(cri_array <= 10) & (cri_array != -9999)] = 1
                classified_array[(cri_array > 10) & (cri_array <= 20)] = 2
                classified_array[(cri_array > 20) & (cri_array <= 30)] = 3
                classified_array[(cri_array > 30) & (cri_array <= 40)] = 4
                classified_array[(cri_array > 40) & (cri_array <= 50)] = 5
                classified_array[(cri_array > 50) & (cri_array <= 60)] = 6
                classified_array[(cri_array > 60) & (cri_array <= 70)] = 7
                classified_array[(cri_array > 70) & (cri_array <= 100)] = 8
                classified_array[(cri_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_cri(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Ground Temperature
class LSTView(APIView):
    def post(self, request, format=None):
        # Extract GeoJSON polygon from the request
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        # Define bounding box based on the polygon
        bbox = polygon.bounds  # (minx, miny, maxx, maxy)
        bbox = BBox(bbox=(bbox[0], bbox[1], bbox[2], bbox[3]), crs=CRS.WGS84)
      
        # Create serializer instance with request data
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            # Extract validated data
            date = serializer.validated_data['date']
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        evalscript = """
            var option = 0;
            var minC = 0;
            var maxC = 50;
            var NDVIs = 0.2;
            var NDVIv = 0.8;
            var waterE = 0.991;
            var soilE = 0.966;
            var vegetationE = 0.973;
            var C = 0.009;
            var bCent = 0.000010854;
            var rho = 0.01438;

            if (option == 2) {
                minC = 0;
                maxC = 25;
            }

            let viz = ColorGradientVisualizer.createRedTemperature(minC, maxC);

            function setup() {
                return {
                    input: [
                        { datasource: "S3SLSTR", bands: ["S8"] },
                        { datasource: "S3OLCI", bands: ["B06", "B08", "B11"] } // Include B11 for vegetation detection under trees
                    ],
                    output: [
                        { id: "default", bands: 3, sampleType: SampleType.AUTO }
                    ],
                    mosaicking: "ORBIT"
                };
            }

            // Cloud validation function
            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds and invalid pixels
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude cloud and cloud shadow pixels
                }
                return true;
            }

            // Function to calculate Land Surface Emissivity (LSE)
            function LSEcalc(NDVI, Pv) {
                var LSE;
                if (NDVI < 0) {
                    LSE = waterE;
                } else if (NDVI < NDVIs) {
                    LSE = soilE;
                } else if (NDVI > NDVIv) {
                    LSE = vegetationE;
                } else {
                    LSE = vegetationE * Pv + soilE * (1 - Pv) + C;
                }
                return LSE;
            }

            function evaluatePixel(samples) {
                var validLSTs = [];
                var invalidLSTs = [];
                var N = samples.S3SLSTR.length;

                for (let i = 0; i < N; i++) {
                    var sampleSLSTR = samples.S3SLSTR[i];
                    var sampleOLCI = samples.S3OLCI[i];

                    var Bi = sampleSLSTR.S8;
                    var B06i = sampleOLCI.B06;
                    var B08i = sampleOLCI.B08;
                    var B11i = sampleOLCI.B11; // Added B11 for better vegetation assessment

                    if ((Bi <= 173 || Bi >= 65000) || (B06i <= 0 || B08i <= 0 || B11i <= 0)) {
                        continue; // Skip invalid measurements
                    }

                    var isValid = validate(sampleOLCI); // Validate using the cloud mask (SCL)
                    var S8BTi = Bi - 273.15; // Convert to Celsius
                    var NDVIi = (B08i - B11i) / (B08i + B11i); // Adjusted NDVI calculation
                    var PVi = Math.pow(((NDVIi - NDVIs) / (NDVIv - NDVIs)), 2);
                    var LSEi = LSEcalc(NDVIi, PVi);
                    var LSTi = (S8BTi / (1 + (((bCent * S8BTi) / rho) * Math.log(LSEi))));

                    if (isValid) {
                        validLSTs.push(LSTi);
                    } else {
                        invalidLSTs.push(LSTi);
                    }
                }

                // Select valid LSTs if available, otherwise fall back to invalid ones
                var LSTsToUse = validLSTs.length > 0 ? validLSTs : invalidLSTs;

                var outLST;
                if (option == 0) {
                    outLST = LSTsToUse.reduce((a, b) => a + b, 0) / LSTsToUse.length; // Average
                } else if (option == 1) {
                    outLST = Math.max(...LSTsToUse); // Maximum
                } else {
                    var avg = LSTsToUse.reduce((a, b) => a + b, 0) / LSTsToUse.length;
                    outLST = Math.sqrt(LSTsToUse.reduce((sum, lst) => sum + Math.pow(lst - avg, 2), 0) / (LSTsToUse.length - 1)); // Standard deviation
                }

                return viz.process(outLST);
            }
        """

        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL3_SLSTR,
                    identifier="S3SLSTR",
                    time_interval=(date, date),
                ),
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL3_OLCI,
                    identifier="S3OLCI",
                    time_interval=(date, date),
                ),
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF),
            ],
            bbox=bbox,
            size=[512, 354.253],
            config=config,
        )

        response = request.get_data()
        response_data = response[0]
        print(response_data)

        transform = rasterio.transform.from_bounds(*bbox, response_data.shape[1], response_data.shape[0])

        # Scaling factors
        minC = 0  # Set based on your evalscript
        maxC = 50  # Set based on your evalscript

        # Convert red channel values to temperature
        response_data_temp = minC + (response_data[..., 0] / 255.0) * (maxC - minC)

        # Check if any value in red channel is 255
        if np.any(response_data[..., 0] == 255):
            return Response({'error': 'Red channel value is 255, possibly querying temperature for tomorrow.'}, status=status.HTTP_400_BAD_REQUEST)

        # Calculate mean of response_data_temp and round to nearest integer
        mean_temp = np.round(np.mean(response_data_temp))

        return Response({'mean_temperature': int(mean_temp)}, status=status.HTTP_200_OK)


#Water Stress
class WaterStressIndexView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B04", "B08", "B11", "SCL"], // Added B11 for better vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" },
                    mosaicking: "ORBIT"
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // Filter by recent data
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var wst;
                if (a > 0) {
                    // Calculate WST using valid pixel data
                    var avgB04 = getValue(validValuesB04);
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);

                    // Modified WST calculation to include under-canopy plants using B11
                    wst = (avgB04 - avgB08) / (avgB04 + avgB08 + avgB11);
                } else {
                    wst = -9999; // No valid data
                }

                return [wst];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_wsi(wsi_array):
                classified_array = np.zeros_like(wsi_array, dtype=np.uint8)
                classified_array[(wsi_array <= -1) & (wsi_array != -9999)] = 1
                classified_array[(wsi_array > -1) & (wsi_array <= 0)] = 2
                classified_array[(wsi_array > 0) & (wsi_array <= 0.1)] = 3
                classified_array[(wsi_array > 0.1) & (wsi_array <= 0.2)] = 4
                classified_array[(wsi_array > 0.2) & (wsi_array <= 0.3)] = 5
                classified_array[(wsi_array > 0.3) & (wsi_array <= 0.4)] = 6
                classified_array[(wsi_array > 0.4) & (wsi_array <= 0.5)] = 7
                classified_array[(wsi_array > 0.5) & (wsi_array <= 1)] = 8
                classified_array[(wsi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_wsi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Crop Yield
class CropYieldIndexView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return { 
                    input: ["B04", "B08", "B02", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude clouds and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB02[a] = sample.B02;
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var arvi;
                if (a > 0) {
                    var NIR = getValue(validValuesB08);
                    var RED = getValue(validValuesB04);
                    var BLUE = getValue(validValuesB02);
                    var avgB11 = getValue(validValuesB11);

                    // Adjusting ARVI for plants under tree canopies using B11
                    if (avgB11 > 0.3) { // Threshold to identify tree canopies
                        arvi = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE + avgB11));
                    } else {
                        arvi = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
                    }
                } else {
                    arvi = -9999; // No valid data
                }

                return [arvi];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            # Check if the response is empty (all invalid values)
            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_cyi(cyi_array):
                classified_array = np.zeros_like(cyi_array, dtype=np.uint8)
                classified_array[(cyi_array <= 0) & (cyi_array != -9999)] = 1
                classified_array[(cyi_array > 0) & (cyi_array <= 0.1)] = 2
                classified_array[(cyi_array > 0.1) & (cyi_array <= 0.2)] = 3
                classified_array[(cyi_array > 0.2) & (cyi_array <= 0.3)] = 4
                classified_array[(cyi_array > 0.3) & (cyi_array <= 0.4)] = 5
                classified_array[(cyi_array > 0.4) & (cyi_array <= 0.5)] = 6
                classified_array[(cyi_array > 0.5) & (cyi_array <= 0.6)] = 7
                classified_array[(cyi_array > 0.6) & (cyi_array <= 1)] = 8
                classified_array[(cyi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_cyi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Disease Weed
class ARVIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B02", "B04", "B08", "B11", "SCL"],  // Include Blue, Red, NIR, SWIR bands and Scene Classification Layer (SCL)
                    output: {
                        id: "default",
                        bands: 1,
                        sampleType: "FLOAT32"
                    },
                    mosaicking: "ORBIT"  // Use ORBIT-based mosaicking
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);  // Filter scenes from the last 3 months
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);  // Use first quartile as the value
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;  // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB02[a] = sample.B02;  
                            validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var ARVI;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB04 = getValue(validValuesB04);
                    var avgB02 = getValue(validValuesB02);
                    var avgB11 = getValue(validValuesB11);
                    
                    // Adjust ARVI calculation to account for plants under tree canopies
                    if (avgB11 > 0.3) { // Threshold to identify tree canopies
                        // Adjust ARVI for under-canopy plants using SWIR
                        ARVI = (avgB08 - (2 * avgB04 - avgB02)) / (avgB08 + (2 * avgB04 - avgB02) + avgB11);
                    } else {
                        // Regular ARVI calculation
                        ARVI = (avgB08 - (2 * avgB04 - avgB02)) / (avgB08 + (2 * avgB04 - avgB02));
                    }
                } else {
                    ARVI = -9999;  // No valid data
                }

                return [ARVI];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == 0):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            # The reclassification logic is the same as before
            def reclassify_arvi(arvi_array):
                classified_array = np.zeros_like(arvi_array, dtype=np.uint8)
                classified_array[(arvi_array <= 0) & (arvi_array != -9999)] = 1
                classified_array[(arvi_array > 0) & (arvi_array <= 0.1)] = 2
                classified_array[(arvi_array > 0.1) & (arvi_array <= 0.2)] = 3
                classified_array[(arvi_array > 0.2) & (arvi_array <= 0.3)] = 4
                classified_array[(arvi_array > 0.3) & (arvi_array <= 0.4)] = 5
                classified_array[(arvi_array > 0.4) & (arvi_array <= 0.5)] = 6
                classified_array[(arvi_array > 0.5) & (arvi_array <= 0.6)] = 7
                classified_array[(arvi_array > 0.6) & (arvi_array <= 1)] = 8
                classified_array[(arvi_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_arvi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"arvi_class": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Chlorophyll
class CARIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B03", "B04", "B08", "B11", "SCL"], // Added B11 for better vegetation detection
                    output: {
                        id: "default",
                        bands: 1,
                        sampleType: "FLOAT32"
                    },
                    mosaicking: "ORBIT"
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // 3-month filter
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude clouds, cloud shadows, and water
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB04 = [], validValuesB03 = [], validValuesB11 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B04 > 0 && sample.B03 > 0 && sample.B11 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB04[a] = sample.B04;
                            validValuesB03[a] = sample.B03;
                            validValuesB11[a] = sample.B11; // Using B11 for vegetation under trees
                            a++;
                        }
                    }
                }

                var CARI;
                if (a > 0) {
                    var GREEN = getValue(validValuesB03);
                    var RED = getValue(validValuesB04);
                    var NIR = getValue(validValuesB08);
                    var SWIR = getValue(validValuesB11); // Incorporating B11

                    // Calculate CARI considering vegetation under trees
                    var term1 = Math.pow((NIR - GREEN) / 150, 2);
                    var term2 = Math.pow((RED - GREEN), 2);
                    CARI = Math.sqrt(term1 + term2);

                    // Adjust CARI calculation for conditions under tree canopies
                    if (SWIR > 0.3) { // Threshold for identifying vegetation under canopies
                        CARI *= 1.1; // Example adjustment factor
                    }
                } else {
                    CARI = -9999; // No valid data
                }

                return [CARI];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_cari(cari_array):
                classified_array = np.zeros_like(cari_array, dtype=np.uint8)
                classified_array[(cari_array <= 0) & (cari_array != -9999)] = 1
                classified_array[(cari_array > 0) & (cari_array <= 0.1)] = 2
                classified_array[(cari_array > 0.1) & (cari_array <= 0.2)] = 3
                classified_array[(cari_array > 0.2) & (cari_array <= 0.3)] = 4
                classified_array[(cari_array > 0.3) & (cari_array <= 0.4)] = 5
                classified_array[(cari_array > 0.4) & (cari_array <= 0.5)] = 6
                classified_array[(cari_array > 0.5) & (cari_array <= 0.6)] = 7
                classified_array[(cari_array > 0.6) & (cari_array <= 1)] = 8
                classified_array[(cari_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_cari(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"cari_class": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Chlorophyll Growth
class MCARIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
                return {
                    input: ["B02", "B03", "B04", "B08", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                    output: {
                        id: "default",
                        bands: 1,
                        sampleType: "FLOAT32"
                    },
                    mosaicking: "ORBIT"
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // Last 3 months of data
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesBlue = [], validValuesGreen = [], validValuesRed = [], validValuesNIR = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B02 > 0 && sample.B03 > 0 && sample.B04 > 0 && sample.B08 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesBlue[a] = sample.B02;
                            validValuesGreen[a] = sample.B03;
                            validValuesRed[a] = sample.B04;
                            validValuesNIR[a] = sample.B08; // Including NIR for under-canopy plants
                            a++;
                        }
                    }
                }

                var mcari;
                if (a > 0) {
                    // Calculate MCARI using valid values
                    var BLUE = getValue(validValuesBlue);
                    var GREEN = getValue(validValuesGreen);
                    var RED = getValue(validValuesRed);
                    var NIR = getValue(validValuesNIR);

                    mcari = (RED - GREEN) - 0.2 * (RED - BLUE) * (RED / NIR);
                } else {
                    mcari = -9999; // No valid data
                }

                return [mcari];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_mcari(mcari_array):
                classified_array = np.zeros_like(mcari_array, dtype=np.uint8)
                classified_array[(mcari_array <= 0) & (mcari_array != -9999)] = 1
                classified_array[(mcari_array > 0) & (mcari_array <= 0.1)] = 2
                classified_array[(mcari_array > 0.1) & (mcari_array <= 0.2)] = 3
                classified_array[(mcari_array > 0.2) & (mcari_array <= 0.3)] = 4
                classified_array[(mcari_array > 0.3) & (mcari_array <= 0.4)] = 5
                classified_array[(mcari_array > 0.4) & (mcari_array <= 0.5)] = 6
                classified_array[(mcari_array > 0.5) & (mcari_array <= 0.6)] = 7
                classified_array[(mcari_array > 0.6) & (mcari_array <= 1)] = 8
                classified_array[(mcari_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_mcari(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"mcari_class": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Chlorophyll Growth
class MCARIView(APIView):
    def post(self, request):
        geojson_polygon = request.data.get('geometry')
        if not geojson_polygon:
            return Response({'error': 'GeoJSON polygon is required.'}, status=status.HTTP_400_BAD_REQUEST)

        try:
            polygon = shape(geojson_polygon['geometry'])
        except Exception as e:
            return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

        bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
        serializer = IndicesSerializer(data=request.data)
        if serializer.is_valid():
            date = serializer.validated_data['date']
            evalscript = """
            function setup() {
              return {
                input: [{
                  bands: [
                    "B04",
                    "B03",
                    "B02",
                    "SCL"
                  ]
                }],
                output: { bands: 3, sampleType: "UINT16" },
                mosaicking: "ORBIT"
              };
            }

            function preProcessScenes(collections) {
              collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                var orbitDateFrom = new Date(orbit.dateFrom)
                return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
              })
              return collections
            }

            function getValue(values) {
              values.sort(function (a, b) { return a - b; });
              return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
              var index = Math.floor(sortedValues.length / 4);
              return sortedValues[index];
            }

            function validate(samples) {
              var scl = samples.SCL;

              if (scl === 3 || scl === 9 || scl === 8 || scl === 7 || scl === 10 || scl === 11 || scl === 1) {
                return false;
              }
              return true;
            }

            function evaluatePixel(samples) {
              var clo_b02 = []; var clo_b03 = []; var clo_b04 = [];
              var a = 0;

              for (var i = 0; i < samples.length; i++) {
                var sample = samples[i];

                if (sample.B02 > 0 && sample.B03 > 0 && sample.B04 > 0) {
                  var isValid = validate(sample);

                  if (isValid) {
                    clo_b02[a] = sample.B02;
                    clo_b03[a] = sample.B03;
                    clo_b04[a] = sample.B04;
                    a = a + 1;
                  }
                }
              }

              var rValue, gValue, bValue;
              if (a > 0) {
                rValue = getValue(clo_b04);
                gValue = getValue(clo_b03);
                bValue = getValue(clo_b02);
              } else {
                rValue = gValue = bValue = 0; // No valid data case
              }
              return [rValue * 10000, gValue * 10000, bValue * 10000];
            }
            """

            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                ],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            def reclassify_mcari(mcari_array):
                classified_array = np.zeros_like(mcari_array, dtype=np.uint8)
                classified_array[(mcari_array <= 0) & (mcari_array != -9999)] = 1
                classified_array[(mcari_array > 0) & (mcari_array <= 0.1)] = 2
                classified_array[(mcari_array > 0.1) & (mcari_array <= 0.2)] = 3
                classified_array[(mcari_array > 0.2) & (mcari_array <= 0.3)] = 4
                classified_array[(mcari_array > 0.3) & (mcari_array <= 0.4)] = 5
                classified_array[(mcari_array > 0.4) & (mcari_array <= 0.5)] = 6
                classified_array[(mcari_array > 0.5) & (mcari_array <= 0.6)] = 7
                classified_array[(mcari_array > 0.6) & (mcari_array <= 1)] = 8
                classified_array[(mcari_array == -9999)] = 0  # Set cloudy pixels to 0
                return classified_array

            classified_image = reclassify_mcari(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"mcari_class": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            return JsonResponse(json.loads(intersection_geojson))
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# Vegetation Health Forecast
class NDVIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return { 
                        input: ["B04", "B08", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                        output: { bands: 1, sampleType: "FLOAT32" }, 
                        mosaicking: "ORBIT" 
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude clouds, cloud shadows, and water
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB08 = [], validValuesB04 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B08 > 0 && sample.B04 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB08[a] = sample.B08;
                                validValuesB04[a] = sample.B04;
                                validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                                a++;
                            }
                        }
                    }

                    var ndvi;
                    if (a > 0) {
                        var avgB08 = getValue(validValuesB08);
                        var avgB04 = getValue(validValuesB04);
                        var avgB11 = getValue(validValuesB11);
                        
                        // Modified NDVI calculation to include under-canopy plants using B11
                        if (avgB11 > 0.3) { // Threshold to identify tree canopies
                            // Adjust NDVI for plants under tree canopies
                            ndvi = (avgB08 - avgB04) / (avgB08 + avgB04 + avgB11); 
                        } else {
                            // Regular NDVI calculation
                            ndvi = (avgB08 - avgB04) / (avgB08 + avgB04);
                        }
                    } else {
                        ndvi = -9999; // No valid data
                    }

                    return [ndvi];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue 

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_ndvi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ndvi(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ndvi(self, ndvi_array):
        classified_array = np.zeros_like(ndvi_array, dtype=np.uint8)
        classified_array[(ndvi_array <= 0) & (ndvi_array != -9999)] = 1
        classified_array[(ndvi_array > 0) & (ndvi_array <= 0.1)] = 2
        classified_array[(ndvi_array > 0.1) & (ndvi_array <= 0.2)] = 3
        classified_array[(ndvi_array > 0.2) & (ndvi_array <= 0.4)] = 4
        classified_array[(ndvi_array > 0.4) & (ndvi_array <= 0.5)] = 5
        classified_array[(ndvi_array > 0.5) & (ndvi_array <= 0.6)] = 6
        classified_array[(ndvi_array > 0.6) & (ndvi_array <= 0.7)] = 7
        classified_array[(ndvi_array > 0.7) & (ndvi_array <= 1)] = 8
        classified_array[(ndvi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_ndvi(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array) 

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates'] 
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Humidity Forecast
class NDWIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return { 
                        input: ["B03", "B08", "B11", "SCL"],  // Added B11 for detecting vegetation under trees
                        output: { bands: 1, sampleType: "FLOAT32" }, 
                        mosaicking: "ORBIT" 
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Using SCL to filter out clouds, shadows, and invalid pixels
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude cloud and cloud shadow pixels
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB03 = [], validValuesB08 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B03 > 0 && sample.B08 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB03[a] = sample.B03;
                                validValuesB08[a] = sample.B08;
                                validValuesB11[a] = sample.B11;  // B11 added for under-tree vegetation detection
                                a++;
                            }
                        }
                    }

                    var ndwi;
                    if (a > 0) {
                        var avgB03 = getValue(validValuesB03);
                        var avgB08 = getValue(validValuesB08);
                        var avgB11 = getValue(validValuesB11);

                        if (avgB11 > 0.3) {  // Threshold to account for vegetation under tree canopies
                            // Adjust NDWI considering B11 for under-canopy water
                            ndwi = (avgB03 - avgB08) / (avgB03 + avgB08 + avgB11);
                        } else {
                            // Standard NDWI calculation
                            ndwi = (avgB03 - avgB08) / (avgB03 + avgB08);
                        }
                    } else {
                        ndwi = -9999; // No valid data
                    }

                    return [ndwi];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[
                        SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                    ],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue 

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_ndwi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ndwi(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ndwi(self, ndwi_array):
        classified_array = np.zeros_like(ndwi_array, dtype=np.uint8)
        classified_array[(ndwi_array <= 0) & (ndwi_array != -9999)] = 1
        classified_array[(ndwi_array > 0) & (ndwi_array <= 0.1)] = 2
        classified_array[(ndwi_array > 0.1) & (ndwi_array <= 0.2)] = 3
        classified_array[(ndwi_array > 0.2) & (ndwi_array <= 0.4)] = 4
        classified_array[(ndwi_array > 0.4) & (ndwi_array <= 0.5)] = 5
        classified_array[(ndwi_array > 0.5) & (ndwi_array <= 0.6)] = 6
        classified_array[(ndwi_array > 0.6) & (ndwi_array <= 0.7)] = 7
        classified_array[(ndwi_array > 0.7) & (ndwi_array <= 1)] = 8
        classified_array[(ndwi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_ndwi(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        # Placeholder for prediction logic, you can implement your own model prediction
        predicted_class = np.round(np.mean(class_numbers_array)).astype(int) 

        # Formulate output features
        for idx, geometry in enumerate(valid_results):
            predicted_features.append({
                "type": "Feature",
                "geometry": geometry['geometry'],
                "properties": {"predicted_class": predicted_class}
            })

        return {
            "type": "FeatureCollection",
            "features": predicted_features
        }


# Plant Moisture Forecast
class NDMIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            
            evalscript = """
            function setup() {
                return { 
                    input: ["B08", "B11", "B04", "SCL"],  // Added B04 for vegetation detection under trees
                    output: { bands: 1, sampleType: "FLOAT32" }, 
                    mosaicking: "ORBIT" 
                };
            }

            function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom);
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                });
                return collections;
            }

            function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
            }

            function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
            }

            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds and invalid pixels, keeping tree canopy (SCL = 4) for analysis
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude clouds, cloud shadows, and water
                }
                return true;
            }

            function evaluatePixel(samples, scenes) {
                var validValuesB08 = [], validValuesB11 = [], validValuesB04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];
                    if (sample.B08 > 0 && sample.B11 > 0) {
                        var isValid = validate(sample);
                        if (isValid) {
                            validValuesB08[a] = sample.B08;
                            validValuesB11[a] = sample.B11;
                            validValuesB04[a] = sample.B04;  // Added B04 for better vegetation assessment
                            a++;
                        }
                    }
                }

                var ndmi;
                if (a > 0) {
                    var avgB08 = getValue(validValuesB08);
                    var avgB11 = getValue(validValuesB11);
                    var avgB04 = getValue(validValuesB04);

                    if (avgB04 > 0.3) { // Threshold to detect vegetation under tree canopies
                        // Adjust NDMI for plants under tree canopies
                        ndmi = (avgB08 - avgB11) / (avgB08 + avgB11 + avgB04); 
                    } else {
                        // Regular NDMI calculation
                        ndmi = (avgB08 - avgB11) / (avgB08 + avgB11);
                    }
                } else {
                    ndmi = -9999; // No valid data
                }

                return [ndmi];
            }
            """
            
            sentinel_request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            response = sentinel_request.get_data()[0]

            if np.all(response == -9999):
                continue 

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

            classified_image = self.reclassify_ndmi(response)
            shapes_gen = shapes(classified_image, mask=None, transform=transform)
            geometries = list(shapes_gen)

            features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
            geojson_data = {"type": "FeatureCollection", "features": features}

            geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
            geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
            intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
            intersection_geojson = intersection_df.to_json()

            results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ndmi(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ndmi(self, ndmi_array):
        classified_array = np.zeros_like(ndmi_array, dtype=np.uint8)
        classified_array[(ndmi_array <= -1) & (ndmi_array != -9999)] = 1
        classified_array[(ndmi_array > -1) & (ndmi_array <= 0)] = 2
        classified_array[(ndmi_array > 0) & (ndmi_array <= 0.1)] = 3
        classified_array[(ndmi_array > 0.1) & (ndmi_array <= 0.2)] = 4
        classified_array[(ndmi_array > 0.2) & (ndmi_array <= 0.3)] = 5
        classified_array[(ndmi_array > 0.3) & (ndmi_array <= 0.4)] = 6
        classified_array[(ndmi_array > 0.4) & (ndmi_array <= 0.5)] = 7
        classified_array[(ndmi_array > 0.5) & (ndmi_array <= 1)] = 8
        classified_array[(ndmi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_ndmi(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 # print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array) 

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates'] 
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Coffee Ripeness Forecast
class CRIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return { 
                        input: ["B04", "B11", "SCL"],  // Included B11 for detecting vegetation under trees
                        output: { bands: 1, sampleType: "FLOAT32" }, 
                        mosaicking: "ORBIT" 
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude clouds, cloud shadows, and water
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB04 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B04 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB04[a] = sample.B04;
                                a++;
                            }
                        }
                    }

                    var cri;
                    if (a > 0) {
                        cri = getValue(validValuesB04);
                    } else {
                        cri = -9999; // No valid data
                    }

                    return [cri];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_ripeness(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"ripeness_class": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_ripeness(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_ripeness(self, ripeness_array):
        classified_array = np.zeros_like(ripeness_array, dtype=np.uint8)
        classified_array[(ripeness_array <= 0) & (ripeness_array != -9999)] = 1  # Low ripeness
        classified_array[(ripeness_array > 0) & (ripeness_array <= 0.3)] = 2  # Medium ripeness
        classified_array[(ripeness_array > 0.3)] = 3  # High ripeness
        classified_array[(ripeness_array == -9999)] = 0  # Cloudy pixels
        return classified_array

    def predict_ripeness(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_ripeness_classes = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        ripeness_class = item.get('properties', {}).get('ripeness_class')
                        if ripeness_class is not None:
                            valid_ripeness_classes.append(ripeness_class)
                        else:
                            print("Ripeness class missing in properties.")
                    else:
                        a = 3  #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_ripeness_classes:
            print("No valid coordinates or ripeness classes found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        ripeness_classes_array = np.array(valid_ripeness_classes)

        model = LinearRegression()
        model.fit(coordinates_array, ripeness_classes_array)

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_ripeness_class = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": ripeness_class
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates']
                }
            }
            for i, (idx, ripeness_class) in enumerate(zip(sampled_indices, predicted_ripeness_class))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Ground Temperature Forecast
class LSTFView(APIView):
    def post(self, request, format=None):
        geojson_features = request.data.get('features')
        if not geojson_features:
            return Response({'error': 'GeoJSON features are required.'}, status=status.HTTP_400_BAD_REQUEST)

        all_temperatures = []

        # Process each polygon feature in the input
        for feature in geojson_features:
            geojson_polygon = feature.get('geometry')
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                return Response({'error': 'Each feature must have a geometry and a date.'}, status=status.HTTP_400_BAD_REQUEST)

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            # Define bounding box based on the polygon
            bbox = polygon.bounds  # (minx, miny, maxx, maxy)
            bbox = BBox(bbox=(bbox[0], bbox[1], bbox[2], bbox[3]), crs=CRS.WGS84)

            evalscript = """
            var option = 0;
            var minC = 0;
            var maxC = 50;
            var NDVIs = 0.2;
            var NDVIv = 0.8;
            var waterE = 0.991;
            var soilE = 0.966;
            var vegetationE = 0.973;
            var C = 0.009;
            var bCent = 0.000010854;
            var rho = 0.01438;

            if (option == 2) {
                minC = 0;
                maxC = 25;
            }

            let viz = ColorGradientVisualizer.createRedTemperature(minC, maxC);

            function setup() {
                return {
                    input: [
                        { datasource: "S3SLSTR", bands: ["S8"] },
                        { datasource: "S3OLCI", bands: ["B06", "B08", "B11"] } // Include B11 for vegetation detection under trees
                    ],
                    output: [
                        { id: "default", bands: 3, sampleType: SampleType.AUTO }
                    ],
                    mosaicking: "ORBIT"
                };
            }

            // Cloud validation function
            function validate(sample) {
                var scl = sample.SCL;
                // Using SCL to filter out clouds and invalid pixels
                if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                    return false; // Exclude cloud and cloud shadow pixels
                }
                return true;
            }

            // Function to calculate Land Surface Emissivity (LSE)
            function LSEcalc(NDVI, Pv) {
                var LSE;
                if (NDVI < 0) {
                    LSE = waterE;
                } else if (NDVI < NDVIs) {
                    LSE = soilE;
                } else if (NDVI > NDVIv) {
                    LSE = vegetationE;
                } else {
                    LSE = vegetationE * Pv + soilE * (1 - Pv) + C;
                }
                return LSE;
            }

            function evaluatePixel(samples) {
                var validLSTs = [];
                var invalidLSTs = [];
                var N = samples.S3SLSTR.length;

                for (let i = 0; i < N; i++) {
                    var sampleSLSTR = samples.S3SLSTR[i];
                    var sampleOLCI = samples.S3OLCI[i];

                    var Bi = sampleSLSTR.S8;
                    var B06i = sampleOLCI.B06;
                    var B08i = sampleOLCI.B08;
                    var B11i = sampleOLCI.B11; // Added B11 for better vegetation assessment

                    if ((Bi <= 173 || Bi >= 65000) || (B06i <= 0 || B08i <= 0 || B11i <= 0)) {
                        continue; // Skip invalid measurements
                    }

                    var isValid = validate(sampleOLCI); // Validate using the cloud mask (SCL)
                    var S8BTi = Bi - 273.15; // Convert to Celsius
                    var NDVIi = (B08i - B11i) / (B08i + B11i); // Adjusted NDVI calculation
                    var PVi = Math.pow(((NDVIi - NDVIs) / (NDVIv - NDVIs)), 2);
                    var LSEi = LSEcalc(NDVIi, PVi);
                    var LSTi = (S8BTi / (1 + (((bCent * S8BTi) / rho) * Math.log(LSEi))));

                    if (isValid) {
                        validLSTs.push(LSTi);
                    } else {
                        invalidLSTs.push(LSTi);
                    }
                }

                // Select valid LSTs if available, otherwise fall back to invalid ones
                var LSTsToUse = validLSTs.length > 0 ? validLSTs : invalidLSTs;

                var outLST;
                if (option == 0) {
                    outLST = LSTsToUse.reduce((a, b) => a + b, 0) / LSTsToUse.length; // Average
                } else if (option == 1) {
                    outLST = Math.max(...LSTsToUse); // Maximum
                } else {
                    var avg = LSTsToUse.reduce((a, b) => a + b, 0) / LSTsToUse.length;
                    outLST = Math.sqrt(LSTsToUse.reduce((sum, lst) => sum + Math.pow(lst - avg, 2), 0) / (LSTsToUse.length - 1)); // Standard deviation
                }

                return viz.process(outLST);
            }
        """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL3_SLSTR,
                        identifier="S3SLSTR",
                        time_interval=(date, date),
                    ),
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL3_OLCI,
                        identifier="S3OLCI",
                        time_interval=(date, date),
                    )
                ],
                responses=[
                    SentinelHubRequest.output_response('default', MimeType.TIFF),
                ],
                bbox=bbox,
                size=[512, 354.253],
                config=config,
            )

            # Get the data and calculate temperature
            response = request.get_data()
            if response and len(response) > 0:
                response_data = response[0]
                minC = 0  # Set based on your evalscript
                maxC = 50  # Set based on your evalscript

                response_data_temp = minC + (response_data[..., 0] / 255.0) * (maxC - minC)

                # Calculate mean of response_data_temp and round to nearest integer
                mean_temp = np.round(np.mean(response_data_temp))
                all_temperatures.append(mean_temp)

        # Predict next week's temperature using a simple model
        if len(all_temperatures) > 0:
            predicted_temperature = self.simple_prediction(all_temperatures)
            return Response({'mean_temperature': predicted_temperature}, status=status.HTTP_200_OK)

        return Response({'error': 'No temperature data available for predictions.'}, status=status.HTTP_400_BAD_REQUEST)

    def simple_prediction(self, temperatures):
        # Simple prediction model: average of the collected temperatures
        return int(np.mean(temperatures))  # Return average as the predicted temperature for next week


# Water Stress Forecast
class WaterStressIndexForecastView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return {
                        input: ["B04", "B08", "B11", "SCL"], // Added B11 for better vegetation detection under trees
                        output: { bands: 1, sampleType: "FLOAT32" },
                        mosaicking: "ORBIT"
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // Filter by recent data
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude cloud and invalid pixels, keep tree canopy (SCL = 4) for processing
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude clouds, cloud shadows, and water
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB08 = [], validValuesB04 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B08 > 0 && sample.B04 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB08[a] = sample.B08;
                                validValuesB04[a] = sample.B04;
                                validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                                a++;
                            }
                        }
                    }

                    var wst;
                    if (a > 0) {
                        // Calculate WST using valid pixel data
                        var avgB04 = getValue(validValuesB04);
                        var avgB08 = getValue(validValuesB08);
                        var avgB11 = getValue(validValuesB11);

                        // Modified WST calculation to include under-canopy plants using B11
                        wst = (avgB04 - avgB08) / (avgB04 + avgB08 + avgB11);
                    } else {
                        wst = -9999; // No valid data
                    }

                    return [wst];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_npci(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_npci(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_npci(self, npci_array):
        classified_array = np.zeros_like(npci_array, dtype=np.uint8)
        classified_array[(npci_array <= 0) & (npci_array != -9999)] = 1
        classified_array[(npci_array > 0) & (npci_array <= 0.1)] = 2
        classified_array[(npci_array > 0.1) & (npci_array <= 0.2)] = 3
        classified_array[(npci_array > 0.2) & (npci_array <= 0.4)] = 4
        classified_array[(npci_array > 0.4) & (npci_array <= 0.5)] = 5
        classified_array[(npci_array > 0.5) & (npci_array <= 0.6)] = 6
        classified_array[(npci_array > 0.6) & (npci_array <= 0.7)] = 7
        classified_array[(npci_array > 0.7) & (npci_array <= 1)] = 8
        classified_array[(npci_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_npci(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array) 

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates'] 
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Crop Yield Forecast
class CropYieldIndexForecastView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return { 
                        input: ["B04", "B08", "B02", "B11", "SCL"],  // Added B11 for better vegetation detection under trees
                        output: { bands: 1, sampleType: "FLOAT32" }, 
                        mosaicking: "ORBIT" 
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude clouds and other invalid pixels, keep tree canopy (SCL = 4) for processing
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false; // Exclude clouds, cloud shadows, and water
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB08[a] = sample.B08;
                                validValuesB04[a] = sample.B04;
                                validValuesB02[a] = sample.B02;
                                validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                                a++;
                            }
                        }
                    }

                    var arvi;
                    if (a > 0) {
                        var NIR = getValue(validValuesB08);
                        var RED = getValue(validValuesB04);
                        var BLUE = getValue(validValuesB02);
                        var avgB11 = getValue(validValuesB11);

                        // Adjusting ARVI for plants under tree canopies using B11
                        if (avgB11 > 0.3) { // Threshold to identify tree canopies
                            arvi = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE + avgB11));
                        } else {
                            arvi = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
                        }
                    } else {
                        arvi = -9999; // No valid data
                    }

                    return [arvi];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_arvi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_crop_yield(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_arvi(self, arvi_array):
        classified_array = np.zeros_like(arvi_array, dtype=np.uint8)
        classified_array[(arvi_array <= 0) & (arvi_array != -9999)] = 1
        classified_array[(arvi_array > 0) & (arvi_array <= 0.1)] = 2
        classified_array[(arvi_array > 0.1) & (arvi_array <= 0.2)] = 3
        classified_array[(arvi_array > 0.2) & (arvi_array <= 0.4)] = 4
        classified_array[(arvi_array > 0.4) & (arvi_array <= 0.5)] = 5
        classified_array[(arvi_array > 0.5) & (arvi_array <= 0.6)] = 6
        classified_array[(arvi_array > 0.6) & (arvi_array <= 0.7)] = 7
        classified_array[(arvi_array > 0.7) & (arvi_array <= 1)] = 8
        classified_array[(arvi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_crop_yield(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array)

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates']
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        return {
            "type": "FeatureCollection",
            "features": predicted_features
        }


# Disease Weed Forecast
class ARVIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return {
                        input: ["B02", "B04", "B08", "B11", "SCL"],  // Include Blue, Red, NIR, SWIR bands and Scene Classification Layer (SCL)
                        output: {
                            id: "default",
                            bands: 1,
                            sampleType: "FLOAT32"
                        },
                        mosaicking: "ORBIT"  // Use ORBIT-based mosaicking
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);  // Filter scenes from the last 3 months
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);  // Use first quartile as the value
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude cloud and other invalid pixels, keep tree canopy (SCL = 4) for processing
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false;  // Exclude clouds, cloud shadows, and water
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB08 = [], validValuesB04 = [], validValuesB02 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B08 > 0 && sample.B04 > 0 && sample.B02 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB08[a] = sample.B08;
                                validValuesB04[a] = sample.B04;
                                validValuesB02[a] = sample.B02;  
                                validValuesB11[a] = sample.B11;  // Using SWIR for under-canopy plants
                                a++;
                            }
                        }
                    }

                    var ARVI;
                    if (a > 0) {
                        var avgB08 = getValue(validValuesB08);
                        var avgB04 = getValue(validValuesB04);
                        var avgB02 = getValue(validValuesB02);
                        var avgB11 = getValue(validValuesB11);
                        
                        // Adjust ARVI calculation to account for plants under tree canopies
                        if (avgB11 > 0.3) { // Threshold to identify tree canopies
                            // Adjust ARVI for under-canopy plants using SWIR
                            ARVI = (avgB08 - (2 * avgB04 - avgB02)) / (avgB08 + (2 * avgB04 - avgB02) + avgB11);
                        } else {
                            // Regular ARVI calculation
                            ARVI = (avgB08 - (2 * avgB04 - avgB02)) / (avgB08 + (2 * avgB04 - avgB02));
                        }
                    } else {
                        ARVI = -9999;  // No valid data
                    }

                    return [ARVI];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[
                        SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date)),
                    ],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_arvi(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_arvi(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_arvi(self, arvi_array):
        classified_array = np.zeros_like(arvi_array, dtype=np.uint8)
        classified_array[(arvi_array <= 0) & (arvi_array != -9999)] = 1
        classified_array[(arvi_array > 0) & (arvi_array <= 0.1)] = 2
        classified_array[(arvi_array > 0.1) & (arvi_array <= 0.2)] = 3
        classified_array[(arvi_array > 0.2) & (arvi_array <= 0.4)] = 4
        classified_array[(arvi_array > 0.4) & (arvi_array <= 0.5)] = 5
        classified_array[(arvi_array > 0.5) & (arvi_array <= 0.6)] = 6
        classified_array[(arvi_array > 0.6) & (arvi_array <= 0.7)] = 7
        classified_array[(arvi_array > 0.7) & (arvi_array <= 1)] = 8
        classified_array[(arvi_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_arvi(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array)

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates']
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Chlorophyll Forecast
class CARIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []
        
        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})  # Adjust data as needed
            if serializer.is_valid():
                evalscript = """
                function setup() {
                    return {
                        input: ["B03", "B04", "B08", "B11", "SCL"], // Added B11 for better vegetation detection
                        output: {
                            id: "default",
                            bands: 1,
                            sampleType: "FLOAT32"
                        },
                        mosaicking: "ORBIT"
                    };
                }

                function preProcessScenes(collections) {
                    collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                        var orbitDateFrom = new Date(orbit.dateFrom);
                        return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000); // 3-month filter
                    });
                    return collections;
                }

                function getValue(values) {
                    values.sort(function (a, b) { return a - b; });
                    return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                    var index = Math.floor(sortedValues.length / 4);
                    return sortedValues[index];
                }

                function validate(sample) {
                    var scl = sample.SCL;
                    // Exclude clouds, cloud shadows, and water
                    if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {
                        return false;
                    }
                    return true;
                }

                function evaluatePixel(samples, scenes) {
                    var validValuesB08 = [], validValuesB04 = [], validValuesB03 = [], validValuesB11 = [];
                    var a = 0;

                    for (var i = 0; i < samples.length; i++) {
                        var sample = samples[i];
                        if (sample.B08 > 0 && sample.B04 > 0 && sample.B03 > 0 && sample.B11 > 0) {
                            var isValid = validate(sample);
                            if (isValid) {
                                validValuesB08[a] = sample.B08;
                                validValuesB04[a] = sample.B04;
                                validValuesB03[a] = sample.B03;
                                validValuesB11[a] = sample.B11; // Using B11 for vegetation under trees
                                a++;
                            }
                        }
                    }

                    var CARI;
                    if (a > 0) {
                        var GREEN = getValue(validValuesB03);
                        var RED = getValue(validValuesB04);
                        var NIR = getValue(validValuesB08);
                        var SWIR = getValue(validValuesB11); // Incorporating B11

                        // Calculate CARI considering vegetation under trees
                        var term1 = Math.pow((NIR - GREEN) / 150, 2);
                        var term2 = Math.pow((RED - GREEN), 2);
                        CARI = Math.sqrt(term1 + term2);

                        // Adjust CARI calculation for conditions under tree canopies
                        if (SWIR > 0.3) { // Threshold for identifying vegetation under canopies
                            CARI *= 1.1; // Example adjustment factor
                        }
                    } else {
                        CARI = -9999; // No valid data
                    }

                    return [CARI];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue 

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_cari(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_cari(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_cari(self, cari_array):
        classified_array = np.zeros_like(cari_array, dtype=np.uint8)
        classified_array[(cari_array <= 0)] = 1
        classified_array[(cari_array > 0) & (cari_array <= 0.1)] = 2
        classified_array[(cari_array > 0.1) & (cari_array <= 0.2)] = 3
        classified_array[(cari_array > 0.2) & (cari_array <= 0.4)] = 4
        classified_array[(cari_array > 0.4) & (cari_array <= 0.5)] = 5
        classified_array[(cari_array > 0.5) & (cari_array <= 0.6)] = 6
        classified_array[(cari_array > 0.6) & (cari_array <= 0.7)] = 7
        classified_array[(cari_array > 0.7)] = 8
        classified_array[(cari_array == -9999)] = 0  # Set cloudy pixels to 0
        return classified_array

    def predict_cari(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array) 

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates'] 
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


# Chlorophyll Growth Forecast
class MCARIFView(APIView):
    def post(self, request):
        geojson_collection = request.data.get('features')
        if not geojson_collection:
            return Response({'error': 'GeoJSON feature collection is required.'}, status=status.HTTP_400_BAD_REQUEST)

        results = []

        for feature in geojson_collection:
            geojson_polygon = feature['geometry']
            date = feature['properties'].get('date')

            if not geojson_polygon or not date:
                continue

            try:
                polygon = shape(geojson_polygon)
            except Exception as e:
                return Response({'error': 'Invalid GeoJSON polygon.'}, status=status.HTTP_400_BAD_REQUEST)

            bbox = BBox(bbox=polygon.bounds, crs=CRS.WGS84)
            serializer = IndicesSerializer(data={'date': date})
            if serializer.is_valid():
                evalscript = """
                function setup() {
                return {
                    input: [{
                    bands: [
                        "B04",
                        "B03",
                        "B02",
                        "SCL"
                    ]
                    }],
                    output: { bands: 3, sampleType: "UINT16" },
                    mosaicking: "ORBIT"
                };
                }

                function preProcessScenes(collections) {
                collections.scenes.orbits = collections.scenes.orbits.filter(function (orbit) {
                    var orbitDateFrom = new Date(orbit.dateFrom)
                    return orbitDateFrom.getTime() >= (collections.to.getTime() - 3 * 31 * 24 * 3600 * 1000);
                })
                return collections
                }

                function getValue(values) {
                values.sort(function (a, b) { return a - b; });
                return getFirstQuartile(values);
                }

                function getFirstQuartile(sortedValues) {
                var index = Math.floor(sortedValues.length / 4);
                return sortedValues[index];
                }

                function validate(samples) {
                var scl = samples.SCL;

                if (scl === 3 || scl === 9 || scl === 8 || scl === 7 || scl === 10 || scl === 11 || scl === 1) {
                    return false;
                }
                return true;
                }

                function evaluatePixel(samples) {
                var clo_b02 = []; var clo_b03 = []; var clo_b04 = [];
                var a = 0;

                for (var i = 0; i < samples.length; i++) {
                    var sample = samples[i];

                    if (sample.B02 > 0 && sample.B03 > 0 && sample.B04 > 0) {
                    var isValid = validate(sample);

                    if (isValid) {
                        clo_b02[a] = sample.B02;
                        clo_b03[a] = sample.B03;
                        clo_b04[a] = sample.B04;
                        a = a + 1;
                    }
                    }
                }

                var rValue, gValue, bValue;
                if (a > 0) {
                    rValue = getValue(clo_b04);
                    gValue = getValue(clo_b03);
                    bValue = getValue(clo_b02);
                } else {
                    rValue = gValue = bValue = 0; // No valid data case
                }
                return [rValue * 10000, gValue * 10000, bValue * 10000];
                }
                """

                sentinel_request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A, time_interval=(date, date))],
                    responses=[SentinelHubRequest.output_response('default', MimeType.TIFF)],
                    bbox=bbox,
                    size=[512, 354.253],
                    config=config,
                )

                response = sentinel_request.get_data()[0]

                if np.all(response == -9999):
                    continue

                transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

                classified_image = self.reclassify_mcari(response)
                shapes_gen = shapes(classified_image, mask=None, transform=transform)
                geometries = list(shapes_gen)

                features = [{"type": "Feature", "geometry": geom, "properties": {"class_no": value}} for geom, value in geometries if value != 0]
                geojson_data = {"type": "FeatureCollection", "features": features}

                geojson_polygon_df = gpd.GeoDataFrame(geometry=[polygon], crs='epsg:4326')
                geojson_data_df = gpd.GeoDataFrame.from_features(geojson_data, crs='epsg:4326')
                intersection_df = gpd.overlay(geojson_data_df, geojson_polygon_df)
                intersection_geojson = intersection_df.to_json()

                results.append(json.loads(intersection_geojson))

        predicted_results = self.predict_mcari(results)
        return Response(predicted_results, status=status.HTTP_200_OK)

    def reclassify_mcari(self, mcari_array):
        classified_array = np.zeros_like(mcari_array, dtype=np.uint8)
        classified_array[(mcari_array <= 0) & (mcari_array != -9999)] = 1
        classified_array[(mcari_array > 0) & (mcari_array <= 0.1)] = 2
        classified_array[(mcari_array > 0.1) & (mcari_array <= 0.2)] = 3
        classified_array[(mcari_array > 0.2) & (mcari_array <= 0.4)] = 4
        classified_array[(mcari_array > 0.4) & (mcari_array <= 0.5)] = 5
        classified_array[(mcari_array > 0.5) & (mcari_array <= 0.6)] = 6
        classified_array[(mcari_array > 0.6) & (mcari_array <= 0.7)] = 7
        classified_array[(mcari_array > 0.7) & (mcari_array <= 1)] = 8
        classified_array[(mcari_array == -9999)] = 0  # Cloudy pixels set to 0
        return classified_array

    def predict_mcari(self, results):
        predicted_features = []
        valid_coordinates = []
        valid_class_numbers = []
        valid_results = []

        if not results or not isinstance(results, list):
            print("No results provided or results are not in the expected format.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features 
            }

        for feature in results:
            if 'features' not in feature:
                print("Feature missing 'features' key:", feature)
                continue

            for item in feature['features']:
                geometry = item.get('geometry', {})
                coords = geometry.get('coordinates', [])

                if isinstance(coords, list) and coords and isinstance(coords[0], list):
                    if len(coords[0]) > 0 and len(coords[0][0]) == 2:
                        valid_coord = coords[0][0]
                        valid_coordinates.append(valid_coord)
                        valid_results.append(item)

                        class_no = item.get('properties', {}).get('class_no')
                        if class_no is not None:
                            valid_class_numbers.append(class_no)
                        else:
                            print("Class number missing in properties.")
                    else:
                        print(f"Invalid coordinate structure found: {coords}")
                else:
                    print(f"Invalid coordinate found: {coords}")

        if not valid_coordinates or not valid_class_numbers:
            print("No valid coordinates or class numbers found.")
            return {
                "type": "FeatureCollection",
                "features": predicted_features
            }

        coordinates_array = np.array(valid_coordinates)
        class_numbers_array = np.array(valid_class_numbers)

        model = LinearRegression()
        model.fit(coordinates_array, class_numbers_array)

        max_sample_size = min(1000, len(coordinates_array))
        sampled_indices = random.sample(range(len(coordinates_array)), max_sample_size)

        sampled_coordinates = coordinates_array[sampled_indices]
        predicted_class_no = model.predict(sampled_coordinates).astype(int)

        predicted_features = [
            {
                "id": str(i),
                "type": "Feature",
                "properties": {
                    "class_no": class_no
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": valid_results[idx]['geometry']['coordinates']
                }
            }
            for i, (idx, class_no) in enumerate(zip(sampled_indices, predicted_class_no))
        ]

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson





