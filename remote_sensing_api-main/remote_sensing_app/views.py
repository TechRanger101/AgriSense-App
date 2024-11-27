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


#Vegetation Health
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
              return { input: ["B04", "B08", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              return [(sample.B08 - sample.B04) / (sample.B08 + sample.B04)];
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


#Humidity level
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
              return { input: ["B03", "B08", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              return [(sample.B03 - sample.B08) / (sample.B03 + sample.B08)];
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


#Plant Moisture
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
              return { input: ["B08", "B11", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              return [(sample.B08 - sample.B11) / (sample.B08 + sample.B11)];
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


#Coffee Ripeness
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
                input: ["B04", "CLM"],
                output: {bands: 1, sampleType: "FLOAT32"}
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              return [sample.B04];
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


#Ground Temperature
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
              { datasource: "S3OLCI", bands: ["B06", "B08", "B17"] }],
            output: [
              { id: "default", bands: 3, sampleType: SampleType.AUTO }
            ],
            mosaicking: "ORBIT"
          }
        }

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
          var LSTmax = -999;
          var LSTavg = 0;
          var LSTstd = 0;
          var reduceNavg = 0;
          var N = samples.S3SLSTR.length;
          var LSTarray = [];

          for (let i = 0; i < N; i++) {
            var Bi = samples.S3SLSTR[i].S8;
            var B06i = samples.S3OLCI[i].B06;
            var B08i = samples.S3OLCI[i].B08;
            var B17i = samples.S3OLCI[i].B17;

            if ((Bi <= 173 || Bi >= 65000) || (B06i <= 0 || B08i <= 0 || B17i <= 0)) {
              ++reduceNavg;
              continue;
            }

            var S8BTi = Bi - 273.15;
            var NDVIi = (B17i - B08i) / (B17i + B08i);
            var PVi = Math.pow(((NDVIi - NDVIs) / (NDVIv - NDVIs)), 2);
            var LSEi = LSEcalc(NDVIi, PVi);
            var LSTi = (S8BTi / (1 + (((bCent * S8BTi) / rho) * Math.log(LSEi))));

            LSTavg = LSTavg + LSTi;
            if (LSTi > LSTmax) { LSTmax = LSTi; }
            LSTarray.push(LSTi);
          }
          N = N - reduceNavg;
          LSTavg = LSTavg / N;
          for (let i = 0; i < LSTarray.length; i++) {
            LSTstd = LSTstd + (Math.pow(LSTarray[i] - LSTavg, 2));
          }
          LSTstd = (Math.pow(LSTstd / (LSTarray.length - 1), 0.5));
          let outLST = (option == 0) ? LSTavg : (option == 1) ? LSTmax : LSTstd;
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
                input: ["B04", "B08", "CLM"],
                output: {
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              //return [(sample.B04 - sample.B05) / (sample.B04 + sample.B05)];
              var npci = (sample.B04 - sample.B08) / (sample.B04 + sample.B08);
              return [npci];
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


#Crop Yield
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
                input: ["B04", "B08", "B02", "CLM"],
                output: {
                  id: "default",
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              var NIR = sample.B08;
              var RED = sample.B04;
              var BLUE = sample.B02;

              // Calculate ARVI
              var ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));

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


class TVIView(APIView):
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
                input: ["B04", "B08", "B03", "CLM"],
                output: {
                  id: "default",
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              var NIR = sample.B08;
              var RED = sample.B04;
              var GREEN = sample.B03;

              // Calculate ARVI
              var TVI1 = 120 * (NIR - GREEN)
              var TVI2 = 200 * (NIR - RED)
              var TVI = 0.5 *(TVI1 - TVI2)

              return [TVI];
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

class EVIView(APIView):
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
                input: ["B04", "B08", "B02", "CLM"],
                output: {
                  id: "default",
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              var NIR = sample.B08;
              var RED = sample.B04;
              var BLUE = sample.B02;

              // Calculate ARVI
              var EVI = 2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1));

              return [EVI];
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
            //VERSION=3
            function setup() {
              return {
                input: ["B02", "B04", "B08", "CLM"],  // Include Blue, Red, and NIR bands
                output: {
                  id: "default",
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              var NIR = sample.B08;
              var RED = sample.B04;
              var BLUE = sample.B02;

              // Calculate Atmospherically Resistant Vegetation Index (ARVI)
              var ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
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

            if np.all(response == -9999):
                return Response({'error': 'No valid data available for the given date and area. Try adjusting the date or area.'}, status=status.HTTP_404_NOT_FOUND)

            transform = rasterio.transform.from_bounds(*bbox, response.shape[1], response.shape[0])

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
            //VERSION=3
            function setup() {
              return {
                input: ["B03", "B04", "B08", "CLM"],  // Green, Red, and NIR bands
                output: {
                  id: "default",
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              var GREEN = sample.B03;
              var RED = sample.B04;
              var NIR = sample.B08;

              // Calculate Chlorophyll Absorption Ratio Index (CARI)
              var term1 = Math.pow((NIR - GREEN) / 150, 2);
              var term2 = Math.pow((RED - GREEN), 2);
              var CARI = Math.sqrt(term1 + term2);

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
                input: ["B02", "B03", "B04", "B08", "CLM"],  // Blue, Green, Red, and NIR bands
                output: {
                  id: "default",
                  bands: 1,
                  sampleType: "FLOAT32"
                }
              };
            }
            function evaluatePixel(sample) {
              if (sample.CLM == 1) {
                return [-9999];
              }
              var BLUE = sample.B02;
              var GREEN = sample.B03;
              var RED = sample.B04;
              var NIR = sample.B08;

              // Calculate Modified Chlorophyll Absorption in Reflectance Index (MCARI)
              var mcari = (RED - GREEN) - 0.2 * (RED - BLUE) * (RED / NIR);

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


#Vegetation Health Forecast
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
                  return { input: ["B04", "B08", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  return [(sample.B08 - sample.B04) / (sample.B08 + sample.B04)];
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
                  return { input: ["B08", "B11", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  return [(sample.B08 - sample.B11) / (sample.B08 + sample.B11)];
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


#Plant Moisture Forecast
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
                return { input: ["B08", "B11", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
            }
            function evaluatePixel(sample) {
                if (sample.CLM == 1) {
                    return [-9999];
                }
                return [(sample.B08 - sample.B11) / (sample.B08 + sample.B11)];
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
                  return { input: ["B04", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  return [sample.B04]; // Using B04 band to estimate ripeness
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
                        a = 3 #print(f"Invalid coordinate structure found: {coords}")
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

            # Create and send SentinelHubRequest
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
            let viz = ColorGradientVisualizer.createRedTemperature(minC, maxC);

            function setup() {
                return {
                    input: [
                        { datasource: "S3SLSTR", bands: ["S8"] },
                        { datasource: "S3OLCI", bands: ["B06", "B08", "B17"] }
                    ],
                    output: [
                        { id: "default", bands: 3, sampleType: SampleType.AUTO }
                    ],
                    mosaicking: "ORBIT"
                }
            }

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
                var LSTmax = -999;
                var LSTavg = 0;
                var N = samples.S3SLSTR.length;
                var LSTarray = [];

                for (let i = 0; i < N; i++) {
                    var Bi = samples.S3SLSTR[i].S8;
                    if (Bi <= 173 || Bi >= 65000) continue;

                    var S8BTi = Bi - 273.15;
                    var NDVIi = (samples.S3OLCI[i].B17 - samples.S3OLCI[i].B08) / (samples.S3OLCI[i].B17 + samples.S3OLCI[i].B08);
                    var PVi = Math.pow(((NDVIi - NDVIs) / (NDVIv - NDVIs)), 2);
                    var LSEi = LSEcalc(NDVIi, PVi);
                    var LSTi = (S8BTi / (1 + (((bCent * S8BTi) / rho) * Math.log(LSEi))));

                    LSTavg += LSTi;
                    LSTarray.push(LSTi);
                }

                LSTavg /= N;
                return viz.process(LSTavg);
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
                  return { input: ["B04", "B08", "CLM"], output: { bands: 1, sampleType: "FLOAT32" } };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  // Water Stress Index (Normalized Pigment Chlorophyll Index - NPCI)
                  var npci = (sample.B04 - sample.B08) / (sample.B04 + sample.B08);
                  return [npci];
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
                    input: ["B04", "B08", "B02", "CLM"],
                    output: {
                      id: "default",
                      bands: 1,
                      sampleType: "FLOAT32"
                    }
                  };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  var NIR = sample.B08;
                  var RED = sample.B04;
                  var BLUE = sample.B02;

                  // Calculate ARVI (Atmospherically Resistant Vegetation Index)
                  var ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
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

        predicted_geojson = {
            "type": "FeatureCollection",
            "features": predicted_features
        }

        return predicted_geojson


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
                    input: ["B02", "B04", "B08", "CLM"],
                    output: {
                      id: "default",
                      bands: 1,
                      sampleType: "FLOAT32"
                    }
                  };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  var NIR = sample.B08;
                  var RED = sample.B04;
                  var BLUE = sample.B02;

                  // Calculate Atmospherically Resistant Vegetation Index (ARVI)
                  var ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE));
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
                    input: ["B03", "B04", "B08", "CLM"],  // Green, Red, and NIR bands
                    output: { bands: 1, sampleType: "FLOAT32" }
                  };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  var GREEN = sample.B03;
                  var RED = sample.B04;
                  var NIR = sample.B08;
                  var term1 = Math.pow((NIR - RED), 2);
                  var term2 = Math.pow((GREEN - RED), 2);
                  var CARI = term1 / Math.sqrt(term2);
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
                    input: ["B02", "B03", "B04", "B08", "CLM"],
                    output: { bands: 1, sampleType: "FLOAT32" }
                  };
                }
                function evaluatePixel(sample) {
                  if (sample.CLM == 1) {
                    return [-9999];
                  }
                  var BLUE = sample.B02;
                  var GREEN = sample.B03;
                  var RED = sample.B04;
                  var NIR = sample.B08;
                  
                  // Calculate MCARI
                  return [(1.5 * (2.5 * (NIR - RED) - 1.3 * (NIR - GREEN)))];
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




