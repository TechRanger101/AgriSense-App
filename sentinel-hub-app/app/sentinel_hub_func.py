from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions
import numpy as np
import os


# Configure Sentinel Hub instance
config = SHConfig()
config.sh_client_id = os.getenv('SH_CLIENT_ID')
config.sh_client_secret = os.getenv('SH_CLIENT_SECRET')


# Function to calculate indices from Sentinel-2
def calculate_optical_indices(latitude, longitude, start_date, end_date):  
    resolution = 10  
    buffer_distance = np.sqrt(20917) / 111320  # Approx. conversion for meter to degree  
    bbox = BBox((longitude - buffer_distance, latitude - buffer_distance,  
                 longitude + buffer_distance, latitude + buffer_distance), CRS.WGS84)  
    time_interval = (start_date, end_date)  

    evalscript = '''  
    //VERSION=3  
    
    function setup() {  
        return {  
            input: [{  
                bands: ["B02", "B03", "B04", "B08", "B11", "SCL"]  
            }],  
            output: {  
                bands: 3,  
                sampleType: "FLOAT32"  
            }  
        };  
    }  

    function validate(sample) {  
        var scl = sample.SCL;  
        // Using SCL to filter out clouds and invalid pixels  
        if (scl === 3 || scl === 9 || scl === 8 || scl === 10 || scl === 11 || scl === 1) {  
            return false; // Exclude cloud and cloud shadow pixels  
        }  
        return true;  
    }  

    function evaluatePixel(sample) {  
        if (!validate(sample)) return [NaN, NaN, NaN];  

        // Extract band values  
        var B04 = sample.B04;  
        var B08 = sample.B08;  
        var B11 = sample.B11;  

        // NDVI under canopy calculation  
        var ndvi;  
        if (B11 > 0.3) { // Threshold to identify tree canopies  
            ndvi = (B08 - B04) / (B08 + B04 + B11);  
        } else {  
            ndvi = (B08 - B04) / (B08 + B04);  
        }  

        // NDWI calculation  
        var ndwi = (sample.B03 - B08) / (sample.B03 + B08);  

        // NDMI under canopy calculation  
        var ndmi;  
        if (B04 > 0.3) { // Threshold to detect vegetation under tree canopies  
            ndmi = (B08 - B11) / (B08 + B11 + B04);  
        } else {  
            ndmi = (B08 - B11) / (B08 + B11);  
        }  

        return [ndvi, ndwi, ndmi];  
    }  
    '''  

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=resolution),
        config=config
    )

    response = request.get_data()
    if not response:
        return None

    indices = response[0]

    # Use numpy for quick statistics
    ndvi = indices[..., 0]
    ndwi = indices[..., 1]
    ndmi = indices[..., 2]

    # Summary statistics for each index
    return {
        'NDVI': {
            'mean': float(np.mean(ndvi)),
            'median': float(np.median(ndvi)),
            'std_dev': float(np.std(ndvi))
        },
        'NDWI': {
            'mean': float(np.mean(ndwi)),
            'median': float(np.median(ndwi)),
            'std_dev': float(np.std(ndwi))
        },
        'NDMI': {
            'mean': float(np.mean(ndmi)),
            'median': float(np.median(ndmi)),
            'std_dev': float(np.std(ndmi))
        }
    }


# Function to retrieve LST from Sentinel-3
def retrieve_lst_from_sentinel3(latitude, longitude, start_date, end_date):
    bbox = BBox(bbox=(longitude-0.1, latitude-0.1,
                longitude+0.1, latitude+0.1), crs=CRS.WGS84)
    resolution = 500  # Resolution in meters
    time_interval = (start_date, end_date)

    evalscript = '''
    //VERSION=3  
    function setup() {  
        return {  
            input: [{  
                bands: ['S8', 'S9']  
            }],  
            output: {  
                bands: 2,  
                sampleType: 'FLOAT32'  
            }  
        };  
    }  

    function evaluatePixel(sample) {  
        return [sample.S8, sample.S9];  
    }  
    '''

    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL3_SLSTR,
                time_interval=time_interval,
            )
        ],
        responses=[
            SentinelHubRequest.output_response('default', MimeType.TIFF)
        ],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution),
        config=config
    )

    results = request.get_data()

    thermal_data = results[0]

    s8_band = thermal_data[..., 0]
    s9_band = thermal_data[..., 1]

    # Calculate statistics
    lst_mean = float(np.mean((s8_band + s9_band) / 2))
    lst_min = float(np.min((s8_band + s9_band) / 2))
    lst_max = float(np.max((s8_band + s9_band) / 2))

    return {
        'LST': {
            'mean': lst_mean,
            'min': lst_min,
            'max': lst_max
        }
    }


# Function to retrieve O3, NO2 and SO2 from Sentinel-5P
def retrieve_atmospheric_data(latitude, longitude, start_date, end_date):
    resolution = 512
    bbox = BBox((longitude-0.01, latitude-0.01,
                longitude+0.01, latitude+0.01), CRS.WGS84)
    time_interval = (start_date, end_date)

    evalscripts = {
        'NO2': '''  
        //VERSION=3
        const band = 'NO2';
        var minVal = 0.0;
        var maxVal = 0.0001;
        
        function setup() {
          return {
            input: [band, 'dataMask'],
            output: {
              bands: 4,
            },
          };
        }
        
        var viz = ColorRampVisualizer.createBlueRed(minVal, maxVal);
        
        function evaluatePixel(samples) {
          let ret = viz.process(samples[band]);
          ret.push(samples.dataMask);
          return ret;
        }
        ''',
        'O3': '''  
        //VERSION=3
        const band = 'O3';
        var minVal = 0.0;
        var maxVal = 0.36;
        
        function setup() {
          return {
            input: [band, 'dataMask'],
            output: {
              bands: 4,
            },
          };
        }
        
        var viz = ColorRampVisualizer.createBlueRed(minVal, maxVal);
        
        function evaluatePixel(samples) {
          let ret = viz.process(samples[band]);
          ret.push(samples.dataMask);
          return ret;
        }
        ''',
        'SO2': '''  
        //VERSION=3
        const band = 'SO2';
        var minVal = 0.0;
        var maxVal = 0.01;
        
        function setup() {
          return {
            input: [band, 'dataMask'],
            output: {
              bands: 4,
            },
          };
        }
        
        var viz = ColorRampVisualizer.createBlueRed(minVal, maxVal);
        
        function evaluatePixel(samples) {
          let ret = viz.process(samples[band]);
          ret.push(samples.dataMask);
          return ret;
        }
        '''
    }

    max_values = {
        'NO2': 0.0001,
        'O3': 0.36,
        'SO2': 0.01
    }

    atmospheric_data = {}

    for pollutant, evalscript in evalscripts.items():
        request = SentinelHubRequest(
            data_folder='.',
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL5P,
                    time_interval=time_interval,
                )
            ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
            ],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, resolution=resolution),
            config=config
        )

        response = request.get_data()

        if response:
            # Extract concentration based on the maximum scaling value
            data = np.array(response[0])
            concentration = (data[..., 0] / 255.0) * max_values[pollutant]
            # Example: Mean of concentration
            atmospheric_data[pollutant] = concentration.mean()
        else:
            atmospheric_data[pollutant] = None

    return {
        'Atmospheric': {
            'NO2': atmospheric_data['NO2'],
            'O3': atmospheric_data['O3'],
            'SO2': atmospheric_data['SO2']
        }
    }


# Function to calculate additional agronomic indicators
def calculate_additional_info(optical_indices, lst_data, atmospheric_data):
    # Variables for the additional indicators
    pest_risk = 'Low'
    crop_health = 'Very Good'
    air_quality = 'Good'
    fire_prevention = 'Safe'
    drought_risk = 'Low'
    flood_risk = 'Low'

    # Calculate average LST and NDVI mean
    avg_lst = lst_data['LST']['mean']
    ndvi_mean = optical_indices['NDVI']['mean']
    ndwi_mean = optical_indices['NDWI']['mean']

    # Pest risk assessment
    if avg_lst > 300 and ndvi_mean > 0.3:
        pest_risk = 'High'
    elif avg_lst > 280 and ndvi_mean > 0.2:
        pest_risk = 'Medium'

    # Crop health assessment
    if ndvi_mean > 0.3 and ndwi_mean > 0:
        crop_health = 'Very Good'
    elif ndvi_mean > 0.2:
        crop_health = 'Good'
    elif ndvi_mean > 0.1:
        crop_health = 'Average'
    else:
        crop_health = 'Low'

    # Air quality assessment based on exemplary thresholds
    avg_no2 = atmospheric_data['Atmospheric']['NO2']
    if avg_no2 > 100:
        air_quality = 'Poor'
    elif avg_no2 > 50:
        air_quality = 'Average'

    # Fire prevention risk assessment
    if avg_lst > 320:
        fire_prevention = 'High Risk'
    elif avg_lst > 300:
        fire_prevention = 'Moderate Risk'

    # Drought risk assessment using NDWI
    if ndwi_mean < -0.3:
        drought_risk = 'High'
    elif ndwi_mean < -0.2:
        drought_risk = 'Medium'

    # Flood risk assessment using NDWI
    if ndwi_mean > 0.3:
        flood_risk = 'High'
    elif ndwi_mean > 0.2:
        flood_risk = 'Medium'

    return {
        'Insights': {
            'Pest Risk': pest_risk,
            'Crop Health': crop_health,
            'Air Quality': air_quality,
            'Fire Prevention': fire_prevention,
            'Drought Risk': drought_risk,
            'Flood Risk': flood_risk
        }
    }


# Function to get all info from sentinel hub
def get_all_crop_and_pest_info(latitude, longitude, start_date, end_date):
    optical_indices = calculate_optical_indices(
        latitude, longitude, start_date, end_date)
    lst_data = retrieve_lst_from_sentinel3(
        latitude, longitude, start_date, end_date)
    atmospheric_data = retrieve_atmospheric_data(
        latitude, longitude, start_date, end_date)

    if any(x is None for x in [optical_indices, lst_data, atmospheric_data]):
        return {'error': 'No data available for the specified parameters', 'status': 404}

    additional_info = calculate_additional_info(
        optical_indices, lst_data, atmospheric_data)

    return {**optical_indices, **lst_data, **atmospheric_data, **additional_info}
