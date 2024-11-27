# User Guide: Crop and Pest Analysis API

## Overview

The Crop and Pest Analysis API utilizes satellite data to provide insights into crop health and pest risks. It processes data related to vegetation indices, surface temperatures, and atmospheric conditions, helping users understand agricultural risks and conditions.

## Authentication

The Crop and Pest Analysis API provides secure access through a simple signup and signin process.

Before making any requests to the API, users must authenticate to gain access. This involves signing up and signing in to receive a JWT token used in subsequent API requests.

Use the given credentials to sign in and receive a token. Implement the process using the provided authentication script or a similar tool.

## Making a Request

### Single Location Analysis

1. **Obtain JWT Token**: Ensure you have your JWT token from the signin process.

2. **API Endpoint**: Access data using:
   ```text
   http://localhost:5000/api/calculate
   ```

3. **Include Headers**: Insert your JWT token into the request headers.
   ```python
   headers = {
       'Authorization': f'Bearer {token}'
   }
   ```

4. **Set Query Parameters**: Provide the parameters for the request:
   - `latitude`, `longitude`: Coordinates of interest.
   - `start_date`, `end_date`: Date range for analysis.

5. **Send GET Request**:
   ```python
   import requests
   
   params = {
       'latitude': '29.346443',
       'longitude': '-95.152565',
       'start_date': '2024-05-01',
       'end_date': '2024-05-31'
   }
   response = requests.get('http://localhost:5000/api/calculate', headers=headers, params=params)
   ```

### Multiple Location Analysis

1. **Obtain JWT Token**: Ensure you have your JWT token from the signin process.

2. **API Endpoint**: Use the enhanced API route for multiple locations:
   ```text
   http://localhost:5000/api/calculate-multi-locations
   ```

3. **Prepare Locations**: Encode multiple locations in a JSON format.
   ```python
   import json
   
   locations = [
       {'latitude': '29.346443', 'longitude': '-95.152565', 'start_date': '2024-05-01', 'end_date': '2024-05-31'},
       {'latitude': '40.7128', 'longitude': '-74.0060', 'start_date': '2024-04-01', 'end_date': '2024-04-30'}
   ]
   ```

4. **Send Request with Multiple Locations**:
   ```python
   params = {
       'locations': json.dumps(locations)
   }
   response = requests.get('http://localhost:5000/api/calculate-multi-locations', headers=headers, params=params)
   ```

## Interpreting the Response

The API returns a JSON object containing insights into various environmental conditions. Key components include atmospheric data, insights, vegetation indices, and land surface temperature (LST).

### Example Response

```json
{
  "result": [{
    "Atmospheric": { "NO2": 0.0, "O3": 0.021176470588235293, "SO2": 0.0 },
    "Insights": {
      "Air Quality": "Good",
      "Crop Health": "Good",
      "Drought Risk": "High",
      "Fire Prevention": "Safe",
      "Flood Risk": "Low",
      "Pest Risk": "Low"
    },
    "LST": {
      "max": 224.6300048828125,
      "mean": 223.29843139648438,
      "min": 221.8699951171875
    },
    "NDMI": {
      "mean": 0.193851500749588,
      "median": 0.21999868750572205,
      "std_dev": 0.16626955568790436
    },
    "NDVI": {
      "mean": 0.4581974446773529,
      "median": 0.4922584891319275,
      "std_dev": 0.1992798149585724
    },
    "NDWI": {
      "mean": -0.3635707199573517,
      "median": -0.3596433401107788,
      "std_dev": 0.1484735906124115
    }
  }],
  "status": "OK"
}
```

### Key Response Elements

- **Insights**: Offers assessments on air quality, crop health, drought risk, fire prevention, flood risk, and pest risk.
- **Atmospheric Data**: Presents levels of NO2, O3, and SO2.
- **Indices (NDMI, NDVI, NDWI)**: Provides mean, median, and standard deviation for each index, indicating vegetation health and water content.
- **LST (Land Surface Temperature)**: Provides maximum, mean, and minimum values.

By analyzing these results, users can derive valuable insights into environmental conditions affecting crops and pests in a given area. Adjust the request parameters to refine search results based on geographic and temporal specificity.

### Insights Available Status

- **Pest Risk**:

  - **Low**: Unfavorable conditions for pest development.
  - **Medium**: Certain conditions exist that may allow pest activity.
  - **High**: Environment highly conducive to pest proliferation, requiring precautionary measures.

- **Crop Health**:

  - **Very Good**: Optimal conditions for crop growth; indicates healthy vegetation.
  - **Good**: Satisfactory growth conditions with minor stress factors.
  - **Average**: Conditions indicate moderate health; mixed signals on vegetation vitality.
  - **Low**: Conditions not supportive of healthy crop growth, indicating significant stress.

- **Air Quality**:

  - **Good**: Clean air with minimal to no pollution; safe for crops and human activity.
  - **Average**: Some levels of pollutants present; might affect sensitive crops.
  - **Poor**: High levels of pollution, potentially harmful to plant health and humans.

- **Fire Prevention**:

  - **Safe**: Low risk of fire; conditions are not favorable for fire ignition or spread.
  - **Moderate Risk**: Possible risk of fire under certain conditions; precaution advised.
  - **High Risk**: Conditions highly favorable for fire outbreaks requiring immediate attention.

- **Drought Risk**:

  - **Low**: Adequate water availability; low risk of drought stress.
  - **Medium**: Moderate levels of water deficiency; potential risk for crops.
  - **High**: Significant lack of water; severe drought conditions anticipated.

- **Flood Risk**:
  - **Low**: Low likelihood of flooding; typical water saturation.
  - **Medium**: Potential flooding risk; requires monitoring.
  - **High**: High potential for flooding; immediate mitigation measures needed.

### Indices Used and Their Ranges

1. **NDVI (Normalized Difference Vegetation Index)**:

   - **Purpose**: Measures vegetation health and density.
   - **Typical Range**: -1 to 1
     - **Interpretation**:
       - Values close to 1 indicate dense, healthy vegetation (e.g., forests, crops).
       - Values around 0 suggest barren or sparse vegetation (e.g., urban areas, rock surfaces).
       - Negative values indicate water, snow, or clouds.

2. **NDWI (Normalized Difference Water Index)**:

   - **Purpose**: Assesses water content in vegetation and soil.
   - **Typical Range**: -1 to 1
     - **Interpretation**:
       - Positive values usually indicate high water content in vegetation.
       - Values near 0 or negative suggest low moisture content, possibly indicating drought conditions or non-vegetated surfaces.

3. **NDMI (Normalized Difference Moisture Index)**:

   - **Purpose**: Evaluates vegetation moisture content.
   - **Typical Range**: -1 to 1
     - **Interpretation**:
       - Values closer to 1 indicate higher moisture levels in the vegetation.
       - Near-zero values indicate moderate to low moisture content.
       - Negative values may indicate dry conditions or non-vegetated areas.

4. **LST (Land Surface Temperature)**:
   - **Purpose**: Measures surface temperature, influencing plant stress.
   - **Typical Range**: Varies widely based on geographic and climatic conditions, commonly:
     - Ranges from below 250 Kelvin (cold areas) to above 320 Kelvin (hot desert regions).
     - **Interpretation**:
       - High LST can indicate stress conditions for plants (e.g., heat stress).
       - Low LST is typical in cooler climates or during nighttime.

### Atmospheric Parameters and Their Typical Ranges

1. **Atmospheric NO2 (Nitrogen Dioxide Concentration)**:

   - **Purpose**: Indicator of air quality, often from combustion processes.
   - **Typical Range** (in parts per million, ppm): 0 to 0.1 ppm in typical environments.
     - **Interpretation**:
       - Low concentrations indicate cleaner air.
       - Higher concentrations can indicate polluted conditions.

2. **Atmospheric O3 (Ozone Concentration)**:

   - **Purpose**: An important factor in atmospheric chemistry, affecting air quality.
   - **Typical Range** (in parts per million, ppm): 0.02 to 0.1 ppm for ground-level ozone.
     - **Interpretation**:
       - Moderate levels may slightly affect sensitive plants.
       - High levels can be harmful to both human health and plant life.

3. **Atmospheric SO2 (Sulfur Dioxide Concentration)**:
   - **Purpose**: Generally a result of industrial activity; can affect plant health.
   - **Typical Range** (in parts per million, ppm): 0 to 0.05 ppm in most environments.
     - **Interpretation**:
       - Low or zero levels are ideal.
       - Elevated levels can be harmful to vegetation and human health.

Understanding these indices and their ranges helps interpret environmental data accurately, guiding decision-making in agricultural management and environmental conservation.
