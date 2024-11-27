# User Guide: Coffee Farmers Information API

## Overview

The Coffee Farmers Information API provides access to geographical data of coffee farmers' locations. By leveraging this API, users can retrieve the latitude, longitude, country, region, and city information for coffee farms, aiding in understanding distribution and location-specific details of coffee production areas.

## Making a Request

To access the Coffee Farmers Information API:

### Obtain a JWT Token

Use the given credentials to sign in and receive a token. Implement the process using the provided authentication script or a similar tool.

### API Endpoint

Access the following endpoint for data requests:

```text
http://localhost:5000/api/coffee-farmers
```

### Request Headers

Include the JWT token in the headers for authorization:

```python
headers = {
    'Authorization': 'Bearer your-jwt-token-here'
}
```

### Sending the Request

To interact with the API and obtain location details:

1. Use an HTTP library, like `requests` in Python, to initiate a GET request to the provided URL.

   ```python
   import requests

   # Define endpoint
   url = 'http://localhost:5000/api/coffee-farmers/'

   # Perform GET request
   response = requests.get(url)

   # Check response status
   if response.status_code == 200:
       print('Response JSON:', response.json())
   else:
       print('Error:', response.status_code)
   ```

## Interpreting the Response

Upon requesting data, the API returns a JSON object. Here is a breakdown of the response structure:

### Example Response

```json
{
  "status": "success",
  "locations": [
    {
      "latitude": 9.0567,
      "longitude": 7.4969,
      "country": "Test",
      "region": "TS",
      "city": "TST"
    }
  ]
}
```

### Key Response Elements

- **Status**: Indicates the result of the request. `"success"` signifies successful data retrieval.

- **Locations**: An array containing objects with specified geographic details:
  - **Latitude**: The latitude coordinate of the coffee farm location.
  - **Longitude**: The longitude coordinate of the coffee farm location.
  - **Country**: The country where the coffee farm is located.
  - **Region**: The region within the country.
  - **City**: The city or locality associated with the farm location.

By analyzing the response, users can obtain comprehensive geographical insights into coffee farming locations, which can be instrumental for further research, market analysis, or logistical planning. Adjust the implementation according to your specific needs, testing, or integrating with larger systems if required.
