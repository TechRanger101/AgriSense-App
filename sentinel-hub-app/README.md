# Crop and Pest Analysis API

## Overview

The Crop and Pest Analysis API leverages satellite data from Sentinel Hub to deliver insights into crop health and pest risks. By processing data related to optical indices, land surface temperature (LST), and atmospheric conditions, it offers a detailed assessment of crop health and potential pest threats within a specified geographic area and time period.

## Features

- **Satellite Integration**: Utilizes data from Sentinel-2, Sentinel-3, and Sentinel-5P satellites.
- **Index Calculations**: Computes indices such as NDVI, NDWI, and NDMI to assess crop health.
- **Temperature and Atmospheric Monitoring**: Provides land surface temperature (LST) data and monitors pollutants like NO2, O3, and SO2.
- **Risk Assessment**: Offers insights into pest infestation risks and overall crop health.
- **User Authentication**: Secure user signup and signin functionalities for accessing the API.

## Prerequisites

- Python 3.7 or newer

## Installation

Follow these steps to set up the project:

1. **Clone the Repository**

   Start by cloning the project repository:

   ```bash
   git clone https://github.com/<>/sentinel-hub-app
   cd sentinel-hub-app
   ```

2. **Set Up Your Environment**

   Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Use `pip` to install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   Copy the example environment file and provide your credentials:

   ```bash
   cp app/config/.env.example app/config/.env
   ```

   Edit `app/config/.env` to include your Sentinel Hub and Firestore credentials:

   ```plaintext
   SH_CLIENT_ID=<your-sentinel-hub-client-id>
   SH_CLIENT_SECRET=<your-sentinel-hub-client-secret>
   SECRET_KEY=<your-secret-key>
   ```

5. **Configure Firebase**

   Place your Firestore credentials JSON file in the `app/config` directory and rename it to `firebase-adminsdk-secret-key.json`.

## Usage

1. **Run the API Server**

   Start the API server by navigating to the `app` directory and executing:

   ```bash
   cd app
   python app.py
   ```

   The API server will run at `http://0.0.0.0:5000`.

2. **Testing the API**

   Use the provided test script located at `test/test.py` to verify the API's functionality:

   ```python
   import requests

   # Define the base URL for the API
   base_url = 'http://localhost:5000'

   # Define headers
   headers = {'Content-Type': 'application/json'}

   def signup(email, password):
       """Attempt to sign up a new user."""
       signup_data = {
           'email': email,
           'password': password
       }

       try:
           response = requests.post(
               f'{base_url}/api/signup', json=signup_data, headers=headers)
           print('Signup Status Code:', response.status_code)
           print('Signup Response JSON:', response.json())
           return response.json()
       except requests.exceptions.RequestException as e:
           print('Signup Error:', e)
           return None

   def signin(email, password):
       """Attempt to sign in with existing credentials."""
       signin_data = {
           'email': email,
           'password': password
       }

       try:
           response = requests.post(
               f'{base_url}/api/signin', json=signin_data, headers=headers)
           print('Signin Status Code:', response.status_code)
           response_json = response.json()
           print('Signin Response JSON:', response_json)
           return response_json
       except requests.exceptions.RequestException as e:
           print('Signin Error:', e)
           return None

   def calculate(token):
       """Use the token to call the /api/calculate endpoint."""
       calculate_headers = {
           'Authorization': f'Bearer {token}'
       }

       params = {
           'latitude': '29.346443',
           'longitude': '-95.152565',
           'start_date': '2024-05-01',
           'end_date': '2024-05-31'
       }

       try:
           response = requests.get(
               f'{base_url}/api/calculate', headers=calculate_headers, params=params)
           print('Calculate Status Code:', response.status_code)
           print('Calculate Response JSON:', response.json())
       except requests.exceptions.RequestException as e:
           print('Calculate Error:', e)

   def main():
       # Define user credentials
       email = 'testuser@example.com'
       password = 'securepassword'

       # Sign up the user
       signup_response = signup(email, password)

       # Sign in to get the JWT token
       signin_response = signin(email, password)

       # If signin was successful, proceed to calculate
       if signin_response and 'token' in signin_response:
           token = signin_response['token']
           calculate(token)
       else:
           print("Could not obtain a valid token; cannot perform calculation.")

   if __name__ == '__main__':
       main()
   ```

3. **Expected Output**

   On successful execution, the API will return a JSON response with indices and assessments, for example:

   ```json
   {
     "result": {
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
     },
     "status": "OK"
   }
   ```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
