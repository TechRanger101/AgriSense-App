import requests
import json

# Define the base URL for the API
base_url = 'http://localhost:5000'

# Define headers
headers = {'Content-Type': 'application/json'}


def signup(userId, password):
    """Attempt to sign up a new user."""
    signup_data = {
        'id': userId,
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


def signin(userId, password):
    """Attempt to sign in with existing credentials."""
    signin_data = {
        'id': userId,
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


def calculate_multiple(token):
    """Use the token to call the /api/calculate endpoint with multiple locations."""
    calculate_headers = {
        'Authorization': f'Bearer {token}'
    }

    # Define multiple locations each with a set of parameters
    locations = [
        {
            'latitude': '29.346443',
            'longitude': '-95.152565',
            'start_date': '2024-05-01',
            'end_date': '2024-05-31'
        },
        {
            'latitude': '40.7128',
            'longitude': '-74.0060',
            'start_date': '2024-04-01',
            'end_date': '2024-04-30'
        }
    ]

    # Encode the locations as a JSON string
    encoded_locations = json.dumps(locations)

    # Prepare the request parameters with the encoded locations
    params = {
        'locations': encoded_locations
    }

    try:
        response = requests.get(
            f'{base_url}/api/calculate-multi-locations', headers=calculate_headers, params=params)
        print('Calculate Status Code:', response.status_code)
        print('Calculate Response JSON:', response.json())
    except requests.exceptions.RequestException as e:
        print('Calculate Error:', e)


def main():
    # Define user credentials
    userId = 'testuser'
    password = 'securepassword'

    # Sign up the user
    signup_response = signup(userId, password)

    # Sign in to get the JWT token
    signin_response = signin(userId, password)
    print(signin_response)

    # If signin was successful, proceed to calculate
    if signin_response and 'token' in signin_response:
        token = signin_response['token']
        calculate(token)
        calculate_multiple(token)
    else:
        print("Could not obtain a valid token; cannot perform calculation.")


if __name__ == '__main__':
    main()
