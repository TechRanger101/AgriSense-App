import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
from flask import Flask, request, jsonify
# from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from dotenv import load_dotenv
import datetime
import requests
import json
import jwt
import os

# from firebase_api_usage_func import use_api_key
from sentinel_hub_func import get_all_crop_and_pest_info


load_dotenv(dotenv_path='./config/.env')

cred = credentials.Certificate('./config/firebase-adminsdk-secret-key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

status_code_messages = {
    200: 'OK',
    400: 'Bad Request',
    401: 'Unauthorized',
    404: 'Not Found',
    500: 'Internal Server Error'
}


def create_token(user_id):
    '''Create a JWT token for a given user_id.'''
    payload = {
        'sub': user_id,
        'iat': datetime.datetime.utcnow(),
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing', 'status': status_code_messages[401]}), 401
        try:
            token = token.split(' ')[1]
            jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired', 'status': status_code_messages[401]}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token', 'status': status_code_messages[401]}), 401
        return f(*args, **kwargs)
    return decorated


@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Crop, Pest Analysis and Coffee Farmers Information API", 200


@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({"message": "This is the API endpoint serving information related to crop, pest analysis and coffee farmers information."}), 200


# @app.route('/api/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        userId = data['id']
        password = data['password']

        # Check if user already exists
        users_ref = db.collection('logins').where(
            filter=FieldFilter('id', '==', userId)).get()
        if users_ref:
            return jsonify({'error': 'User already exists', 'status': status_code_messages[400]}), 400

        # Hash the password for security
        # hashed_password = generate_password_hash(password)

        # Create new user in Firestore
        user_ref = db.collection('logins').add({
            'id': userId,
            'password': password
        })
        token = create_token(user_ref[1].id)

        return jsonify({'message': 'Account created successfully.', 'token': token, 'status': status_code_messages[200]}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': status_code_messages[400]}), 400


@app.route('/api/signin', methods=['POST'])
def signin():
    try:
        data = request.get_json()
        userId = data['id']
        password = data['password']

        # Retrieve user from Firestore
        users_ref = db.collection('logins').where(
            filter=FieldFilter('id', '==', userId)).stream()
        user_doc = next(users_ref, None)

        if not user_doc:
            return jsonify({'error': 'User not found', 'status': status_code_messages[401]}), 401

        user_data = user_doc.to_dict()

        # Verify the password
        # if not check_password_hash(user_data['password'], password):
        if user_data['password'] != password:
            return jsonify({'error': 'Incorrect password', 'status': status_code_messages[401]}), 401

        token = create_token(user_doc.id)

        return jsonify({'token': token, 'status': status_code_messages[200]}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'status': status_code_messages[400]}), 400


def increment_api_usage(user_id):
    try:
        user_ref = db.collection('logins').document(user_id)
        user_snapshot = user_ref.get()
        if user_snapshot.exists:
            current_count = user_snapshot.to_dict().get('calculate_usage', 0)
        else:
            current_count = 0
        update_data = {
            'calculate_usage': current_count + 1,
            'last_used': datetime.datetime.utcnow()
        }
        user_ref.update(update_data)
    except Exception as e:
        print(f"Error incrementing API usage for {user_id}: {str(e)}")


@app.route('/api/calculate', methods=['GET'])
@token_required
def calculate():
    try:
        # Extract the user ID from the token
        token = request.headers.get('Authorization').split(' ')[1]
        decoded_token = jwt.decode(
            token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = decoded_token['sub']

        # Retrieve parameters from the request
        latitude = float(request.args.get('latitude'))
        longitude = float(request.args.get('longitude'))
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        result = get_all_crop_and_pest_info(
            latitude, longitude, start_date, end_date)

        # Increment the usage count for this user
        increment_api_usage(user_id)

        if 'error' in result and result['error']:
            return jsonify({'error': result['error'], 'status': status_code_messages[result['status']]}), result['status']

        return jsonify({'result': [result], 'status': status_code_messages[200]}), 200

    except ValueError as ve:
        return jsonify({'error': f'Invalid value: {str(ve)}', 'status': status_code_messages[400]}), 400
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}', 'status': status_code_messages[500]}), 500


@app.route('/api/calculate-multi-locations', methods=['GET'])
@token_required
def calculate_multiple():
    try:
        # Extract the user ID from the token
        token = request.headers.get('Authorization').split(' ')[1]
        decoded_token = jwt.decode(
            token, app.config['SECRET_KEY'], algorithms=['HS256'])
        user_id = decoded_token['sub']

        # Retrieve parameters from the request
        locations_param = request.args.get('locations')

        if not locations_param:
            return jsonify({'error': 'No locations provided', 'status': status_code_messages[400]}), 400

        # Parse the locations parameter
        try:
            locations = json.loads(locations_param)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid format for locations', 'status': status_code_messages[400]}), 400

        if not isinstance(locations, list):
            return jsonify({'error': 'Locations should be a list of objects', 'status': status_code_messages[400]}), 400

        results = []

        for location in locations:
            try:
                latitude = float(location.get('latitude'))
                longitude = float(location.get('longitude'))
                start_date = location.get('start_date')
                end_date = location.get('end_date')

                result = get_all_crop_and_pest_info(
                    latitude, longitude, start_date, end_date)

                # Increment the usage count for this user
                increment_api_usage(user_id)

                if 'error' in result and result['error']:
                    results.append(
                        {'location': location, 'error': result['error'], 'status': status_code_messages[result['status']]})
                else:
                    results.append(
                        {'location': location, 'result': result, 'status': status_code_messages[200]})

            except ValueError as ve:
                results.append(
                    {'location': location, 'error': f'Invalid value: {str(ve)}', 'status': status_code_messages[400]})
            except Exception as e:
                results.append(
                    {'location': location, 'error': f'An error occurred: {str(e)}', 'status': status_code_messages[500]})

        return jsonify({'results': results}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}', 'status': status_code_messages[500]}), 500


@app.route('/api/coffee-farmers', methods=['GET'])
@token_required
def get_coffee_farmers():
    try:
        # Make a GET request to the provided URL
        response = requests.get(
            'http://54.198.197.228/backend/api/get_location/')

        # Check if the request was successful
        if response.status_code == 200:
            # Return the JSON data fetched from the endpoint
            return jsonify(response.json()), 200
        else:
            # Return an error message if the request was not successful
            return jsonify({"error": "Failed to fetch data"}), response.status_code
    except requests.exceptions.RequestException as e:
        # Handle exceptions that occur during the request
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
