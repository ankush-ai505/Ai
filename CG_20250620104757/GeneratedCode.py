# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
# ***********************************************************
# *                                                         *
# *  Constants and Static Values                           *
# *                                                         *
# ***********************************************************

# Constants for Validation Error Messages
INVALID_EMAIL_ERROR = "Invalid email address"
INVALID_PHONE_NUMBER_ERROR = "Invalid phone number"
PASSWORD_LENGTH_ERROR = "Password must be at least 8 characters long"
PASSWORD_COMPLEXITY_ERROR = "Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character"
INVALID_NAME_ERROR = "Name should be at least 2 characters long and only contain letters"
INVALID_AGE_ERROR = "Age must be a number between 18 and 99"
INVALID_SEX_ERROR = "Sex should be either 'Male' or 'Female'"
INVALID_ADDRESS_ERROR = "Address should be at least 5 characters long and contain only letters, numbers, and spaces"

# Constants for MongoDB
MONGO_DB_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "user_database"
COLLECTION_NAME = "users"

# Constants for Food Preferences
FOOD_PREFERENCES = [
    "Indian",
    "Chinese",
    "French",
    "Italian",
    "Mexican",
    "Japanese",
    "Thai",
    "American",
    "Greek",
    "Mediterranean"
]

# ***********************************************************
# *                                                         *
# *  Import Statements                                      *
# *                                                         *
# ***********************************************************

import re
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from bson.objectid import ObjectId
import bcrypt
import logging

# ***********************************************************
# *                                                         *
# *  Flask Application                                      *
# *                                                         *
# ***********************************************************

app = Flask(__name__)
app.config["MONGO_URI"] = MONGO_DB_URL + DATABASE_NAME
mongo = PyMongo(app)

# ***********************************************************
# *                                                         *
# *  Function to Validate Email Address                    *
# *                                                         *
# ***********************************************************

def validate_email(email):
    """
    Validate Email Address.

    Args:
        email (str): Email Address to be validated.

    Returns:
        bool: True if Email Address is valid, False otherwise.
    """
    try:
        email_regex = r"[^@]+@[^@]+\.[^@]+"
        if re.match(email_regex, email):
            return True
        else:
            raise ValueError(INVALID_EMAIL_ERROR)
    except ValueError as e:
        logging.error(e)
        return False

# ***********************************************************
# *                                                         *
# *  Function to Validate Phone Number                     *
# *                                                         *
# ***********************************************************

def validate_phone_number(phone_number):
    """
    Validate Phone Number.

    Args:
        phone_number (str): Phone Number to be validated.

    Returns:
        bool: True if Phone Number is valid, False otherwise.
    """
    try:
        phone_regex = r"\d{3}-\d{3}-\d{4}"
        if re.match(phone_regex, phone_number):
            return True
        else:
            raise ValueError(INVALID_PHONE_NUMBER_ERROR)
    except ValueError as e:
        logging.error(e)
        return False

# ***********************************************************
# *                                                         *
# *  Function to Validate Password                         *
# *                                                         *
# ***********************************************************

def validate_password(password):
    """
    Validate Password.

    Args:
        password (str): Password to be validated.

    Returns:
        bool: True if Password is valid, False otherwise.
    """
    try:
        if len(password) < 8:
            raise ValueError(PASSWORD_LENGTH_ERROR)
        if not any(char.isupper() for char in password):
            raise ValueError(PASSWORD_COMPLEXITY_ERROR)
        if not any(char.islower() for char in password):
            raise ValueError(PASSWORD_COMPLEXITY_ERROR)
        if not any(char.isdigit() for char in password):
            raise ValueError(PASSWORD_COMPLEXITY_ERROR)
        if not any(char.special for char in password):
            raise ValueError(PASSWORD_COMPLEXITY_ERROR)
        return True
    except ValueError as e:
        logging.error(e)
        return False

# ***********************************************************
# *                                                         *
# *  Function to Validate User Details                     *
# *                                                         *
# ***********************************************************

def validate_user_details(name, age, sex, address):
    """
    Validate User Details.

    Args:
        name (str): User Name.
        age (int): User Age.
        sex (str): User Sex.
        address (str): User Address.

    Returns:
        bool: True if User Details are valid, False otherwise.
    """
    try:
        if len(name) < 2 or not name.isalpha():
            raise ValueError(INVALID_NAME_ERROR)
        if age < 18 or age > 99:
            raise ValueError(INVALID_AGE_ERROR)
        if sex not in ["Male", "Female"]:
            raise ValueError(INVALID_SEX_ERROR)
        if len(address) < 5 or not all(char.isalnum() or char.isspace() for char in address):
            raise ValueError(INVALID_ADDRESS_ERROR)
        return True
    except ValueError as e:
        logging.error(e)
        return False

# ***********************************************************
# *                                                         *
# *  API Endpoints                                          *
# *                                                         *
# ***********************************************************

# Register User
@app.route('/register', methods=['POST'])
def register_user():
    """
    Register User.

    Returns:
        jsonify: Success or Error Message.
    """
    try:
        data = request.json
        email = data.get('email')
        phone_number = data.get('phone_number')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        security_question = data.get('security_question')

        if email:
            if not validate_email(email):
                return jsonify({'error': INVALID_EMAIL_ERROR}), 400
        elif phone_number:
            if not validate_phone_number(phone_number):
                return jsonify({'error': INVALID_PHONE_NUMBER_ERROR}), 400

        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400

        if not validate_password(password):
            return jsonify({'error': PASSWORD_LENGTH_ERROR}), 400

        # Hash Password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Save User to MongoDB
        user_id = mongo.db[COLLECTION_NAME].insert_one({
            'email': email,
            'phone_number': phone_number,
            'password': hashed_password,
            'security_question': security_question
        }).inserted_id

        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        logging.error(e)
        return jsonify({'error': 'Internal Server Error'}), 500

# Save User Details
@app.route('/save-user-details', methods=['POST'])
def save_user_details():
    """
    Save User Details.

    Returns:
        jsonify: Success or Error Message.
    """
    try:
        data = request.json
        user_id = data.get('user_id')
        name = data.get('name')
        age = data.get('age')
        sex = data.get('sex')
        address = data.get('address')

        if not validate_user_details(name, age, sex, address):
            return jsonify({'error': 'Invalid user details'}), 400

        # Update User Details in MongoDB
        mongo.db[COLLECTION_NAME].update_one({
            '_id': ObjectId(user_id)
        }, {
            '$set': {
                'name': name,
                'age': age,
                'sex': sex,
                'address': address
            }
        })

        return jsonify({'message': 'User details saved successfully'}), 200
    except Exception as e:
        logging.error(e)
        return jsonify({'error': 'Internal Server Error'}), 500

# Forgot Password
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    """
    Forgot Password.

    Returns:
        jsonify: Success or Error Message.
    """
    try:
        data = request.json
        email = data.get('email')
        security_question_answer = data.get('security_question_answer')

        # Check if Email Exists in MongoDB
        user = mongo.db[COLLECTION_NAME].find_one({
            'email': email
        })

        if not user:
            return jsonify({'error': 'Email not found'}), 404

        # Check if Security Question Answer is Correct
        if user['security_question'] != security_question_answer:
            return jsonify({'error': 'Incorrect security question answer'}), 401

        # Send Password Reset Link to Email
        # TODO: Implement Email Service

        return jsonify({'message': 'Password reset link sent successfully'}), 200
    except Exception as e:
        logging.error(e)
        return jsonify({'error': 'Internal Server Error'}), 500

# Get Food Preferences
@app.route('/food-preferences', methods=['GET'])
def get_food_preferences():
    """
    Get Food Preferences.

    Returns:
        jsonify: List of Food Preferences.
    """
    try:
        return jsonify({'food_preferences': FOOD_PREFERENCES}), 200
    except Exception as e:
        logging.error(e)
        return jsonify({'error': 'Internal Server Error'}), 500

if __name__ == '__main__':
    app.run(debug=True)


#*End of AI Generated Content*