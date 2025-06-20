# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
# -*- coding: utf-8 -*-

"""
User Registration and Management API
=====================================
"""

import os
import re
import bcrypt
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename

# Constants and Static Values
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "user_management"
COLLECTION_USERS = "users"
COLLECTION_RESET_PASSWORD = "reset_password"
TOP_FOOD_TYPES = [
    "Indian", "Chinese", "French", "Italian", "Mexican",
    "Japanese", "Thai", "American", "Greek", "Mediterranean"
]
EMAIL_REGEX = r"[^@]+@[^@]+\.[^@]+"
PHONE_REGEX = r"\d{3}-\d{3}-\d{4}"
PASSWORD_REGEX = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$"
NAME_REGEX = r"^[a-zA-Z]{2,}$"
AGE_RANGE = range(18, 100)
SEX_OPTIONS = ["Male", "Female"]
ADDRESS_REGEX = r"^[a-zA-Z0-9\s]{5,}$"

app = Flask(__name__)
app.config["MONGO_URI"] = MONGO_URI
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
mongo = PyMongo(app)


def allowed_file(filename):
    """
    Check if the file extension is allowed.
    
    :param filename: The filename to check.
    :return: True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_email(email):
    """
    Validate the email address.
    
    :param email: The email address to validate.
    :return: True if the email is valid, False otherwise.
    """
    return bool(re.match(EMAIL_REGEX, email))


def validate_phone(phone):
    """
    Validate the phone number.
    
    :param phone: The phone number to validate.
    :return: True if the phone number is valid, False otherwise.
    """
    return bool(re.match(PHONE_REGEX, phone))


def validate_password(password):
    """
    Validate the password.
    
    :param password: The password to validate.
    :return: True if the password is valid, False otherwise.
    """
    return bool(re.match(PASSWORD_REGEX, password))


def validate_name(name):
    """
    Validate the name.
    
    :param name: The name to validate.
    :return: True if the name is valid, False otherwise.
    """
    return bool(re.match(NAME_REGEX, name))


def validate_age(age):
    """
    Validate the age.
    
    :param age: The age to validate.
    :return: True if the age is valid, False otherwise.
    """
    return age in AGE_RANGE


def validate_sex(sex):
    """
    Validate the sex.
    
    :param sex: The sex to validate.
    :return: True if the sex is valid, False otherwise.
    """
    return sex in SEX_OPTIONS


def validate_address(address):
    """
    Validate the address.
    
    :param address: The address to validate.
    :return: True if the address is valid, False otherwise.
    """
    return bool(re.match(ADDRESS_REGEX, address))


@app.route('/register', methods=['POST'])
def register():
    """
    Handle user registration.
    
    :return: A JSON response with the result of the registration.
    """
    try:
        data = request.get_json()
        email = data.get('email')
        phone = data.get('phone')
        password = data.get('password')
        confirm_password = data.get('confirm_password')
        security_question = data.get('security_question')
        
        if not (validate_email(email) or validate_phone(phone)):
            return jsonify({'error': 'Invalid email or phone number'}), 400
        
        if not validate_password(password):
            return jsonify({'error': 'Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character'}), 400
        
        if password != confirm_password:
            return jsonify({'error': 'Passwords do not match'}), 400
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user = {
            'email': email,
            'phone': phone,
            'password': hashed_password,
            'security_question': security_question
        }
        mongo.db[COLLECTION_USERS].insert_one(user)
        return jsonify({'message': 'User registered successfully'}), 201
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    """
    Handle forgot password.
    
    :return: A JSON response with the result of the forgot password.
    """
    try:
        data = request.get_json()
        email = data.get('email')
        phone = data.get('phone')
        security_question_answer = data.get('security_question_answer')
        
        if not (validate_email(email) or validate_phone(phone)):
            return jsonify({'error': 'Invalid email or phone number'}), 400
        
        user = mongo.db[COLLECTION_USERS].find_one({'email': email, 'phone': phone})
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        if user['security_question_answer'] != security_question_answer:
            return jsonify({'error': 'Incorrect security question answer'}), 400
        
        # Send password reset link to user's email or phone
        # ...
        return jsonify({'message': 'Password reset link sent successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/save-personal-info', methods=['POST'])
def save_personal_info():
    """
    Handle saving user's personal information.
    
    :return: A JSON response with the result of saving personal information.
    """
    try:
        data = request.get_json()
        name = data.get('name')
        age = int(data.get('age'))
        sex = data.get('sex')
        address = data.get('address')
        profile_picture = request.files.get('profile_picture')
        
        if not validate_name(name):
            return jsonify({'error': 'Invalid name'}), 400
        
        if not validate_age(age):
            return jsonify({'error': 'Age must be between 18 and 99'}), 400
        
        if not validate_sex(sex):
            return jsonify({'error': 'Invalid sex'}), 400
        
        if not validate_address(address):
            return jsonify({'error': 'Invalid address'}), 400
        
        if profile_picture and allowed_file(profile_picture.filename):
            filename = secure_filename(profile_picture.filename)
            profile_picture.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        user = {
            'name': name,
            'age': age,
            'sex': sex,
            'address': address,
            'profile_picture': filename if profile_picture else None
        }
        mongo.db[COLLECTION_USERS].update_one({'email': data.get('email')}, {'$set': user})
        return jsonify({'message': 'Personal information saved successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get-food-types', methods=['GET'])
def get_food_types():
    """
    Handle getting top 10 food types.
    
    :return: A JSON response with the top 10 food types.
    """
    try:
        return jsonify({'food_types': TOP_FOOD_TYPES}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/login', methods=['POST'])
def login():
    """
    Handle user login.
    
    :return: A JSON response with the result of the login.
    """
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        user = mongo.db[COLLECTION_USERS].find_one({'email': email})
        if not user:
            return jsonify({'error': 'Incorrect email or password'}), 401
        
        if not bcrypt.checkpw(password.encode('utf-8'), user['password']):
            return jsonify({'error': 'Incorrect email or password'}), 401
        
        return jsonify({'message': 'User logged in successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/logout', methods=['POST'])
def logout():
    """
    Handle user logout.
    
    :return: A JSON response with the result of the logout.
    """
    try:
        # Logout logic here
        return jsonify({'message': 'User logged out successfully'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Unit Tests
import unittest

class TestUserRegistrationAPI(unittest.TestCase):
    
    def test_register_valid_user(self):
        # Test registering a valid user
        data = {
            'email': 'test@example.com',
            'password': 'Test@123',
            'confirm_password': 'Test@123',
            'security_question': 'What is your favorite color?'
        }
        response = app.test_client().post('/register', json=data)
        self.assertEqual(response.status_code, 201)
    
    def test_register_invalid_email(self):
        # Test registering a user with an invalid email
        data = {
            'email': 'invalid_email',
            'password': 'Test@123',
            'confirm_password': 'Test@123',
            'security_question': 'What is your favorite color?'
        }
        response = app.test_client().post('/register', json=data)
        self.assertEqual(response.status_code, 400)
    
    def test_forgot_password_valid_user(self):
        # Test forgot password for a valid user
        data = {
            'email': 'test@example.com',
            'security_question_answer': 'Blue'
        }
        response = app.test_client().post('/forgot-password', json=data)
        self.assertEqual(response.status_code, 200)
    
    def test_save_personal_info_valid_user(self):
        # Test saving personal info for a valid user
        data = {
            'email': 'test@example.com',
            'name': 'John Doe',
            'age': 25,
            'sex': 'Male',
            'address': '123 Main St'
        }
        response = app.test_client().post('/save-personal-info', json=data)
        self.assertEqual(response.status_code, 200)
    
    def test_login_valid_user(self):
        # Test logging in a valid user
        data = {
            'email': 'test@example.com',
            'password': 'Test@123'
        }
        response = app.test_client().post('/login', json=data)
        self.assertEqual(response.status_code, 200)
    
    def test_logout_valid_user(self):
        # Test logging out a valid user
        response = app.test_client().post('/logout')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    app.run(debug=True)


#*End of AI Generated Content*