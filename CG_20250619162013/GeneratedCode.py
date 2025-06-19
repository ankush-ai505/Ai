# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
import re
import json
from flask import Flask, request, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# Constants
DATABASE_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "user_registration"
COLLECTION_NAME = "users"
FOOD_PREFERENCES = [
    "Indian", "Chinese", "French", "Italian", "Mexican",
    "Japanese", "Thai", "American", "Greek", "Mediterranean"
]
PASSWORD_RESET_LINK = "https://example.com/reset-password"

# Flask App Initialization
app = Flask(__name__)

# MongoDB Client Initialization
client = MongoClient(DATABASE_URL)
db = client[DATABASE_NAME]
users_collection = db[COLLECTION_NAME]

def validate_email(email):
    """Validate email format."""
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

def validate_phone(phone):
    """Validate phone number format."""
    phone_regex = r'^\d{3}-\d{3}-\d{4}$'
    return re.match(phone_regex, phone)

def validate_password(password):
    """Validate password complexity."""
    password_regex = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$'
    return re.match(password_regex, password)

def validate_personal_info(name, age, sex, address):
    """Validate personal information."""
    if not name.isalpha() or len(name) < 2:
        return "Invalid name"
    if not (18 <= age <= 99):
        return "Age must be between 18 and 99"
    if sex not in ["Male", "Female"]:
        return "Invalid sex"
    if not re.match(r'^[a-zA-Z0-9\s]{5,}$', address):
        return "Invalid address"
    return None

@app.route('/register', methods=['POST'])
def register_user():
    """Register a new user."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        password = data.get("password")
        confirm_password = data.get("confirm_password")
        security_question = data.get("security_question")

        if not email_or_phone or not password or not confirm_password or not security_question:
            return jsonify({"error": "All fields are required"}), 400

        if "@" in email_or_phone:
            if not validate_email(email_or_phone):
                return jsonify({"error": "Invalid email address"}), 400
        else:
            if not validate_phone(email_or_phone):
                return jsonify({"error": "Invalid phone number"}), 400

        if password != confirm_password:
            return jsonify({"error": "Passwords do not match"}), 400

        if not validate_password(password):
            return jsonify({"error": "Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character"}), 400

        hashed_password = generate_password_hash(password)
        user_data = {
            "email_or_phone": email_or_phone,
            "password": hashed_password,
            "security_question": security_question
        }
        users_collection.insert_one(user_data)
        return jsonify({"message": "User registered successfully"}), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Handle forgot password functionality."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        security_answer = data.get("security_answer")

        user = users_collection.find_one({"email_or_phone": email_or_phone})
        if not user:
            return jsonify({"error": "User not found"}), 404

        if user.get("security_question") != security_answer:
            return jsonify({"error": "Incorrect security answer"}), 400

        return jsonify({"message": f"Password reset link sent to {email_or_phone}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-personal-info', methods=['POST'])
def save_personal_info():
    """Save user's personal information."""
    try:
        data = request.json
        name = data.get("name")
        age = data.get("age")
        sex = data.get("sex")
        address = data.get("address")
        profile_picture = data.get("profile_picture")

        validation_error = validate_personal_info(name, age, sex, address)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        personal_info = {
            "name": name,
            "age": age,
            "sex": sex,
            "address": address,
            "profile_picture": profile_picture
        }
        users_collection.update_one(
            {"email_or_phone": data.get("email_or_phone")},
            {"$set": personal_info}
        )
        return jsonify({"message": "Personal information saved successfully"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/food-preferences', methods=['GET'])
def get_food_preferences():
    """Get list of food preferences."""
    try:
        return jsonify({"food_preferences": FOOD_PREFERENCES}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_user():
    """Handle user login."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        password = data.get("password")

        user = users_collection.find_one({"email_or_phone": email_or_phone})
        if not user or not check_password_hash(user.get("password"), password):
            return jsonify({"error": "Incorrect email or password"}), 400

        return jsonify({"message": "Login successful"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout_user():
    """Handle user logout."""
    try:
        return jsonify({"message": "Logout successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


#*End of AI Generated Content*