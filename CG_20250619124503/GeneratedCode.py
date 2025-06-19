# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
import re
import json
from flask import Flask, request, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# Constants
MONGO_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "user_registration"
COLLECTION_NAME = "users"
FOOD_TYPES = [
    "Indian", "Chinese", "French", "Italian", "Mexican",
    "Japanese", "Thai", "American", "Greek", "Mediterranean"
]
PASSWORD_RESET_LINK = "https://example.com/reset-password"

# Initialize Flask app and MongoDB client
app = Flask(__name__)
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
users_collection = db[COLLECTION_NAME]

# Utility functions
def is_valid_email(email):
    """Validate email format."""
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)

def is_valid_phone(phone):
    """Validate phone number format."""
    return re.match(r"^\d{3}-\d{3}-\d{4}$", phone)

def is_valid_password(password):
    """Validate password strength."""
    return re.match(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$", password)

def is_valid_name(name):
    """Validate name format."""
    return len(name) >= 2 and name.isalpha()

def is_valid_age(age):
    """Validate age range."""
    return 18 <= age <= 99

def is_valid_sex(sex):
    """Validate sex value."""
    return sex in ["Male", "Female"]

def is_valid_address(address):
    """Validate address format."""
    return len(address) >= 5 and re.match(r"^[A-Za-z0-9 ]+$", address)

# Routes
@app.route('/register', methods=['POST'])
def register_user():
    """Handle user registration."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        password = data.get("password")
        confirm_password = data.get("confirm_password")
        security_question = data.get("security_question")

        if not email_or_phone or not password or not confirm_password or not security_question:
            return jsonify({"error": "All fields are required"}), 400

        if "@" in email_or_phone:
            if not is_valid_email(email_or_phone):
                return jsonify({"error": "Invalid email address"}), 400
        else:
            if not is_valid_phone(email_or_phone):
                return jsonify({"error": "Invalid phone number"}), 400

        if not is_valid_password(password):
            return jsonify({"error": "Password must be at least 8 characters long and contain at least one uppercase letter, one lowercase letter, one number, and one special character"}), 400

        if password != confirm_password:
            return jsonify({"error": "Passwords do not match"}), 400

        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            "email_or_phone": email_or_phone,
            "password": hashed_password,
            "security_question": security_question
        })

        return jsonify({"message": "User registered successfully"}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Handle forgot password functionality."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        security_question = data.get("security_question")

        if not email_or_phone or not security_question:
            return jsonify({"error": "All fields are required"}), 400

        user = users_collection.find_one({"email_or_phone": email_or_phone, "security_question": security_question})
        if not user:
            return jsonify({"error": "Invalid security question answer"}), 400

        return jsonify({"message": f"Password reset link sent to {email_or_phone}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-user-info', methods=['POST'])
def save_user_info():
    """Save user's personal information."""
    try:
        data = request.json
        name = data.get("name")
        age = data.get("age")
        sex = data.get("sex")
        address = data.get("address")
        profile_picture = data.get("profile_picture")

        if not name or not age or not sex or not address:
            return jsonify({"error": "All fields are required"}), 400

        if not is_valid_name(name):
            return jsonify({"error": "Invalid name"}), 400

        if not is_valid_age(age):
            return jsonify({"error": "Age must be between 18 and 99"}), 400

        if not is_valid_sex(sex):
            return jsonify({"error": "Sex must be either 'Male' or 'Female'"}), 400

        if not is_valid_address(address):
            return jsonify({"error": "Invalid address"}), 400

        users_collection.update_one(
            {"email_or_phone": data.get("email_or_phone")},
            {"$set": {
                "name": name,
                "age": age,
                "sex": sex,
                "address": address,
                "profile_picture": profile_picture
            }}
        )

        return jsonify({"message": "User information saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/food-preferences', methods=['GET'])
def get_food_preferences():
    """Provide list of food preferences."""
    try:
        return jsonify({"food_types": FOOD_TYPES}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login_user():
    """Handle user login."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        password = data.get("password")

        if not email_or_phone or not password:
            return jsonify({"error": "All fields are required"}), 400

        user = users_collection.find_one({"email_or_phone": email_or_phone})
        if not user or not check_password_hash(user["password"], password):
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