# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
import re
from flask import Flask, request, jsonify
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash

# Constants
DATABASE_URI = "mongodb://localhost:27017/"
DATABASE_NAME = "user_registration"
COLLECTION_NAME = "users"
FOOD_PREFERENCES = [
    "Indian", "Chinese", "French", "Italian", "Mexican",
    "Japanese", "Thai", "American", "Greek", "Mediterranean"
]
PASSWORD_REGEX = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$"
EMAIL_REGEX = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
PHONE_REGEX = r"^\d{3}-\d{3}-\d{4}$"

# MongoDB Client
client = MongoClient(DATABASE_URI)
db = client[DATABASE_NAME]
users_collection = db[COLLECTION_NAME]

# Flask App
app = Flask(__name__)

def validate_email_or_phone(value):
    """Validate email or phone number format."""
    if re.match(EMAIL_REGEX, value) or re.match(PHONE_REGEX, value):
        return True
    return False

def validate_password(password):
    """Validate password format."""
    return bool(re.match(PASSWORD_REGEX, password))

def validate_personal_info(name, age, sex, address):
    """Validate personal information fields."""
    if not (name.isalpha() and len(name) >= 2):
        return "Invalid name"
    if not (18 <= age <= 99):
        return "Age must be between 18 and 99"
    if sex not in ["Male", "Female"]:
        return "Invalid sex"
    if not (len(address) >= 5 and all(c.isalnum() or c.isspace() for c in address)):
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

        if not validate_email_or_phone(email_or_phone):
            return jsonify({"error": "Invalid email or phone number"}), 400
        if not validate_password(password):
            return jsonify({"error": "Password must meet complexity requirements"}), 400
        if password != confirm_password:
            return jsonify({"error": "Passwords do not match"}), 400

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
        if not user or user.get("security_question") != security_answer:
            return jsonify({"error": "Invalid security answer"}), 400

        # Simulate sending a reset link
        return jsonify({"message": "Password reset link sent"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-personal-info', methods=['POST'])
def save_personal_info():
    """Save user's personal information."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        name = data.get("name")
        age = data.get("age")
        sex = data.get("sex")
        address = data.get("address")
        profile_picture = data.get("profile_picture")

        validation_error = validate_personal_info(name, age, sex, address)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        users_collection.update_one(
            {"email_or_phone": email_or_phone},
            {"$set": {
                "name": name,
                "age": age,
                "sex": sex,
                "address": address,
                "profile_picture": profile_picture
            }}
        )
        return jsonify({"message": "Personal information saved successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/food-preferences', methods=['GET'])
def get_food_preferences():
    """Get top 10 food preferences."""
    try:
        return jsonify({"food_preferences": FOOD_PREFERENCES}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    """Handle user login."""
    try:
        data = request.json
        email_or_phone = data.get("email_or_phone")
        password = data.get("password")

        user = users_collection.find_one({"email_or_phone": email_or_phone})
        if not user or not check_password_hash(user["password"], password):
            return jsonify({"error": "Incorrect email or password"}), 400

        return jsonify({"message": "Login successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout."""
    try:
        # Simulate logout
        return jsonify({"message": "Logout successful"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Unit Tests
import unittest

class TestUserRegistration(unittest.TestCase):
    def test_validate_email_or_phone(self):
        self.assertTrue(validate_email_or_phone("user@example.com"))
        self.assertTrue(validate_email_or_phone("555-555-5555"))
        self.assertFalse(validate_email_or_phone("invalid"))

    def test_validate_password(self):
        self.assertTrue(validate_password("Password1!"))
        self.assertFalse(validate_password("password"))

    def test_validate_personal_info(self):
        self.assertIsNone(validate_personal_info("John", 25, "Male", "123 Main St"))
        self.assertEqual(validate_personal_info("J", 25, "Male", "123 Main St"), "Invalid name")
        self.assertEqual(validate_personal_info("John", 17, "Male", "123 Main St"), "Age must be between 18 and 99")
        self.assertEqual(validate_personal_info("John", 25, "Other", "123 Main St"), "Invalid sex")
        self.assertEqual(validate_personal_info("John", 25, "Male", "123"), "Invalid address")

if __name__ == '__main__':
    app.run(debug=True)


#*End of AI Generated Content*