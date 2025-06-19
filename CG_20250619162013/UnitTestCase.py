# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
import unittest
from unittest.mock import patch, MagicMock
from werkzeug.security import generate_password_hash
from app import (
    validate_email_or_phone,
    validate_password,
    validate_personal_info,
    app,
    users_collection
)

class TestUserRegistration(unittest.TestCase):
    """Unit tests for user registration and related functionalities."""

    def test_validate_email_or_phone(self):
        """Test email or phone number validation."""
        self.assertTrue(validate_email_or_phone("user@example.com"))
        self.assertTrue(validate_email_or_phone("555-555-5555"))
        self.assertFalse(validate_email_or_phone("invalid_email"))
        self.assertFalse(validate_email_or_phone("1234567890"))

    def test_validate_password(self):
        """Test password validation."""
        self.assertTrue(validate_password("Password1!"))
        self.assertFalse(validate_password("password"))
        self.assertFalse(validate_password("PASSWORD1!"))
        self.assertFalse(validate_password("Password"))
        self.assertFalse(validate_password("Pass1!"))

    def test_validate_personal_info(self):
        """Test personal information validation."""
        self.assertIsNone(validate_personal_info("John", 25, "Male", "123 Main St"))
        self.assertEqual(validate_personal_info("J", 25, "Male", "123 Main St"), "Invalid name")
        self.assertEqual(validate_personal_info("John", 17, "Male", "123 Main St"), "Age must be between 18 and 99")
        self.assertEqual(validate_personal_info("John", 25, "Other", "123 Main St"), "Invalid sex")
        self.assertEqual(validate_personal_info("John", 25, "Male", "123"), "Invalid address")

    @patch("app.users_collection.insert_one")
    def test_register_user(self, mock_insert_one):
        """Test user registration endpoint."""
        with app.test_client() as client:
            mock_insert_one.return_value = MagicMock()
            payload = {
                "email_or_phone": "user@example.com",
                "password": "Password1!",
                "confirm_password": "Password1!",
                "security_question": "What is your pet's name?"
            }
            response = client.post("/register", json=payload)
            self.assertEqual(response.status_code, 201)
            self.assertIn("User registered successfully", response.get_json().get("message"))

    @patch("app.users_collection.find_one")
    def test_login_success(self, mock_find_one):
        """Test successful login."""
        hashed_password = generate_password_hash("Password1!")
        mock_find_one.return_value = {"email_or_phone": "user@example.com", "password": hashed_password}
        with app.test_client() as client:
            payload = {
                "email_or_phone": "user@example.com",
                "password": "Password1!"
            }
            response = client.post("/login", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Login successful", response.get_json().get("message"))

    @patch("app.users_collection.find_one")
    def test_login_failure(self, mock_find_one):
        """Test login failure with incorrect credentials."""
        mock_find_one.return_value = None
        with app.test_client() as client:
            payload = {
                "email_or_phone": "user@example.com",
                "password": "WrongPassword1!"
            }
            response = client.post("/login", json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn("Incorrect email or password", response.get_json().get("error"))

    @patch("app.users_collection.find_one")
    def test_forgot_password_success(self, mock_find_one):
        """Test forgot password with correct security answer."""
        mock_find_one.return_value = {
            "email_or_phone": "user@example.com",
            "security_question": "What is your pet's name?"
        }
        with app.test_client() as client:
            payload = {
                "email_or_phone": "user@example.com",
                "security_answer": "What is your pet's name?"
            }
            response = client.post("/forgot-password", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Password reset link sent", response.get_json().get("message"))

    @patch("app.users_collection.find_one")
    def test_forgot_password_failure(self, mock_find_one):
        """Test forgot password with incorrect security answer."""
        mock_find_one.return_value = {
            "email_or_phone": "user@example.com",
            "security_question": "What is your pet's name?"
        }
        with app.test_client() as client:
            payload = {
                "email_or_phone": "user@example.com",
                "security_answer": "Wrong answer"
            }
            response = client.post("/forgot-password", json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid security answer", response.get_json().get("error"))

    @patch("app.users_collection.update_one")
    def test_save_personal_info_success(self, mock_update_one):
        """Test saving personal information successfully."""
        mock_update_one.return_value = MagicMock()
        with app.test_client() as client:
            payload = {
                "email_or_phone": "user@example.com",
                "name": "John",
                "age": 25,
                "sex": "Male",
                "address": "123 Main St",
                "profile_picture": "profile_pic_url"
            }
            response = client.post("/save-personal-info", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Personal information saved successfully", response.get_json().get("message"))

    @patch("app.users_collection.update_one")
    def test_save_personal_info_failure(self, mock_update_one):
        """Test saving personal information with invalid data."""
        mock_update_one.return_value = MagicMock()
        with app.test_client() as client:
            payload = {
                "email_or_phone": "user@example.com",
                "name": "J",
                "age": 25,
                "sex": "Male",
                "address": "123",
                "profile_picture": "profile_pic_url"
            }
            response = client.post("/save-personal-info", json=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid name", response.get_json().get("error"))

    def test_get_food_preferences(self):
        """Test retrieving food preferences."""
        with app.test_client() as client:
            response = client.get("/food-preferences")
            self.assertEqual(response.status_code, 200)
            self.assertIn("food_preferences", response.get_json())
            self.assertGreater(len(response.get_json().get("food_preferences")), 0)

    def test_logout(self):
        """Test logout functionality."""
        with app.test_client() as client:
            response = client.post("/logout")
            self.assertEqual(response.status_code, 200)
            self.assertIn("Logout successful", response.get_json().get("message"))

if __name__ == "__main__":
    unittest.main()


#*End of AI Generated Content*