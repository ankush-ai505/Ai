# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
import unittest
from unittest.mock import patch, MagicMock
from app import app, validate_email, validate_phone, validate_password, validate_personal_info

class TestUserRegistrationApp(unittest.TestCase):
    """Unit test cases for User Registration Flask App"""

    def setUp(self):
        """Set up the test client for Flask app"""
        self.app = app.test_client()
        self.app.testing = True

    def test_validate_email(self):
        """Test email validation with valid and invalid email addresses"""
        self.assertTrue(validate_email("test@example.com"))
        self.assertFalse(validate_email("invalid-email"))
        self.assertFalse(validate_email("test@.com"))

    def test_validate_phone(self):
        """Test phone number validation with valid and invalid phone numbers"""
        self.assertTrue(validate_phone("123-456-7890"))
        self.assertFalse(validate_phone("1234567890"))
        self.assertFalse(validate_phone("123-45-67890"))

    def test_validate_password(self):
        """Test password validation with valid and invalid passwords"""
        self.assertTrue(validate_password("Strong@123"))
        self.assertFalse(validate_password("weakpassword"))
        self.assertFalse(validate_password("Short1!"))

    def test_validate_personal_info(self):
        """Test personal information validation with valid and invalid data"""
        self.assertIsNone(validate_personal_info("John", 25, "Male", "123 Main St"))
        self.assertEqual(validate_personal_info("J", 25, "Male", "123 Main St"), "Invalid name")
        self.assertEqual(validate_personal_info("John", 17, "Male", "123 Main St"), "Age must be between 18 and 99")
        self.assertEqual(validate_personal_info("John", 25, "Other", "123 Main St"), "Invalid sex")
        self.assertEqual(validate_personal_info("John", 25, "Male", "123"), "Invalid address")

    @patch("app.users_collection.insert_one")
    def test_register_user_success(self, mock_insert_one):
        """Test successful user registration"""
        mock_insert_one.return_value = MagicMock()
        response = self.app.post('/register', json={
            "email_or_phone": "test@example.com",
            "password": "Strong@123",
            "confirm_password": "Strong@123",
            "security_question": "What is your pet's name?"
        })
        self.assertEqual(response.status_code, 201)
        self.assertIn("User registered successfully", response.get_json()["message"])

    def test_register_user_invalid_email(self):
        """Test user registration with invalid email"""
        response = self.app.post('/register', json={
            "email_or_phone": "invalid-email",
            "password": "Strong@123",
            "confirm_password": "Strong@123",
            "security_question": "What is your pet's name?"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid email address", response.get_json()["error"])

    @patch("app.users_collection.find_one")
    def test_forgot_password_success(self, mock_find_one):
        """Test forgot password functionality with correct security answer"""
        mock_find_one.return_value = {
            "email_or_phone": "test@example.com",
            "security_question": "What is your pet's name?"
        }
        response = self.app.post('/forgot-password', json={
            "email_or_phone": "test@example.com",
            "security_answer": "What is your pet's name?"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Password reset link sent", response.get_json()["message"])

    @patch("app.users_collection.find_one")
    def test_forgot_password_incorrect_answer(self, mock_find_one):
        """Test forgot password functionality with incorrect security answer"""
        mock_find_one.return_value = {
            "email_or_phone": "test@example.com",
            "security_question": "What is your pet's name?"
        }
        response = self.app.post('/forgot-password', json={
            "email_or_phone": "test@example.com",
            "security_answer": "Wrong answer"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Incorrect security answer", response.get_json()["error"])

    @patch("app.users_collection.update_one")
    def test_save_personal_info_success(self, mock_update_one):
        """Test saving personal information successfully"""
        mock_update_one.return_value = MagicMock()
        response = self.app.post('/save-personal-info', json={
            "email_or_phone": "test@example.com",
            "name": "John",
            "age": 25,
            "sex": "Male",
            "address": "123 Main St",
            "profile_picture": "profile.jpg"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Personal information saved successfully", response.get_json()["message"])

    @patch("app.users_collection.find_one")
    def test_login_user_success(self, mock_find_one):
        """Test successful user login"""
        mock_find_one.return_value = {
            "email_or_phone": "test@example.com",
            "password": generate_password_hash("Strong@123")
        }
        response = self.app.post('/login', json={
            "email_or_phone": "test@example.com",
            "password": "Strong@123"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Login successful", response.get_json()["message"])

    @patch("app.users_collection.find_one")
    def test_login_user_invalid_credentials(self, mock_find_one):
        """Test login with invalid credentials"""
        mock_find_one.return_value = None
        response = self.app.post('/login', json={
            "email_or_phone": "test@example.com",
            "password": "WrongPassword"
        })
        self.assertEqual(response.status_code, 400)
        self.assertIn("Incorrect email or password", response.get_json()["error"])

    def test_get_food_preferences(self):
        """Test retrieving food preferences"""
        response = self.app.get('/food-preferences')
        self.assertEqual(response.status_code, 200)
        self.assertIn("food_preferences", response.get_json())

    def test_logout_user(self):
        """Test user logout"""
        response = self.app.post('/logout')
        self.assertEqual(response.status_code, 200)
        self.assertIn("Logout successful", response.get_json()["message"])

if __name__ == '__main__':
    unittest.main()


#*End of AI Generated Content*