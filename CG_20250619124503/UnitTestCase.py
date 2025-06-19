# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
import unittest
from unittest.mock import patch, MagicMock
from app import app, is_valid_email, is_valid_phone, is_valid_password, is_valid_name, is_valid_age, is_valid_sex, is_valid_address

class TestUserRegistrationApp(unittest.TestCase):
    """Unit test cases for user registration app."""

    def setUp(self):
        """Set up the test client for Flask app."""
        self.app = app.test_client()
        self.app.testing = True

    @patch('app.users_collection')
    def test_register_user_valid_email(self, mock_users_collection):
        """Test user registration with valid email."""
        mock_users_collection.insert_one = MagicMock()
        payload = {
            "email_or_phone": "test@example.com",
            "password": "Password@123",
            "confirm_password": "Password@123",
            "security_question": "Test Answer"
        }
        response = self.app.post('/register', json=payload)
        self.assertEqual(response.status_code, 201)
        self.assertIn("User registered successfully", response.json.get("message"))

    @patch('app.users_collection')
    def test_register_user_invalid_email(self, mock_users_collection):
        """Test user registration with invalid email."""
        payload = {
            "email_or_phone": "invalid-email",
            "password": "Password@123",
            "confirm_password": "Password@123",
            "security_question": "Test Answer"
        }
        response = self.app.post('/register', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid email address", response.json.get("error"))

    @patch('app.users_collection')
    def test_register_user_password_mismatch(self, mock_users_collection):
        """Test user registration with mismatched passwords."""
        payload = {
            "email_or_phone": "test@example.com",
            "password": "Password@123",
            "confirm_password": "Password@124",
            "security_question": "Test Answer"
        }
        response = self.app.post('/register', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Passwords do not match", response.json.get("error"))

    @patch('app.users_collection')
    def test_forgot_password_valid(self, mock_users_collection):
        """Test forgot password with valid inputs."""
        mock_users_collection.find_one.return_value = {"email_or_phone": "test@example.com", "security_question": "Test Answer"}
        payload = {
            "email_or_phone": "test@example.com",
            "security_question": "Test Answer"
        }
        response = self.app.post('/forgot-password', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Password reset link sent", response.json.get("message"))

    @patch('app.users_collection')
    def test_forgot_password_invalid(self, mock_users_collection):
        """Test forgot password with invalid inputs."""
        mock_users_collection.find_one.return_value = None
        payload = {
            "email_or_phone": "test@example.com",
            "security_question": "Wrong Answer"
        }
        response = self.app.post('/forgot-password', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid security question answer", response.json.get("error"))

    @patch('app.users_collection')
    def test_save_user_info_valid(self, mock_users_collection):
        """Test saving user information with valid inputs."""
        mock_users_collection.update_one = MagicMock()
        payload = {
            "email_or_phone": "test@example.com",
            "name": "John",
            "age": 25,
            "sex": "Male",
            "address": "123 Main Street",
            "profile_picture": "profile.jpg"
        }
        response = self.app.post('/save-user-info', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("User information saved successfully", response.json.get("message"))

    def test_save_user_info_invalid_age(self):
        """Test saving user information with invalid age."""
        payload = {
            "email_or_phone": "test@example.com",
            "name": "John",
            "age": 17,
            "sex": "Male",
            "address": "123 Main Street",
            "profile_picture": "profile.jpg"
        }
        response = self.app.post('/save-user-info', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Age must be between 18 and 99", response.json.get("error"))

    def test_get_food_preferences(self):
        """Test fetching food preferences."""
        response = self.app.get('/food-preferences')
        self.assertEqual(response.status_code, 200)
        self.assertIn("food_types", response.json)
        self.assertGreater(len(response.json.get("food_types", [])), 0)

    @patch('app.users_collection')
    def test_login_user_valid(self, mock_users_collection):
        """Test user login with valid credentials."""
        mock_users_collection.find_one.return_value = {"email_or_phone": "test@example.com", "password": generate_password_hash("Password@123")}
        payload = {
            "email_or_phone": "test@example.com",
            "password": "Password@123"
        }
        response = self.app.post('/login', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Login successful", response.json.get("message"))

    @patch('app.users_collection')
    def test_login_user_invalid(self, mock_users_collection):
        """Test user login with invalid credentials."""
        mock_users_collection.find_one.return_value = None
        payload = {
            "email_or_phone": "test@example.com",
            "password": "WrongPassword"
        }
        response = self.app.post('/login', json=payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Incorrect email or password", response.json.get("error"))

    def test_logout_user(self):
        """Test user logout."""
        response = self.app.post('/logout')
        self.assertEqual(response.status_code, 200)
        self.assertIn("Logout successful", response.json.get("message"))

    def test_is_valid_email(self):
        """Test email validation utility."""
        self.assertTrue(is_valid_email("test@example.com"))
        self.assertFalse(is_valid_email("invalid-email"))

    def test_is_valid_phone(self):
        """Test phone number validation utility."""
        self.assertTrue(is_valid_phone("123-456-7890"))
        self.assertFalse(is_valid_phone("1234567890"))

    def test_is_valid_password(self):
        """Test password validation utility."""
        self.assertTrue(is_valid_password("Password@123"))
        self.assertFalse(is_valid_password("password"))

    def test_is_valid_name(self):
        """Test name validation utility."""
        self.assertTrue(is_valid_name("John"))
        self.assertFalse(is_valid_name("J"))

    def test_is_valid_age(self):
        """Test age validation utility."""
        self.assertTrue(is_valid_age(25))
        self.assertFalse(is_valid_age(17))

    def test_is_valid_sex(self):
        """Test sex validation utility."""
        self.assertTrue(is_valid_sex("Male"))
        self.assertFalse(is_valid_sex("Unknown"))

    def test_is_valid_address(self):
        """Test address validation utility."""
        self.assertTrue(is_valid_address("123 Main Street"))
        self.assertFalse(is_valid_address("!@#"))

if __name__ == '__main__':
    unittest.main()


#*End of AI Generated Content*