# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

markdown
python
# -*- coding: utf-8 -*-

import unittest
from your_module import app, mongo, COLLECTION_USERS, TOP_FOOD_TYPES  # Replace 'your_module' with the actual module name

class TestUserRegistrationAPI(unittest.TestCase):
    """
    Unit Test Cases for User Registration and Management API
    """

    def setUp(self):
        """
        Setup method to initialize the test client and database
        """
        self.app = app.test_client()
        self.mongo = mongo

    def tearDown(self):
        """
        Teardown method to drop the test database after each test
        """
        self.mongo.db.drop_collection(COLLECTION_USERS)

    def test_register_valid_user(self):
        """
        Test registering a valid user with all required fields

        :input: 
            - Valid email
            - Valid password (meets the password regex criteria)
            - Matching confirm password
            - Security question

        :expected output: 
            - HTTP Status Code: 201
            - JSON Response: 'message' key with 'User registered successfully' value
        """
        try:
            data = {
                'email': 'test@example.com',
                'password': 'Test@123',
                'confirm_password': 'Test@123',
                'security_question': 'What is your favorite color?'
            }
            response = self.app.post('/register', json=data)
            self.assertEqual(response.status_code, 201)
            self.assertIn('message', response.json)
            self.assertEqual(response.json['message'], 'User registered successfully')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_register_invalid_email(self):
        """
        Test registering a user with an invalid email

        :input: 
            - Invalid email
            - Valid password
            - Matching confirm password
            - Security question

        :expected output: 
            - HTTP Status Code: 400
            - JSON Response: 'error' key with 'Invalid email or phone number' value
        """
        try:
            data = {
                'email': 'invalid_email',
                'password': 'Test@123',
                'confirm_password': 'Test@123',
                'security_question': 'What is your favorite color?'
            }
            response = self.app.post('/register', json=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn('error', response.json)
            self.assertEqual(response.json['error'], 'Invalid email or phone number')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_register_password_mismatch(self):
        """
        Test registering a user with non-matching passwords

        :input: 
            - Valid email
            - Valid password
            - Non-matching confirm password
            - Security question

        :expected output: 
            - HTTP Status Code: 400
            - JSON Response: 'error' key with 'Passwords do not match' value
        """
        try:
            data = {
                'email': 'test@example.com',
                'password': 'Test@123',
                'confirm_password': 'MismatchedPassword',
                'security_question': 'What is your favorite color?'
            }
            response = self.app.post('/register', json=data)
            self.assertEqual(response.status_code, 400)
            self.assertIn('error', response.json)
            self.assertEqual(response.json['error'], 'Passwords do not match')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_forgot_password_valid_user(self):
        """
        Test forgot password for a valid user with correct security question answer

        :input: 
            - Valid email
            - Correct security question answer

        :expected output: 
            - HTTP Status Code: 200
            - JSON Response: 'message' key with 'Password reset link sent successfully' value
        """
        try:
            # Register a user first
            register_data = {
                'email': 'test@example.com',
                'password': 'Test@123',
                'confirm_password': 'Test@123',
                'security_question': 'What is your favorite color?',
                'security_question_answer': 'Blue'
            }
            self.app.post('/register', json=register_data)

            # Test forgot password
            data = {
                'email': 'test@example.com',
                'security_question_answer': 'Blue'
            }
            response = self.app.post('/forgot-password', json=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn('message', response.json)
            self.assertEqual(response.json['message'], 'Password reset link sent successfully')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_save_personal_info_valid_user(self):
        """
        Test saving personal info for a valid user

        :input: 
            - Valid email
            - Valid name
            - Valid age
            - Valid sex
            - Valid address

        :expected output: 
            - HTTP Status Code: 200
            - JSON Response: 'message' key with 'Personal information saved successfully' value
        """
        try:
            # Register a user first
            register_data = {
                'email': 'test@example.com',
                'password': 'Test@123',
                'confirm_password': 'Test@123',
                'security_question': 'What is your favorite color?'
            }
            self.app.post('/register', json=register_data)

            # Test saving personal info
            data = {
                'email': 'test@example.com',
                'name': 'John Doe',
                'age': 25,
                'sex': 'Male',
                'address': '123 Main St'
            }
            response = self.app.post('/save-personal-info', json=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn('message', response.json)
            self.assertEqual(response.json['message'], 'Personal information saved successfully')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_get_food_types(self):
        """
        Test getting top food types

        :expected output: 
            - HTTP Status Code: 200
            - JSON Response: 'food_types' key with a list of top food types
        """
        try:
            response = self.app.get('/get-food-types')
            self.assertEqual(response.status_code, 200)
            self.assertIn('food_types', response.json)
            self.assertEqual(response.json['food_types'], TOP_FOOD_TYPES)
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_login_valid_user(self):
        """
        Test logging in a valid user with correct credentials

        :input: 
            - Valid email
            - Correct password

        :expected output: 
            - HTTP Status Code: 200
            - JSON Response: 'message' key with 'User logged in successfully' value
        """
        try:
            # Register a user first
            register_data = {
                'email': 'test@example.com',
                'password': 'Test@123',
                'confirm_password': 'Test@123',
                'security_question': 'What is your favorite color?'
            }
            self.app.post('/register', json=register_data)

            # Test login
            data = {
                'email': 'test@example.com',
                'password': 'Test@123'
            }
            response = self.app.post('/login', json=data)
            self.assertEqual(response.status_code, 200)
            self.assertIn('message', response.json)
            self.assertEqual(response.json['message'], 'User logged in successfully')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

    def test_logout_valid_user(self):
        """
        Test logging out a valid user

        :expected output: 
            - HTTP Status Code: 200
            - JSON Response: 'message' key with 'User logged out successfully' value
        """
        try:
            # Register and login a user first
            register_data = {
                'email': 'test@example.com',
                'password': 'Test@123',
                'confirm_password': 'Test@123',
                'security_question': 'What is your favorite color?'
            }
            self.app.post('/register', json=register_data)
            self.app.post('/login', json={'email': 'test@example.com', 'password': 'Test@123'})

            # Test logout
            response = self.app.post('/logout')
            self.assertEqual(response.status_code, 200)
            self.assertIn('message', response.json)
            self.assertEqual(response.json['message'], 'User logged out successfully')
        except Exception as e:
            self.fail(f"Test failed with exception: {str(e)}")

if __name__ == '__main__':
    unittest.main()


#*End of AI Generated Content*