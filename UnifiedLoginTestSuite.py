import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import time

class UnifiedLoginTestSuite(unittest.TestCase):
    def setUp(self):
        """Setup Chrome WebDriver and test data."""
        self.driver = webdriver.Chrome()
        self.base_url = "http://example.com"
        self.api_url = "http://api.example.com/login"
        self.valid_username = "validUser"
        self.valid_password = "validPass"
        self.invalid_username = "invalidUser"
        self.invalid_password = "invalidPass"

    def test_successful_login_ui_and_api(self):
        """Requirement 1001: Verify successful login via UI and API."""
        try:
            driver = self.driver
            driver.get(f"{self.base_url}/login")
            driver.find_element(By.ID, "username").send_keys(self.valid_username)
            driver.find_element(By.ID, "password").send_keys(self.valid_password)
            driver.find_element(By.ID, "loginBtn").click()
            time.sleep(2)
            self.assertIn("dashboard", driver.current_url.lower(), "UI did not navigate to dashboard")

            response = requests.post(self.api_url, json={"username": self.valid_username, "password": self.valid_password})
            self.assertEqual(response.status_code, 200, "API did not return HTTP 200")
            self.assertIn("token", response.json(), "API response missing token")
        except Exception as e:
            self.fail(f"Exception occurred in successful login test: {e}")

    def test_unsuccessful_login_ui_and_api(self):
        """Requirement 1001: Verify unsuccessful login via UI and API."""
        try:
            driver = self.driver
            driver.get(f"{self.base_url}/login")
            driver.find_element(By.ID, "username").send_keys(self.invalid_username)
            driver.find_element(By.ID, "password").send_keys(self.invalid_password)
            driver.find_element(By.ID, "loginBtn").click()
            time.sleep(2)
            self.assertIn("error", driver.page_source.lower(), "UI did not show error message")

            response = requests.post(self.api_url, json={"username": self.invalid_username, "password": self.invalid_password})
            self.assertIn(response.status_code, [401, 403], "API did not return unauthorized status")
        except Exception as e:
            self.fail(f"Exception occurred in unsuccessful login test: {e}")

    def test_blank_fields_validation(self):
        """Requirement 1001: Verify blank field validation in UI."""
        try:
            driver = self.driver
            driver.get(f"{self.base_url}/login")
            driver.find_element(By.ID, "loginBtn").click()
            time.sleep(1)
            self.assertIn("required", driver.page_source.lower(), "UI did not show required field message")
        except Exception as e:
            self.fail(f"Exception occurred in blank field validation test: {e}")

    def test_api_error_handling(self):
        """Requirement 1001: Verify API error handling."""
        try:
            response = requests.post(self.api_url, json={"username": "timeoutUser", "password": "timeoutPass"}, timeout=1)
            self.assertIn(response.status_code, [500, 504], "API did not handle server error properly")
        except requests.exceptions.RequestException as e:
            self.fail(f"API request exception: {e}")

    def test_dashboard_redirection_and_session(self):
        """Requirement 1002: Verify dashboard redirection and session establishment."""
        try:
            driver = self.driver
            driver.get(f"{self.base_url}/login")
            driver.find_element(By.ID, "username").send_keys(self.valid_username)
            driver.find_element(By.ID, "password").send_keys(self.valid_password)
            driver.find_element(By.ID, "loginBtn").click()
            time.sleep(2)
            self.assertIn("dashboard", driver.current_url.lower(), "User not redirected to dashboard")

            profile_icon = driver.find_element(By.ID, "profileIcon")
            self.assertTrue(profile_icon.is_displayed(), "Profile icon not displayed, session may not be established")
        except Exception as e:
            self.fail(f"Exception occurred in dashboard redirection test: {e}")

    def test_failed_login_error_message_and_navigation(self):
        """Requirement 1003: Verify error message and navigation on failed login."""
        try:
            driver = self.driver
            driver.get(f"{self.base_url}/login")
            driver.find_element(By.ID, "username").send_keys(self.invalid_username)
            driver.find_element(By.ID, "password").send_keys(self.invalid_password)
            driver.find_element(By.ID, "loginBtn").click()
            time.sleep(2)

            error_message = driver.find_element(By.ID, "errorMsg")
            self.assertTrue(error_message.is_displayed(), "Error message not displayed for failed login")
            self.assertIn("login", driver.current_url.lower(), "User navigated away from login page after failed login")
        except Exception as e:
            self.fail(f"Exception occurred in failed login UI test: {e}")

    def tearDown(self):
        """Close the WebDriver."""
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()