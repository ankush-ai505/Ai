import pytest
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Requirement ID 1001: Login Functionality
BASE_LOGIN_URL = "https://example.com/api/login"
UI_LOGIN_URL = "https://example.com/login"

@pytest.fixture(scope="module")
def driver():
    driver = webdriver.Chrome()
    yield driver
    driver.quit()

def test_successful_login_ui_and_api(driver):
    driver.get(UI_LOGIN_URL)
    driver.find_element(By.ID, "username").send_keys("validUser")
    driver.find_element(By.ID, "password").send_keys("validPass")
    driver.find_element(By.ID, "loginBtn").click()
    assert "Dashboard" in driver.title, "UI login failed"

    response = requests.post(BASE_LOGIN_URL, json={"username": "validUser", "password": "validPass"})
    assert response.status_code == 200, "API login failed"
    assert "token" in response.json(), "Authentication token missing"

def test_unsuccessful_login_invalid_credentials(driver):
    driver.get(UI_LOGIN_URL)
    driver.find_element(By.ID, "username").send_keys("invalidUser")
    driver.find_element(By.ID, "password").send_keys("invalidPass")
    driver.find_element(By.ID, "loginBtn").click()
    assert "Invalid credentials" in driver.page_source, "UI error message missing"

    response = requests.post(BASE_LOGIN_URL, json={"username": "invalidUser", "password": "invalidPass"})
    assert response.status_code in [401, 403], "API error code incorrect"

def test_blank_username_password(driver):
    driver.get(UI_LOGIN_URL)
    driver.find_element(By.ID, "loginBtn").click()
    assert "Username required" in driver.page_source
    assert "Password required" in driver.page_source

def test_api_error_handling():
    try:
        requests.post(BASE_LOGIN_URL, json={"username": "validUser", "password": "validPass"}, timeout=0.001)
    except requests.exceptions.Timeout:
        pytest.fail("API timeout occurred")

    response = requests.post(BASE_LOGIN_URL, json={"username": "unauthorizedUser", "password": "pass"})
    assert response.status_code == 401, "Unauthorized access not handled"

    response = requests.post(BASE_LOGIN_URL, json={"username": "validUser", "password": "pass"})
    if response.status_code == 500:
        pytest.fail("Internal server error occurred")

def test_ui_api_consistency(driver):
    driver.get(UI_LOGIN_URL)
    driver.find_element(By.ID, "username").send_keys("validUser")
    driver.find_element(By.ID, "password").send_keys("validPass")
    driver.find_element(By.ID, "loginBtn").click()
    ui_message = driver.find_element(By.ID, "welcomeMsg").text

    response = requests.post(BASE_LOGIN_URL, json={"username": "validUser", "password": "validPass"})
    api_message = response.json().get("message", "")
    assert ui_message == api_message, "UI and API messages inconsistent"

# Requirement ID 1002: Password Reset Functionality
BASE_RESET_URL = "https://example.com/api/reset-password"
UI_RESET_URL = "https://example.com/reset-password"

def test_password_reset_ui_and_api(driver):
    driver.get(UI_RESET_URL)
    driver.find_element(By.ID, "email").send_keys("user@example.com")
    driver.find_element(By.ID, "resetBtn").click()
    assert "Reset link sent" in driver.page_source, "UI reset message missing"

    response = requests.post(BASE_RESET_URL, json={"email": "user@example.com"})
    assert response.status_code == 200, "API reset request failed"
    assert "token" in response.json(), "Reset token missing"

def test_password_reset_invalid_email(driver):
    driver.get(UI_RESET_URL)
    driver.find_element(By.ID, "email").send_keys("invalid@example")
    driver.find_element(By.ID, "resetBtn").click()
    assert "Invalid email" in driver.page_source, "UI invalid email message missing"

    response = requests.post(BASE_RESET_URL, json={"email": "invalid@example"})
    assert response.status_code == 400, "API did not handle invalid email correctly"

def test_password_update_with_token():
    token = "validResetToken"
    response = requests.put(BASE_RESET_URL, json={"token": token, "new_password": "NewPass123"})
    assert response.status_code == 200, "Password update failed"
    assert response.json().get("message") == "Password updated successfully"

def test_password_update_with_invalid_token():
    token = "invalidToken"
    response = requests.put(BASE_RESET_URL, json={"token": token, "new_password": "NewPass123"})
    assert response.status_code == 403, "Invalid token not handled"

# Requirement ID 1003: Profile Update Functionality
BASE_PROFILE_URL = "https://example.com/api/update-profile"
UI_PROFILE_URL = "https://example.com/profile"

def test_profile_update_ui_and_api(driver):
    driver.get(UI_PROFILE_URL)
    driver.find_element(By.ID, "name").clear()
    driver.find_element(By.ID, "name").send_keys("John Doe")
    driver.find_element(By.ID, "phone").clear()
    driver.find_element(By.ID, "phone").send_keys("1234567890")
    driver.find_element(By.ID, "updateBtn").click()
    assert "Profile updated" in driver.page_source, "UI update message missing"

    response = requests.put(BASE_PROFILE_URL, json={"name": "John Doe", "phone": "1234567890"})
    assert response.status_code == 200, "API profile update failed"
    assert response.json().get("message") == "Profile updated successfully"

def test_profile_update_invalid_data(driver):
    driver.get(UI_PROFILE_URL)
    driver.find_element(By.ID, "phone").clear()
    driver.find_element(By.ID, "phone").send_keys("abcde")
    driver.find_element(By.ID, "updateBtn").click()
    assert "Invalid phone number" in driver.page_source, "UI invalid phone message missing"

    response = requests.put(BASE_PROFILE_URL, json={"name": "John Doe", "phone": "abcde"})
    assert response.status_code == 400, "API did not handle invalid phone correctly"

def test_profile_update_missing_fields():
    response = requests.put(BASE_PROFILE_URL, json={"name": ""})
    assert response.status_code == 400, "Missing fields not handled"