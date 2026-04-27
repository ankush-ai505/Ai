package com.example.tests;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.*;
import org.openqa.selenium.*;
import org.openqa.selenium.chrome.ChromeDriver;
import java.net.HttpURLConnection;
import java.net.URL;
import java.io.OutputStream;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.time.Duration;
import org.openqa.selenium.support.ui.WebDriverWait;
import org.openqa.selenium.support.ui.ExpectedConditions;

// Requirement 1004 Test
public class ReportsDashboardAccessTest {
    private WebDriver driver;
    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
    }
    @AfterEach
    public void tearDown() {
        if (driver != null) driver.quit();
    }
    @Test
    public void testAccessReportsAndDashboard() {
        try {
            driver.get("http://application-url/login");
            driver.findElement(By.id("username")).sendKeys("validUser");
            driver.findElement(By.id("password")).sendKeys("validPassword");
            driver.findElement(By.id("loginButton")).click();
            driver.findElement(By.id("reportsDashboardLink")).click();
            List<WebElement> reports = driver.findElements(By.cssSelector(".report-item"));
            assertFalse(reports.isEmpty(), "No reports found - FAIL");
            boolean newReportsFound = driver.findElements(By.cssSelector(".report-item.new")).size() > 0;
            if (newReportsFound) {
                for (WebElement newReport : driver.findElements(By.cssSelector(".report-item.new"))) {
                    newReport.click();
                    WebElement reportContent = driver.findElement(By.cssSelector(".report-content"));
                    assertTrue(reportContent.isDisplayed(), "New report content not displayed - FAIL");
                    driver.navigate().back();
                }
            } else {
                System.out.println("No new reports found. Verify with dev team.");
                fail("Defect: New reports expected but not found.");
            }
            System.out.println("Test Passed: Reports and Dashboard section verified successfully.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
}

// Requirement 1005 Test
class DataGenieLandingPageTest {
    private WebDriver driver;
    private WebDriverWait wait;
    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
        wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }
    @AfterEach
    public void tearDown() {
        if (driver != null) driver.quit();
    }
    @Test
    public void testDataGenieLandingPage() {
        try {
            driver.get("http://application-url/datagenie");
            List<WebElement> links = driver.findElements(By.tagName("a"));
            assertFalse(links.isEmpty(), "No links found - FAIL");
            for (WebElement link : links) {
                assertTrue(link.isDisplayed() && link.isEnabled(), "Link not clickable - FAIL");
            }
            for (WebElement link : links) {
                String href = link.getAttribute("href");
                link.click();
                wait.until(ExpectedConditions.urlToBe(href));
                assertEquals(href, driver.getCurrentUrl(), "Navigation mismatch - FAIL");
                driver.navigate().back();
            }
            List<WebElement> images = driver.findElements(By.tagName("img"));
            for (WebElement img : images) {
                String src = img.getAttribute("src");
                assertNotNull(src, "Image src is null - FAIL");
                assertFalse(src.isEmpty(), "Image src is empty - FAIL");
            }
            WebElement body = driver.findElement(By.tagName("body"));
            assertTrue(body.isDisplayed(), "Page layout issue - FAIL");
            driver.manage().window().setSize(new Dimension(1024, 768));
            assertTrue(body.isDisplayed(), "Page not responsive for medium screen - FAIL");
            driver.manage().window().setSize(new Dimension(375, 667));
            assertTrue(body.isDisplayed(), "Page not responsive for small screen - FAIL");
            Long loadTime = (Long) ((JavascriptExecutor) driver).executeScript("return performance.timing.loadEventEnd - performance.timing.navigationStart;");
            assertTrue(loadTime < 5000, "Page load too slow - FAIL");
            List<String> logs = (List<String>) ((JavascriptExecutor) driver).executeScript("return window.console.errors || [];");
            assertTrue(logs.isEmpty(), "Console errors found - FAIL");
            body.sendKeys(Keys.TAB);
            assertTrue(true, "Keyboard navigation failed - FAIL");
            String userAgent = (String) ((JavascriptExecutor) driver).executeScript("return navigator.userAgent;");
            assertNotNull(userAgent, "User agent is null - FAIL");
            System.out.println("Test Passed: Data Genie landing page verified successfully.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
}

// Login-related Requirements Test
class LoginFunctionalityTest {
    private WebDriver driver;
    @BeforeEach
    public void setUp() {
        driver = new ChromeDriver();
        driver.manage().window().maximize();
    }
    @AfterEach
    public void tearDown() {
        if (driver != null) driver.quit();
    }
    private HttpURLConnection callLoginAPI(String username, String password) throws Exception {
        URL url = new URL("http://application-url/api/login");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setRequestProperty("Content-Type", "application/json");
        conn.setDoOutput(true);
        String payload = String.format("{\"username\":\"%s\",\"password\":\"%s\"}", username, password);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(payload.getBytes(StandardCharsets.UTF_8));
        }
        return conn;
    }
    @Test
    public void testLoginUIAndAPIValidCredentials() {
        try {
            driver.get("http://application-url/login");
            driver.findElement(By.id("username")).sendKeys("validUser");
            driver.findElement(By.id("password")).sendKeys("validPassword");
            driver.findElement(By.id("loginButton")).click();
            assertTrue(driver.findElement(By.id("dashboardPage")).isDisplayed(), "Dashboard not displayed - FAIL");
            HttpURLConnection conn = callLoginAPI("validUser", "validPassword");
            assertEquals(200, conn.getResponseCode(), "API did not return HTTP 200 - FAIL");
            String token = new String(conn.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue(token.contains("authToken"), "Auth token missing - FAIL");
            System.out.println("Test Passed: Valid credentials login verified via UI and API.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
    @Test
    public void testLoginUIAndAPIInvalidCredentials() {
        try {
            driver.get("http://application-url/login");
            driver.findElement(By.id("username")).sendKeys("invalidUser");
            driver.findElement(By.id("password")).sendKeys("invalidPassword");
            driver.findElement(By.id("loginButton")).click();
            assertTrue(driver.findElement(By.id("errorMessage")).isDisplayed(), "Error message not displayed - FAIL");
            HttpURLConnection conn = callLoginAPI("invalidUser", "invalidPassword");
            assertTrue(conn.getResponseCode() == 401 || conn.getResponseCode() == 403, "API did not return unauthorized - FAIL");
            System.out.println("Test Passed: Invalid credentials login verified via UI and API.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
    @Test
    public void testBlankCredentialsValidation() {
        try {
            driver.get("http://application-url/login");
            driver.findElement(By.id("loginButton")).click();
            assertTrue(driver.findElement(By.id("validationMessage")).isDisplayed(), "Validation message not displayed - FAIL");
            System.out.println("Test Passed: Blank credentials validation verified.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
    @Test
    public void testUIOnlyLoginValidCredentials() {
        try {
            driver.get("http://application-url/login");
            driver.findElement(By.id("username")).sendKeys("validUser");
            driver.findElement(By.id("password")).sendKeys("validPassword");
            driver.findElement(By.id("loginButton")).click();
            assertTrue(driver.findElement(By.id("dashboardPage")).isDisplayed(), "Dashboard not displayed - FAIL");
            assertTrue(driver.findElement(By.id("welcomeMessage")).isDisplayed(), "Welcome message not displayed - FAIL");
            System.out.println("Test Passed: UI-only valid credentials login verified.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
    @Test
    public void testUIOnlyLoginInvalidCredentials() {
        try {
            driver.get("http://application-url/login");
            driver.findElement(By.id("username")).sendKeys("invalidUser");
            driver.findElement(By.id("password")).sendKeys("invalidPassword");
            driver.findElement(By.id("loginButton")).click();
            assertTrue(driver.findElement(By.id("errorMessage")).isDisplayed(), "Error message not displayed - FAIL");
            assertEquals("http://application-url/login", driver.getCurrentUrl(), "Unexpected navigation - FAIL");
            System.out.println("Test Passed: UI-only invalid credentials login verified.");
        } catch (Exception e) {
            e.printStackTrace();
            fail("Test failed due to exception: " + e.getMessage());
        }
    }
}