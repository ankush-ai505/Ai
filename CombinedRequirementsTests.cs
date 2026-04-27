using NUnit.Framework;
using OpenQA.Selenium;
using OpenQA.Selenium.Chrome;
using System;
using System.Net.Http;
using System.Threading.Tasks;

namespace CombinedRequirementsTests
{
    [TestFixture]
    public class ApplicationTests
    {
        private IWebDriver driver;
        private HttpClient httpClient;

        [SetUp]
        public void Setup()
        {
            driver = new ChromeDriver();
            httpClient = new HttpClient();
        }

        [TearDown]
        public void Teardown()
        {
            driver.Quit();
            httpClient.Dispose();
        }

        [Test]
        public void TestReportsAndDashboardAccess()
        {
            try
            {
                driver.Navigate().GoToUrl("http://application-url/login");
                // Login steps here...
                driver.Navigate().GoToUrl("http://application-url/reports-dashboard");

                var reports = driver.FindElements(By.CssSelector(".report-item"));
                Assert.IsTrue(reports.Count > 0, "No reports found");

                var newReports = driver.FindElements(By.CssSelector(".report-item.new"));
                if (newReports.Count > 0)
                {
                    foreach (var report in newReports)
                    {
                        Assert.IsTrue(report.Displayed, $"Report {report.Text} not displayed properly");
                    }
                }
                else
                {
                    Assert.Fail("No new reports found - verify with development team");
                }
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception occurred: {ex.Message}");
            }
        }

        [Test]
        public void TestDataGenieLandingPage()
        {
            try
            {
                driver.Navigate().GoToUrl("http://application-url/data-genie");

                var links = driver.FindElements(By.TagName("a"));
                foreach (var link in links)
                {
                    Assert.IsTrue(link.Displayed && link.Enabled, $"Link {link.Text} is not functional");
                }

                var images = driver.FindElements(By.TagName("img"));
                foreach (var img in images)
                {
                    Assert.IsTrue(img.Displayed, "Image not loaded correctly");
                }

                Assert.IsTrue(driver.FindElement(By.TagName("body")).Displayed, "Page layout issue");

                driver.Manage().Window.Size = new System.Drawing.Size(800, 600);
                Assert.IsTrue(driver.FindElement(By.TagName("body")).Displayed, "Page not responsive");
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception occurred: {ex.Message}");
            }
        }

        [Test]
        public async Task TestLoginUIAndAPI()
        {
            try
            {
                driver.Navigate().GoToUrl("http://application-url/login");
                driver.FindElement(By.Id("username")).SendKeys("validUser");
                driver.FindElement(By.Id("password")).SendKeys("validPass");
                driver.FindElement(By.Id("loginBtn")).Click();

                Assert.IsTrue(driver.Url.Contains("dashboard"), "UI did not navigate to dashboard");

                var response = await httpClient.PostAsync("http://api-url/login",
                    new StringContent("{\"username\":\"validUser\",\"password\":\"validPass\"}", System.Text.Encoding.UTF8, "application/json"));
                Assert.AreEqual(System.Net.HttpStatusCode.OK, response.StatusCode, "API did not return success");

                var token = await response.Content.ReadAsStringAsync();
                Assert.IsNotNull(token, "Authentication token missing");

                var invalidResponse = await httpClient.PostAsync("http://api-url/login",
                    new StringContent("{\"username\":\"invalid\",\"password\":\"wrong\"}", System.Text.Encoding.UTF8, "application/json"));
                Assert.AreEqual(System.Net.HttpStatusCode.Unauthorized, invalidResponse.StatusCode, "Invalid credentials not handled");
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception occurred: {ex.Message}");
            }
        }

        [Test]
        public void TestLoginUIOnly()
        {
            try
            {
                driver.Navigate().GoToUrl("http://application-url/login");
                driver.FindElement(By.Id("username")).SendKeys("validUser");
                driver.FindElement(By.Id("password")).SendKeys("validPass");
                driver.FindElement(By.Id("loginBtn")).Click();

                Assert.IsTrue(driver.Url.Contains("dashboard"), "UI did not navigate to dashboard");
                Assert.IsTrue(driver.FindElement(By.Id("welcomeMessage")).Displayed, "Welcome message not displayed");

                driver.Navigate().GoToUrl("http://application-url/login");
                driver.FindElement(By.Id("username")).SendKeys("invalidUser");
                driver.FindElement(By.Id("password")).SendKeys("wrongPass");
                driver.FindElement(By.Id("loginBtn")).Click();

                Assert.IsTrue(driver.Url.Contains("login"), "User navigated away after failed login");
                Assert.IsTrue(driver.FindElement(By.Id("errorMessage")).Displayed, "Error message not displayed");
            }
            catch (Exception ex)
            {
                Assert.Fail($"Exception occurred: {ex.Message}");
            }
        }
    }
}