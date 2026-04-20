import unittest
import time
from adls_utility import ADLSUtility


class TestADLSUtility(unittest.TestCase):
    """Unit test cases for ADLSUtility class covering positive, negative,
    boundary, performance, security, and usability scenarios."""

    def setUp(self):
        """Setup prerequisites for ADLSUtility tests."""
        try:
            self.utility = ADLSUtility()
        except Exception as e:
            self.fail(f"Setup failed: {e}")

    def test_upload_file_positive(self):
        """Test uploading a valid file to ADLS."""
        try:
            result = self.utility.upload_file("test_data/sample.txt",
                                              "container/sample.txt")
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"Positive upload test failed: {e}")

    def test_upload_file_negative(self):
        """Test uploading a non-existent file to ADLS."""
        try:
            result = self.utility.upload_file("invalid/path.txt",
                                              "container/path.txt")
            self.assertFalse(result)
        except Exception as e:
            self.assertIsInstance(e, FileNotFoundError)

    def test_upload_file_boundary(self):
        """Test uploading an empty file to ADLS."""
        try:
            result = self.utility.upload_file("test_data/empty.txt",
                                              "container/empty.txt")
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"Boundary upload test failed: {e}")

    def test_download_file_positive(self):
        """Test downloading a valid file from ADLS."""
        try:
            result = self.utility.download_file("container/sample.txt",
                                                "downloads/sample.txt")
            self.assertTrue(result)
        except Exception as e:
            self.fail(f"Positive download test failed: {e}")

    def test_download_file_negative(self):
        """Test downloading a non-existent file from ADLS."""
        try:
            result = self.utility.download_file("container/missing.txt",
                                                "downloads/missing.txt")
            self.assertFalse(result)
        except Exception as e:
            self.assertIsInstance(e, FileNotFoundError)

    def test_list_files_positive(self):
        """Test listing files in a valid container."""
        try:
            files = self.utility.list_files("container")
            self.assertIsInstance(files, list)
        except Exception as e:
            self.fail(f"Positive list files test failed: {e}")

    def test_list_files_negative(self):
        """Test listing files in a non-existent container."""
        try:
            files = self.utility.list_files("invalid_container")
            self.assertEqual(files, [])
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test_performance_upload(self):
        """Performance test for uploading a file."""
        try:
            start_time = time.time()
            self.utility.upload_file("test_data/sample.txt",
                                     "container/sample_perf.txt")
            duration = time.time() - start_time
            self.assertLess(duration, 5)
        except Exception as e:
            self.fail(f"Performance upload test failed: {e}")

    def test_security_access(self):
        """Security test for unauthorized access."""
        try:
            self.utility.set_credentials("invalid_key", "invalid_secret")
            result = self.utility.list_files("container")
            self.assertEqual(result, [])
        except Exception as e:
            self.assertIsInstance(e, PermissionError)

    def test_usability_error_message(self):
        """Usability test for clear error messages."""
        try:
            result = self.utility.download_file("container/missing.txt",
                                                "downloads/missing.txt")
            self.assertFalse(result)
        except Exception as e:
            self.assertIn("not found", str(e).lower())


if __name__ == "__main__":
    unittest.main()