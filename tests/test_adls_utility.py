import unittest
import time
import os
import pandas as pd
from Ai import adls_utility


class TestAdlsUtility(unittest.TestCase):
    """
    Unit test cases for adls_utility.py functions covering positive,
    negative, boundary, performance, security, and usability scenarios.
    """

    def test_initialize_storage_account_positive(self):
        """
        Positive test case for initialize_storage_account.
        Verifies that a valid connection object is returned when
        correct parameters are provided.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            conn = adls_utility.initialize_storage_account(
                account_name, account_key
            )
            self.assertIsNotNone(conn)
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_initialize_storage_account_negative(self):
        """
        Negative test case for initialize_storage_account.
        Verifies that an exception is raised when invalid parameters
        are provided.
        """
        try:
            account_name = ""
            account_key = ""
            with self.assertRaises(Exception):
                adls_utility.initialize_storage_account(
                    account_name, account_key
                )
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_initialize_storage_account_boundary(self):
        """
        Boundary test case for initialize_storage_account.
        Tests with minimal valid account name and key lengths.
        """
        try:
            account_name = "a"
            account_key = "b"
            conn = adls_utility.initialize_storage_account(
                account_name, account_key
            )
            self.assertIsNotNone(conn)
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_get_field_value_from_adls_positive(self):
        """
        Positive test case for get_field_value_from_adls.
        Verifies that a DataFrame is returned for a valid file path
        and delimiter.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            file_path = "tests/data/sample.csv"
            delimiter = ","
            fields = ["column1", "column2"]
            df = adls_utility.get_field_value_from_adls(
                account_name, account_key, file_path, delimiter, fields
            )
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_get_field_value_from_adls_negative(self):
        """
        Negative test case for get_field_value_from_adls.
        Verifies that an exception is raised for a non-existent file.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            file_path = "non_existent.csv"
            delimiter = ","
            fields = ["column1"]
            with self.assertRaises(Exception):
                adls_utility.get_field_value_from_adls(
                    account_name, account_key, file_path, delimiter, fields
                )
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_get_field_value_from_adls_boundary(self):
        """
        Boundary test case for get_field_value_from_adls.
        Tests with an empty fields list to ensure proper handling.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            file_path = "tests/data/sample.csv"
            delimiter = ","
            fields = []
            df = adls_utility.get_field_value_from_adls(
                account_name, account_key, file_path, delimiter, fields
            )
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_get_field_value_from_adls_performance(self):
        """
        Performance test case for get_field_value_from_adls.
        Measures execution time for reading a large file.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            file_path = "tests/data/large.csv"
            delimiter = ","
            fields = ["column1", "column2"]
            start_time = time.time()
            df = adls_utility.get_field_value_from_adls(
                account_name, account_key, file_path, delimiter, fields
            )
            end_time = time.time()
            self.assertLess(end_time - start_time, 5)
            self.assertIsInstance(df, pd.DataFrame)
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_get_field_value_from_adls_security(self):
        """
        Security test case for get_field_value_from_adls.
        Ensures that sensitive data is not exposed in returned DataFrame.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            file_path = "tests/data/sample.csv"
            delimiter = ","
            fields = ["column1", "column2"]
            df = adls_utility.get_field_value_from_adls(
                account_name, account_key, file_path, delimiter, fields
            )
            for col in df.columns:
                self.assertNotIn("password", col.lower())
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")

    def test_get_field_value_from_adls_usability(self):
        """
        Usability test case for get_field_value_from_adls.
        Ensures that returned DataFrame has user-friendly column names.
        """
        try:
            account_name = "valid_account"
            account_key = "valid_key"
            file_path = "tests/data/sample.csv"
            delimiter = ","
            fields = ["column1", "column2"]
            df = adls_utility.get_field_value_from_adls(
                account_name, account_key, file_path, delimiter, fields
            )
            for col in df.columns:
                self.assertTrue(col.isidentifier())
        except Exception as e:
            self.fail(f"Unexpected exception occurred: {e}")


if __name__ == "__main__":
    unittest.main()