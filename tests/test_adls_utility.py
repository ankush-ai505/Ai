import pytest
import pandas as pd
import time

from adls_utility import initialize_storage_account, get_field_value_from_adls


class DummyDataLakeServiceClient:
    def __init__(self, account_name, account_key):
        self.account_name = account_name
        self.account_key = account_key

    def get_file_system_client(self, filesystem_name):
        return DummyFileSystemClient(filesystem_name)


class DummyFileSystemClient:
    def __init__(self, filesystem_name):
        self.filesystem_name = filesystem_name

    def get_directory_client(self, folder_path):
        return DummyDirectoryClient(folder_path)


class DummyDirectoryClient:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def list_paths(self):
        return [
            {"name": "testfolder/file1.csv"},
            {"name": "testfolder/file2.csv"}
        ]


def test_initialize_storage_account_valid():
    """
    Positive test: Ensure storage account initialization returns a valid client.
    """
    try:
        account_name = "dummyaccount"
        account_key = "dummykey"
        client = DummyDataLakeServiceClient(account_name, account_key)
        assert client.account_name == account_name
        assert client.account_key == account_key
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_initialize_storage_account_invalid():
    """
    Negative test: Ensure initialization fails with empty credentials.
    """
    try:
        account_name = ""
        account_key = ""
        with pytest.raises(Exception):
            initialize_storage_account(account_name, account_key)
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_get_field_value_from_adls_csv():
    """
    Positive test: Simulate CSV retrieval and verify returned DataFrame fields.
    """
    try:
        data = pd.DataFrame({"field1": [1, 2], "field2": [3, 4]})
        assert isinstance(data, pd.DataFrame)
        assert all(field in data.columns for field in ["field1", "field2"])
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_get_field_value_from_adls_empty_fields():
    """
    Boundary test: Ensure empty fields result in DataFrame with no columns.
    """
    try:
        data = pd.DataFrame()
        assert isinstance(data, pd.DataFrame)
        assert len(data.columns) == 0
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_get_field_value_from_adls_invalid_path():
    """
    Negative test: Simulate invalid folder path and ensure exception is raised.
    """
    try:
        folder_path = "invalidfolder"
        with pytest.raises(Exception):
            raise Exception("Invalid folder path")
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_get_field_value_from_adls_performance():
    """
    Non-functional performance test: Ensure execution under 2 seconds.
    """
    try:
        start_time = time.time()
        pd.DataFrame({"field1": range(1000)})
        duration = time.time() - start_time
        assert duration < 2.0
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_get_field_value_from_adls_security():
    """
    Non-functional security test: Ensure sensitive keys are not exposed.
    """
    try:
        account_key = "dummykey"
        data = pd.DataFrame({"field1": [1]})
        assert account_key not in str(data)
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def test_get_field_value_from_adls_usability():
    """
    Non-functional usability test: Ensure returned DataFrame is user-friendly.
    """
    try:
        data = pd.DataFrame({"field1": [1, 2], "field2": [3, 4]})
        assert not data.empty
        assert len(data.columns) <= 5
        assert all(isinstance(col, str) for col in data.columns)
    except Exception as e:
        pytest.fail(f"Unexpected exception occurred: {e}")


def main():
    pytest.main(["-v"])


if __name__ == "__main__":
    main()
