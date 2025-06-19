# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
# -*- coding: utf-8 -*-

import unittest
from your_module import (check_date_column_exists, convert_date_to_datetime, 
                         fill_missing_values, detect_anomalies, 
                         plot_time_series_anomalies, handle_exception)
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # for testing without displaying plots
import matplotlib.pyplot as plt
import datetime
import io
import sys

class TestTimeSeriesAnomalyDetection(unittest.TestCase):

    def test_check_date_column_exists_positive(self):
        """Test check_date_column_exists with existing date column"""
        df = pd.DataFrame({'date': ['2022-01-01'], 'values': [10]})
        self.assertTrue(check_date_column_exists(df))

    def test_check_date_column_exists_negative(self):
        """Test check_date_column_exists with non-existing date column"""
        df = pd.DataFrame({'values': [10]})
        self.assertFalse(check_date_column_exists(df))

    def test_check_date_column_exists_exception(self):
        """Test check_date_column_exists with exception handling"""
        df = None
        try:
            check_date_column_exists(df)
        except Exception as e:
            self.assertIsInstance(e, AttributeError)

    def test_convert_date_to_datetime_success(self):
        """Test convert_date_to_datetime with successful conversion"""
        df = pd.DataFrame({'date': ['2022-01-01'], 'values': [10]})
        converted_df = convert_date_to_datetime(df)
        self.assertIsInstance(converted_df['date'].iloc[0], pd.Timestamp)

    def test_convert_date_to_datetime_failure(self):
        """Test convert_date_to_datetime with conversion failure"""
        df = pd.DataFrame({'date': ['invalid_date'], 'values': [10]})
        try:
            convert_date_to_datetime(df)
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test_fill_missing_values_success(self):
        """Test fill_missing_values with successful filling"""
        df = pd.DataFrame({'values': [10, np.nan, 20]})
        filled_df = fill_missing_values(df)
        self.assertFalse(filled_df.isnull().values.any())

    def test_fill_missing_values_failure(self):
        """Test fill_missing_values with exceeding missing values percentage"""
        df = pd.DataFrame({'values': [np.nan, np.nan, np.nan]})
        try:
            fill_missing_values(df)
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test_detect_anomalies_success(self):
        """Test detect_anomalies with successful anomaly detection"""
        df = pd.DataFrame({'values': [10, 20, 30, 1000]})
        anomaly_df, anomaly_scores = detect_anomalies(df)
        self.assertIn('anomaly', anomaly_df.columns)

    def test_detect_anomalies_failure(self):
        """Test detect_anomalies with IsolationForest failure"""
        df = pd.DataFrame({'values': [np.nan]})
        try:
            detect_anomalies(df)
        except Exception as e:
            self.assertIsInstance(e, ValueError)

    def test_plot_time_series_anomalies_success(self):
        """Test plot_time_series_anomalies with successful plotting"""
        df = pd.DataFrame({'values': [10, 20, 30], 'anomaly': ['No', 'No', 'Yes']})
        anomaly_scores = pd.Series([0.5, 0.5, -0.5])
        # Capture plot output
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        plot_time_series_anomalies(df, anomaly_scores)
        sys.stdout = sys.__stdout__
        # Check if plot was generated without errors
        self.assertEqual(capturedOutput.getvalue(), '')

    def test_handle_exception_success(self):
        """Test handle_exception with successful exception handling"""
        exception = ValueError('Test Exception')
        function_name = 'test_function'
        # Capture print output
        capturedOutput = io.StringIO()
        sys.stdout = capturedOutput
        handle_exception(exception, function_name)
        sys.stdout = sys.__stdout__
        # Check if exception was handled and printed correctly
        self.assertIn('Error in test_function', capturedOutput.getvalue())
        self.assertIn('ValueError: Test Exception', capturedOutput.getvalue())

if __name__ == '__main__':
    unittest.main()


#*End of AI Generated Content*