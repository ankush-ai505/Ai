# Disclaimer: This output contains AI-generated content; user is advised to review it before consumption.
#*Start of AI Generated Content*

python
# -*- coding: utf-8 -*-

# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from datetime import datetime

# Constants and Variables
MAX_MISSING_VALUES_PERCENT = 0.3
DATE_COLUMN_NAME = 'date'
TIME_SERIES_COLUMN_NAME = 'values'
ANOMALY_THRESHOLD = 0.05
PLOT_TITLE = 'Time Series Anomaly Detection'
PLOT_X_LABEL = 'Time'
PLOT_Y_LABEL = 'Values'

# Function to check if date column exists
def check_date_column_exists(df: pd.DataFrame) -> bool:
    """
    Check if date column exists in the DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    bool: True if date column exists, False otherwise.
    """
    try:
        return DATE_COLUMN_NAME in df.columns
    except Exception as e:
        handle_exception(e, 'check_date_column_exists')

# Function to convert date column to datetime format
def convert_date_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date column to datetime format.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with date column in datetime format.
    """
    try:
        df[DATE_COLUMN_NAME] = pd.to_datetime(df[DATE_COLUMN_NAME])
        return df
    except Exception as e:
        handle_exception(e, 'convert_date_to_datetime')

# Function to fill missing values
def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values in the DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with filled missing values.
    """
    try:
        missing_values_percent = df.isnull().mean().max()
        if missing_values_percent > MAX_MISSING_VALUES_PERCENT:
            raise ValueError('Missing values exceed the maximum allowed percentage')
        df.fillna(df.mean(), inplace=True)
        return df
    except Exception as e:
        handle_exception(e, 'fill_missing_values')

# Function to detect anomalies in time series data
def detect_anomalies(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Detect anomalies in time series data using Isolation Forest.

    Args:
    df (pd.DataFrame): Input DataFrame with time series data.

    Returns:
    pd.DataFrame: DataFrame with anomaly detection results.
    pd.Series: Anomaly scores.
    """
    try:
        X_train, X_test = train_test_split(df[TIME_SERIES_COLUMN_NAME], test_size=0.2, random_state=42)
        iso_forest = IsolationForest(contamination=ANOMALY_THRESHOLD, random_state=42)
        iso_forest.fit(X_train.values.reshape(-1, 1))
        anomaly_scores = iso_forest.decision_function(X_test.values.reshape(-1, 1))
        df_anomaly = pd.DataFrame({TIME_SERIES_COLUMN_NAME: X_test, 'anomaly_score': anomaly_scores})
        df_anomaly['anomaly'] = df_anomaly['anomaly_score'].apply(lambda x: 'Yes' if x < 0 else 'No')
        return df_anomaly, pd.Series(anomaly_scores)
    except Exception as e:
        handle_exception(e, 'detect_anomalies')

# Function to plot time series with anomalies
def plot_time_series_anomalies(df: pd.DataFrame, anomaly_scores: pd.Series) -> None:
    """
    Plot time series data with anomalies.

    Args:
    df (pd.DataFrame): DataFrame with time series data and anomaly detection results.
    anomaly_scores (pd.Series): Anomaly scores.
    """
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(df[TIME_SERIES_COLUMN_NAME], label='Time Series')
        plt.scatter(df.index, df.loc[df['anomaly'] == 'Yes', TIME_SERIES_COLUMN_NAME], color='red', label='Anomalies')
        plt.title(PLOT_TITLE)
        plt.xlabel(PLOT_X_LABEL)
        plt.ylabel(PLOT_Y_LABEL)
        plt.legend()
        plt.show()
    except Exception as e:
        handle_exception(e, 'plot_time_series_anomalies')

# Function to handle exceptions
def handle_exception(exception: Exception, function_name: str) -> None:
    """
    Handle exceptions and print error messages.

    Args:
    exception (Exception): Caught exception.
    function_name (str): Name of the function where the exception occurred.
    """
    print(f"Error in {function_name}: {str(exception)}")
    print(f"Error Type: {type(exception).__name__}")
    print("Error Timestamp: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Main Function
def main() -> None:
    try:
        # Load time series data
        df = pd.read_csv('time_series_data.csv')

        # Check if date column exists
        if not check_date_column_exists(df):
            raise ValueError('Date column does not exist in the DataFrame')

        # Convert date column to datetime format
        df = convert_date_to_datetime(df)

        # Fill missing values
        df = fill_missing_values(df)

        # Detect anomalies
        df_anomaly, anomaly_scores = detect_anomalies(df)

        # Plot time series with anomalies
        plot_time_series_anomalies(df_anomaly, anomaly_scores)
    except Exception as e:
        handle_exception(e, 'main')

if __name__ == "__main__":
    main()


#*End of AI Generated Content*