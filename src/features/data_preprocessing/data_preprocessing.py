import pandas as pd
import os

def preprocess_data(ticker: str):
    try:
        # Define file paths
        input_file = os.path.join("src", "data", "raw", f"{ticker}_data.xlsx")
        output_file = os.path.join("src", "data", "processed", f"{ticker}_processed.xlsx")

        # Check if the input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Raw data file not found: {input_file}")

        # Load raw data
        data = pd.read_excel(input_file)

        # Check for a date-related column
        date_column = None
        for col in data.columns:
            if col.lower() in ['date', 'timestamp']:
                date_column = col
                break

        if not date_column:
            raise ValueError("No valid date column found in the dataset.")

        # Rename and parse the Date column
        data.rename(columns={date_column: 'Date'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

        # Check for invalid date formats after coercion
        if data['Date'].isnull().any():
            raise ValueError("Invalid date format detected in the Date column.")

        # Sort data by Date
        data = data.sort_values(by='Date').reset_index(drop=True)

        # Ensure required financial columns
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing_cols = required_columns - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in the dataset: {missing_cols}")

        # Clean numeric columns
        for col in required_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing values
        missing_before = data.isnull().sum()
        data = data.fillna(method='ffill').fillna(method='bfill')  # Forward and backward fill
        missing_after = data.isnull().sum()

        # Log columns that still have missing values after imputation
        if missing_after.any():
            print(f"Columns with remaining missing values after imputation: {missing_after[missing_after > 0]}")

        # Ensure no negative values in Volume
        if 'Volume' in data.columns and (data['Volume'] < 0).any():
            negative_volume_count = (data['Volume'] < 0).sum()
            print(f"Warning: {negative_volume_count} negative values detected in Volume. Setting them to 0.")
            data['Volume'] = data['Volume'].clip(lower=0)

        # Save processed data
        data.to_excel(output_file, index=False)
        print(f"Processed data saved to {output_file}")

        return data

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except ValueError as e:
        print(f"Value error: {e}")
    except pd.errors.EmptyDataError:
        print("The file is empty or cannot be read.")
    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
    return None
