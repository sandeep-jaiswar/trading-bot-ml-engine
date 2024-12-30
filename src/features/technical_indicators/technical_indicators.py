import pandas as pd
import numpy as np
import talib
import os

def add_technical_indicators(ticker: str):
    try:
        # Define file paths
        input_file = os.path.join("src", "data", "processed", f"{ticker}_processed.xlsx")
        output_file = os.path.join("src", "data", "processed", f"{ticker}_with_indicators.xlsx")

        # Load the processed data
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found.")

        data = pd.read_excel(input_file)

        # Validate required columns
        required_columns = {'High', 'Low', 'Close', 'Volume'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Input data is missing one or more required columns: {required_columns}")

        # Convert columns to numeric and handle errors
        for column in required_columns:
            data[column] = pd.to_numeric(data[column], errors='coerce')

        # Handle missing data
        if data.isnull().any().any():
            print("Missing or invalid data detected. Filling missing values...")
            data = data.fillna(method="ffill").fillna(method="bfill")

        # Add technical indicators
        data['SMA'] = data['Close'].rolling(window=14).mean()
        data['EMA'] = data['Close'].ewm(span=14, adjust=False).mean()
        data['RSI'] = talib.RSI(data['Close'], timeperiod=14)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        data['ATR'] = talib.ATR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['Stoch_K'], data['Stoch_D'] = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=14, slowk_period=3, slowd_period=3)
        data['OBV'] = talib.OBV(data['Close'], data['Volume'])
        data['CCI'] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['ROC'] = talib.ROC(data['Close'], timeperiod=10)
        data['MFI'] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=14)
        data['Chaikin'] = talib.ADOSC(data['High'], data['Low'], data['Close'], data['Volume'], fastperiod=3, slowperiod=10)
        data['WILLR'] = talib.WILLR(data['High'], data['Low'], data['Close'], timeperiod=14)
        data['SAR'] = talib.SAR(data['High'], data['Low'], acceleration=0.02, maximum=0.2)

        # Save the data with added indicators
        data.to_excel(output_file, index=False)

        print(f"Technical indicators added and saved to {output_file}")
        return data
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
