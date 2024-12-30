import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBRegressor
import pickle

def preprocess_backtest_data(data: pd.DataFrame, scaler, features):
    """
    Preprocess data for backtesting: include lag features and apply scaler.
    """
    # Forward fill missing values
    data = data.fillna(method='ffill')

    # Create lag features
    for col in features:
        data[f'{col}_Lag1'] = data[col].shift(1)

    # Drop rows with NaN values introduced by lagging
    data.dropna(inplace=True)

    # Extract the required features (including lag features)
    X = data[features + [f'{col}_Lag1' for col in features]]

    # Apply the scaler
    X_scaled = scaler.transform(X)

    return data, X_scaled

def backtest(ticker: str, initial_balance=10000):
    """
    Backtest a trading strategy using a trained model.

    Parameters:
    - ticker: Stock ticker for backtesting.
    - initial_balance: Starting capital for the backtest.

    Returns:
    - backtest_report: A dictionary with performance metrics.
    """
    try:
        # Load data
        d_path = os.path.join("src", "data", "processed", f"{ticker}_with_indicators.xlsx")
        if not os.path.exists(d_path):
            raise FileNotFoundError(f"Processed data file not found: {d_path}")
        data = pd.read_excel(d_path)

        # Load model and scaler
        model_path = os.path.join("src", "models", "artifacts", f"{ticker}_xgb_model.json")
        scaler_path = os.path.join("src", "models", "artifacts", f"{ticker}_scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        model = XGBRegressor()
        model.load_model(model_path)

        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        features = [
            "Close", "High", "Low", "Open", "Volume", "SMA", "EMA", 
            "RSI", "MACD", "MACD_signal", "MACD_hist", "BB_upper", "BB_middle", 
            "BB_lower", "ATR", "Stoch_K", "Stoch_D", "OBV", "CCI", "ROC", 
            "MFI", "Chaikin", "WILLR", "SAR"
        ]
        
        print(f"Backtesting strategy for {ticker}...")

        # Preprocess data for backtesting
        data, X_scaled = preprocess_backtest_data(data, scaler, features)

        # Predict price movements
        data['Prediction'] = model.predict(X_scaled)
        data['Prediction'] = np.where(data['Prediction'] > 0, 1, -1)  # Convert predictions to buy/sell signals

        # Generate target
        data['Target'] = np.sign(data['Close'].shift(-1) - data['Close'])  # 1 for up, -1 for down

        # Simulate trades
        balance = initial_balance
        positions = 0  # Current holdings
        equity_curve = [balance]

        for i in range(len(data) - 1):
            # Buy signal
            if data.loc[i, 'Prediction'] == 1 and positions == 0:
                positions = balance / data.loc[i, 'Close']  # Buy stock
                balance = 0

            # Sell signal
            elif data.loc[i, 'Prediction'] == -1 and positions > 0:
                balance = positions * data.loc[i, 'Close']  # Sell stock
                positions = 0

            # Track equity
            equity_curve.append(balance + (positions * data.loc[i + 1, 'Close'] if positions > 0 else 0))

        # Final balance
        final_balance = balance + (positions * data.iloc[-1]['Close'] if positions > 0 else 0)

        # Performance metrics
        roi = (final_balance - initial_balance) / initial_balance * 100
        accuracy = (data['Target'] == data['Prediction']).mean() * 100

        conf_matrix = confusion_matrix(data['Target'], data['Prediction'], labels=[-1, 1])
        class_report = classification_report(data['Target'], data['Prediction'], labels=[-1, 1])

        # Visualizations
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label="Equity Curve")
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()

        # Generate backtest report
        backtest_report = {
            "Initial Balance": initial_balance,
            "Final Balance": final_balance,
            "ROI (%)": roi,
            "Accuracy (%)": accuracy,
            "Confusion Matrix": conf_matrix.tolist(),
            "Classification Report": class_report,
        }

        print("\nConfusion Matrix:\n", conf_matrix)
        print("\nClassification Report:\n", class_report)
        print("\nBacktest Summary:")
        for k, v in backtest_report.items():
            if k != "Confusion Matrix" and k != "Classification Report":
                print(f"{k}: {v}")

        return backtest_report

    except Exception as e:
        print(f"An error occurred during backtesting: {e}")
        return None
