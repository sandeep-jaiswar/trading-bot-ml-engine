import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBRegressor
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def preprocess_backtest_data(data: pd.DataFrame, scaler, features: list):
    """
    Preprocess data for backtesting: fill missing values, create lag features, and scale the data.
    """
    try:
        # Fill missing values
        data.ffill(inplace=True)

        data.dropna(inplace=True)

        # Create lag features
        lag_features = []
        for col in features:
            lag_col = f"{col}_Lag1"
            data[lag_col] = data[col].shift(1)
            lag_features.append(lag_col)

        # Drop rows with NaN values introduced by lagging
        data.dropna(inplace=True)

        # Prepare feature matrix
        X = data[features + lag_features]

        # Scale features
        X_scaled = scaler.transform(X)

        return data, X_scaled
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}")
        raise


def simulate_trades(data: pd.DataFrame, initial_balance: float):
    """
    Simulate trading strategy based on predictions.
    """
    try:
        balance = initial_balance
        positions = 0  # Current holdings
        equity_curve = [balance]

        for i in range(len(data) - 1):
            prediction = data.iloc[i]['Prediction']
            close_price = data.iloc[i]['Close']

            # Buy signal
            if prediction == 1 and positions == 0:
                positions = balance / close_price
                balance = 0

            # Sell signal
            elif prediction == -1 and positions > 0:
                balance = positions * close_price
                positions = 0

            # Track equity
            next_close = data.iloc[i + 1]['Close']
            equity = balance + (positions * next_close if positions > 0 else 0)
            equity_curve.append(equity)

        # Final balance
        final_balance = balance + (positions * data.iloc[-1]['Close'] if positions > 0 else 0)

        return final_balance, equity_curve
    except Exception as e:
        logging.error(f"Error during trade simulation: {e}")
        raise


def backtest(ticker: str, initial_balance=10000):
    """
    Backtest a trading strategy using a trained model.
    """
    try:
        # File paths
        data_path = os.path.join("src", "data", "processed", f"{ticker}_with_indicators.xlsx")
        model_path = os.path.join("src", "models", "artifacts", "incremental_xgb_model.json")
        scaler_path = os.path.join("src", "models", "artifacts", "incremental_scaler.pkl")

        # Load data
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Processed data file not found: {data_path}")
        data = pd.read_excel(data_path)

        # Load model and scaler
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

        # Preprocess data
        data, X_scaled = preprocess_backtest_data(data, scaler, features)

        # Predict price movements
        data['Prediction'] = model.predict(X_scaled)
        data['Prediction'] = np.where(data['Prediction'] > 0, 1, -1)

        # Generate target
        data['Target'] = np.sign(data['Close'].shift(-1) - data['Close'])

        # Simulate trades
        final_balance, equity_curve = simulate_trades(data, initial_balance)

        # Performance metrics
        roi = (final_balance - initial_balance) / initial_balance * 100
        accuracy = (data['Target'] == data['Prediction']).mean() * 100

        # Drop rows with NaN in Target or Prediction
        data.dropna(subset=['Target', 'Prediction'], inplace=True)

        # Calculate metrics
        conf_matrix = confusion_matrix(data['Target'], data['Prediction'], labels=[-1, 1])
        class_report = classification_report(data['Target'], data['Prediction'], labels=[-1, 1], zero_division=1)

        # Save results to Excel
        results_path = os.path.join("src", "backtesting", f"results/{ticker}_backtest_results.xlsx")
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        data.to_excel(results_path, index=False)
        logging.info(f"Backtest results saved to {results_path}")

        # Visualize equity curve
        plt.figure(figsize=(10, 6))
        plt.plot(equity_curve, label="Equity Curve")
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()

        # Generate report
        backtest_report = {
            "Initial Balance": initial_balance,
            "Final Balance": final_balance,
            "ROI (%)": roi,
            "Accuracy (%)": accuracy,
            "Confusion Matrix": conf_matrix.tolist(),
            "Classification Report": class_report,
        }

        logging.info(f"Backtest completed for {ticker}.")
        return backtest_report

    except Exception as e:
        logging.error(f"An error occurred during backtesting: {e}")
        return None
