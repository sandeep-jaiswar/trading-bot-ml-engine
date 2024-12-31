import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the data: handle missing values, feature scaling, and feature engineering.
    """
    # Forward fill and backward fill missing values (fixing deprecated usage)
    data = data.ffill().bfill()

    # Features list
    features = [
        "Close", "High", "Low", "Open", "Volume", "SMA", "EMA", 
        "RSI", "MACD", "MACD_signal", "MACD_hist", "BB_upper", "BB_middle", 
        "BB_lower", "ATR", "Stoch_K", "Stoch_D", "OBV", "CCI", "ROC", 
        "MFI", "Chaikin", "WILLR", "SAR"
    ]
    target = 'Close'

    # Create lag features (shift by 1 day)
    for col in features:
        if col in data.columns:
            data[f'{col}_Lag1'] = data[col].shift(1)

    data.dropna(inplace=True)  # Drop NaN values after creating lag features

    X = data[features + [f'{col}_Lag1' for col in features]]  # Include lag features
    y = data[target]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def incremental_training(folder_path: str, output_folder: str, initial_model: XGBRegressor = None):
    """
    Incrementally train a model on multiple XLSX datasets.
    """
    try:
        # Initialize model if not provided
        if initial_model is None:
            model = XGBRegressor(random_state=42, warm_start=True, n_estimators=100)  # Setting initial n_estimators
        else:
            model = initial_model

        scaler = None

        # Iterate through each file
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".xlsx"):
                file_path = os.path.join(folder_path, file_name)

                # Load dataset
                data = pd.read_excel(file_path)
                logger.info(f"Training on file: {file_name}")

                # Preprocess the data
                X, y, file_scaler = preprocess_data(data)

                # Split data into train and test sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Hyperparameter tuning (optional)
                # You can perform hyperparameter tuning here or use the same settings
                # For simplicity, we continue training without any change in parameters

                # Train the model on the current dataset with early stopping
                model.fit(
                    X_train, 
                    y_train, 
                    eval_set=[(X_test, y_test)], 
                    early_stopping_rounds=10, 
                    verbose=True,
                    eval_metric='rmse'  # Directly passing eval_metric here
                )

                # Evaluate the model
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                logger.info(f"File: {file_name} Evaluation:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}")

                # Update the scaler
                scaler = file_scaler

        # Save the final model and scaler
        model_path = os.path.join(output_folder, "incremental_xgb_model.json")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        logger.info(f"Final model saved to: {model_path}")

        scaler_path = os.path.join(output_folder, "incremental_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        return model, scaler
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return None, None
