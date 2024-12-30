import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np

def preprocess_data(data: pd.DataFrame):
    """
    Preprocess the data: handle missing values, feature scaling, and feature engineering.
    """
    # Forward fill missing values
    data = data.fillna(method='ffill')

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
        data[f'{col}_Lag1'] = data[col].shift(1)

    data.dropna(inplace=True)  # Drop NaN values after creating lag features

    X = data[features + [f'{col}_Lag1' for col in features]]  # Include lag features
    y = data[target]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(data: pd.DataFrame, ticker: str):
    """
    Train a model using XGBoost on the provided dataset.
    """
    # Preprocess data
    X, y, scaler = preprocess_data(data)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model and hyperparameter tuning using GridSearchCV
    model = XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Model Evaluation:\nMSE: {mse:.4f}\nMAE: {mae:.4f}\nR2 Score: {r2:.4f}")

    # Save the model and scaler
    save_path = os.path.join("src", "models", "artifacts", f"{ticker}_xgb_model.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    best_model.save_model(save_path)
    print(f"Model saved to: {save_path}")
    
    scaler_path = os.path.join("src", "models", "artifacts", f"{ticker}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    return best_model, scaler
