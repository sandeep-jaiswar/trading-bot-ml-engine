import json
import os
import pandas as pd
from utils.db_utils import fetch_all_tickers
from data.raw.fetch_data import fetch_stock_data
from features.data_preprocessing.data_preprocessing import preprocess_data
from features.technical_indicators.technical_indicators import add_technical_indicators
from models.training.train_model import incremental_training
from backtesting.strategy_validation.backtest import backtest

if __name__ == "__main__":
    tickers = fetch_all_tickers()
    if not tickers:
        print("No tickers found in the database.")
    start_date = "2023-01-01"
    end_date = "2023-12-31"

    for ticker in tickers:
        print(f"Fetching and processing data for {ticker}...")
        raw_data = fetch_stock_data(ticker, start_date, end_date)
        processed_data = preprocess_data(ticker)
        data_with_indicators = add_technical_indicators(ticker)
        folder_path = os.path.join("src", "data", "processed")
        output_folder = os.path.join("src", "models", "artifacts")
        final_model, final_scaler = incremental_training(folder_path, output_folder)
        report = backtest(ticker)
        if report:
            print(json.dumps(report, indent=4))
