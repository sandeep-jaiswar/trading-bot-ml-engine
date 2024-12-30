import os
import yfinance as yf
import logging
from openpyxl import Workbook

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def fetch_stock_data(ticker: str, start: str, end: str):
    """
    Fetch stock data from Yahoo Finance for a given ticker and save it to an Excel file.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'INFY.NS').
        start (str): The start date in 'YYYY-MM-DD' format.
        end (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        None
    """
    try:
        # Define the file path
        input_file = os.path.join("src", "data", "raw", f"{ticker}_data.xlsx")
        
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(input_file), exist_ok=True)
        
        # Download stock data from Yahoo Finance
        logging.info(f"Fetching data for {ticker} from {start} to {end}...")
        stock_data = yf.download(ticker, start=start, end=end)
        
        if stock_data.empty:
            logging.warning(f"No data found for {ticker} between {start} and {end}.")
            return None
        
        # Reset the index to make 'Date' a column
        stock_data.reset_index(inplace=True)
        
        # Flatten column names (remove multi-level structure)
        stock_data.columns = [col if isinstance(col, str) else col[0] for col in stock_data.columns]

        # Initialize a new workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Stock Data"

        # Write the header row
        headers = list(stock_data.columns)
        ws.append(headers)

        # Write the data rows
        for _, row in stock_data.iterrows():
            ws.append(row.tolist())

        # Remove the first column (e.g., 'Price') if needed
        # ws.delete_cols(1)  # Adjust column index as necessary

        # Remove the second row (Ticker row, if present)
        if ws.max_row > 1:
            ws.delete_rows(2)

        # Save the Excel file
        wb.save(input_file)

        logging.info(f"Data saved successfully to {input_file}.")

        return stock_data

    except Exception as e:
        logging.error(f"Error fetching or saving data for {ticker}: {e}")
        return None
