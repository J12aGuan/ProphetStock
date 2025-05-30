import yfinance as yf
import pandas as pd
import os
from datetime import datetime

currentYear = datetime.now().year
years = [(currentYear) - x for x in list(range(11))]
stockList = ["AAPL"]

def download_stock_data(ticker, year):
    # Download historical stock data for a given ticker symbol
    df = yf.download(ticker, start=f"{year}-01-01", end=f"{year + 1}-01-01")

    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Select only the 'Close' price
    df = df[['Close']].reset_index()

    # Add the ticker as a new column (safe and clean)
    df['Ticker'] = ticker

    # Build the folder path
    folderPath = f"stockPrediction/data/csv/{stock}"
    # Create the folder if it doesn't exist
    os.makedirs(folderPath, exist_ok=True)

    # Save the DataFrame to a CSV file
    # Replace the old data with the new data if it already exist (index=False)
    df.to_csv(f"{folderPath}/{year} {stock}.csv", index=False)

for stock in stockList:
    for year in years:
        download_stock_data(stock, year)