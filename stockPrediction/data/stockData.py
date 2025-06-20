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

    # Add Previous Day's Close
    df['PrevClose'] = df['Close'].shift(1)

    # Add Daily Price Range
    df['Range'] = df['High'] - df['Low']

    # Add 3-day moving average
    df['MA3'] = df['Close'].rolling(window=3).mean()

    # Add tomorrow’s price
    df["Target"] = df["Close"].shift(-1)

    # Add the ticker as a new column
    df['Ticker'] = ticker

    # Add Date
    df.reset_index(inplace=True)

    # Build the folder path
    folderPath = f"csv/{ticker}"
    # Create the folder if it doesn't exist
    os.makedirs(folderPath, exist_ok=True)

    # Save the DataFrame to a CSV file
    # Replace the old data with the new data if it already exist (index=False)
    df.to_csv(f"{folderPath}/{year} {ticker}.csv", index=False)

for stock in stockList:
    for year in years:
        download_stock_data(stock, year)