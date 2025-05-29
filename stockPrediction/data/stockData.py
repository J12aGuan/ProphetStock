import yfinance as yf
import pandas as pd

stockList = ["AAPL"]

def download_stock_data(ticker, start="2015-01-01", end="2025-01-01"):
    # Download historical stock data for a given ticker symbol
    df = yf.download(ticker, start=start, end=end)

    # Flatten MultiIndex columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Select only the 'Close' price
    df = df[['Close']].reset_index()

    # Add the ticker as a new column (safe and clean)
    df['Ticker'] = ticker

    # Save the DataFrame to a CSV file
    # Replace the old data with the new data if it already exist (index=False)
    df.to_csv(f"stockPrediction/data/csv/{stock}.csv", index=False)

    print(df)

for stock in stockList:
    download_stock_data(ticker=stock)