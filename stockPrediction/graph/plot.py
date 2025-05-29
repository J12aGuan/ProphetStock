import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# Load the data
def loadData(stock):
    filePath = f"stockPrediction/data/csv/{stock}.csv"

    if not os.path.exists(filePath):
        raise FileNotFoundError(f"{filePath} not found.")
    
    df = pd.read_csv(filePath)

    # Convert the 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    plot(stock, df)

def plot(stock, df):
    # Plot
    plt.figure(figsize=(10, 5))  # width, height in inches

    # Plot the closing price over time
    # 'label' here is for the legend (not axis labels)
    plt.plot(df['Date'], df['Close'], label=f'{stock} Closing Price', linewidth=1.5)

    # Set labels for the x and y axes
    plt.xlabel("Date")              # X-axis label
    plt.ylabel("Close Price (USD)") # Y-axis label

    # Add grid lines to make the chart easier to read
    plt.grid(True)

    # Display the legend (uses the label from plt.plot)
    plt.legend()

    # Limit the number of y-axis ticks to 10
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(10))

    # Auto-adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Show the plot window
    plt.show()
    