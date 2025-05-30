# plot.py

import pandas as pd

class Plot:
    def __init__(self, stock, year, w, b):
        self.stock = stock
        self.year = year
        self.w = w
        self.b = b
        self.filePath = f"stockPrediction/data/csv/{self.stock}/{self.year} {self.stock}.csv"
        self.df = pd.read_csv(self.filePath)
        self.df['Date'] = pd.to_datetime(self.df['Date'])

    def getPlotData(self):
        # Returns actual and predicted prices
        y_pred = [self.w * i + self.b for i in range(1, len(self.df) + 1)]
        return self.df['Date'], self.df['Close'], y_pred
