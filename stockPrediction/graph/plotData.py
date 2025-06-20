# plot.py

import pandas as pd

class plotData:
    def __init__(self, stock, year):
        self.stock = stock
        self.year = year
        self.filePath = f"stockPrediction/data/csv/{self.stock}/{self.year} {self.stock}.csv"
        self.df = pd.read_csv(self.filePath)
        self.dates = pd.to_datetime(self.df['Date'])
        self.actual = self.df['Close']
        self.predicted = []

    def getLinearRegressionPlotData(self, w, b):
        # Returns actual and predicted prices
        self.predicted = [w * i + b for i in range(1, len(self.df) + 1)]
    
    def getRandomForestPlotData(self, y_preds):
        self.predicted = y_preds