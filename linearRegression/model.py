# from stockPrediction.graph import plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def getTrainData(stock):
    filePath = f"stockPrediction/data/csv/{stock}.csv"
    df = pd.read_csv(filePath)
    xTrain = np.array(range(df.shape[0]))   #Date
    yTrain = df['Close'].values             #Closing Price

    print(type(xTrain))
    print(type(yTrain))
#We are going to use f_w,b(x^(i)) = wx^(i) + b to predict the data

getTrainData("AAPL")
# plot.loadData("AAPL")