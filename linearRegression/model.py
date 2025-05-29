# from stockPrediction.graph import plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def getTrainData(stock):
    filePath = f"stockPrediction/data/csv/{stock}.csv"
    df = pd.read_csv(filePath)
    xTrain = np.array(range(df.shape[0])) + 1   #Date
    yTrain = df['Close'].values                 #Closing Price

    temp_f_wb = computeModelOutput(xTrain, 200, 200)
    compute_cost(xTrain, yTrain, 15, 15)

# We are going to use f_wb(x^(i)) = wx^(i) + b to predict the data
def computeModelOutput(xTrain, w, b):   # w is the slope, b is the y-intercept, xTrain is the days
    dataAmount = xTrain.shape[0]        # shape[0] gets the length of the array
    f_wb = np.zeros(dataAmount)         # Creates an array, initalized with all entries in the array with 0. Entry length is the (dataAmount)
    for i in range(dataAmount):
        f_wb[i] = w * xTrain[i] + b     # Sets each entry in the array with the prediction --> w * xTrain[i] + b

    return f_wb

def compute_cost(xTrain, yTrain, w, b):
    dataAmount = xTrain.shape[0]
    
    costSum = 0
    for i in range(dataAmount):
        f_wb = w * xTrain[i] + b
        cost = (f_wb - yTrain[i])**2    #It is sum of squares due to Error (SSE)
        costSum = costSum + cost
    totalCost = (1/(2 * dataAmount) * costSum) # It's just a formula :( idk how to explain it
    
    return totalCost # This is J(w,b)

def computeGradient(xTrain, yTrain, w, b):
    dataAmount = xTrain.shape[0]

    dj_dw = 0
    dj_db = 0

    for i in range(dataAmount):
        f_wb = w * xTrain[i] + b
        dj_dw_i = (f_wb - yTrain[i]) * xTrain[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

def gradient_descent():
    x = None #PlaceHolder

#So if the derivative is negative, it means the prediction line (f_wb) is  below our points, while if the derivative is positive, it means the prediction line (f_wb) is above our points.
getTrainData("AAPL")
# plot.loadData("AAPL")