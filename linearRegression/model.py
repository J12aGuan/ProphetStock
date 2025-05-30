import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stockPrediction.graph import plot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def getTrainData(stock, year):
    filePath = f"stockPrediction/data/csv/{stock}/{year} {stock}.csv"
    df = pd.read_csv(filePath)
    xTrain = np.array(range(df.shape[0])) + 1   #Date
    yTrain = df['Close'].values                 #Closing Price

    w_init = 0
    b_init = 0

    iterations = 10000
    temp_alpha = 0.001

    w_final, b_final, J_hist, p_hist = gradientDescent(xTrain, yTrain, w_init, b_init, temp_alpha, iterations,
computeCost, computeGradient)

# We are going to use f_wb(x^(i)) = wx^(i) + b to predict the data
def computeModelOutput(xTrain, w, b):   # w is the slope, b is the y-intercept, xTrain is the days
    dataAmount = xTrain.shape[0]        # shape[0] gets the length of the array
    f_wb = np.zeros(dataAmount)         # Creates an array, initalized with all entries in the array with 0. Entry length is the (dataAmount)
    for i in range(dataAmount):
        f_wb[i] = w * xTrain[i] + b     # Sets each entry in the array with the prediction --> w * xTrain[i] + b

    return f_wb

def computeCost(xTrain, yTrain, w, b):
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
        dj_db_i = (f_wb - yTrain[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw = dj_dw / dataAmount  #dj_dw is the change in total cost due to change in slope
    dj_db = dj_db / dataAmount  #dj_db is the change in total cost due to the change in the y-intercept
    
    return dj_dw, dj_db

def gradientDescent(xTrain, yTrain, w_in, b_in, alpha, numberIterations, computeCost, computeGradient):
    J_history = []  #Total cost history
    p_history = []  #History of parameters [w, b]
    b = b_in
    w = w_in
    
    for i in range(numberIterations):
        dj_dw, dj_db = computeGradient(xTrain, yTrain, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        
        #Save total cost J at each iteration
        if i < 100000:
            J_history.append(computeCost(xTrain, yTrain, w, b))     #History of total cost
            p_history.append([w, b])                                #History of parameters [w, b]

        #Print the result every ten time we interate
        if i % (numberIterations / 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]} ",
                f"dj_dw: {dj_dw}, dj_db: {dj_db} ",
                f"w: {w}, b: {b}")
            
    return w, b, J_history, p_history

#So if the derivative is negative, it means the prediction line (f_wb) is  below our points, while if the derivative is positive, it means the prediction line (f_wb) is above our points.
getTrainData("AAPL", 2017)


# plot.loadData("AAPL", 2015)
# plot.loadData("AAPL", 2016)
# plot.loadData("AAPL", 2017)
# plot.loadData("AAPL", 2018)
# plot.loadData("AAPL", 2019)
# plot.loadData("AAPL", 2020)
# plot.loadData("AAPL", 2021)
# plot.loadData("AAPL", 2022)