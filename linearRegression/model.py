from stockPrediction.graph import graph
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

currentYear = datetime.now().year
years = [(currentYear) - x for x in list(range(11))]

def getTrainData(stock, year):
    filePath = f"stockPrediction/data/csv/{stock}/{year} {stock}.csv"
    df = pd.read_csv(filePath)
    xTrain = np.array(range(df.shape[0])) + 1   #Date
    xTrainCentered = xTrain - np.mean(xTrain)
    yTrain = df['Close'].values                 #Closing Price

    w_init = 0
    b_init = np.mean(yTrain)

    iterations = 10000
    alpha = 0.0001

    w_final, b_final, J_hist, p_hist = gradientDescent(
        xTrainCentered, yTrain, w_init, b_init, alpha, iterations,
        computeCost, computeGradient
    )

    # Convert centered model back to original scale
    x_mean = np.mean(np.array(range(df.shape[0])) + 1)  # mean of uncentered xTrain
    true_b = -w_final * x_mean + b_final

    # Print both for comparison
    print(f"Cost {J_hist[-1]} ",
        f"w (slope): {w_final}, b (intercept in centered space): {b_final}")
    print(f"Adjusted intercept (original space): {true_b}")

    # Plot using original slope and adjusted intercept
    graph.graphFigures(stock, year, w_final, true_b)

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

for year in years:
    getTrainData("AAPL", year)

graph.showGraph()

# python -m linearRegression.model
