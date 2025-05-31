from stockPrediction.graph import graph
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime

currentYear = datetime.now().year
years = [(currentYear) - x for x in list(range(11))]

class linearRegression:
    def __init__(self, stock, year, iterations, alpha):
        self.stock = stock
        self.year = year
        self.filePath = f"stockPrediction/data/csv/{self.stock}/{self.year} {self.stock}.csv"
        self.df = pd.read_csv(self.filePath)
        self.xTrain = np.array(range(self.df.shape[0])) + 1     #Date
        self.xTrainCentered = self.xTrain - np.mean(self.xTrain)
        self.yTrain = self.df['Close'].values                   #Closing Price
        self.w = 0
        self.b = np.mean(self.yTrain)
        self.iterations = iterations
        self.alpha = alpha
        self.J_history = []     #Total cost history
        self.p_history = []     #History of parameters [w, b]

    def computeCost(self):
        dataAmount = self.xTrainCentered.shape[0]
    
        costSum = 0
        for i in range(dataAmount):
            f_wb = self.w * self.xTrainCentered[i] + self.b
            cost = (f_wb - self.yTrain[i])**2    #It is sum of squares due to Error (SSE)
            costSum = costSum + cost
        totalCost = (1/(2 * dataAmount) * costSum)
        
        return totalCost # This is J(w,b)
    
    def computeGradient(self):
        dataAmount = self.xTrainCentered.shape[0]

        dj_dw = 0
        dj_db = 0

        for i in range(dataAmount):
            f_wb = self.w * self.xTrainCentered[i] + self.b
            dj_dw_i = (f_wb - self.yTrain[i]) * self.xTrainCentered[i]
            dj_db_i = (f_wb - self.yTrain[i])
            dj_dw += dj_dw_i
            dj_db += dj_db_i

        dj_dw = dj_dw / dataAmount  #dj_dw is the change in total cost due to change in slope
        dj_db = dj_db / dataAmount  #dj_db is the change in total cost due to the change in the y-intercept
        
        return dj_dw, dj_db
    
    def gradientDescent(self):        
        for i in range(self.iterations):
            dj_dw, dj_db = self.computeGradient()
            self.w = self.w - self.alpha * dj_dw
            self.b = self.b - self.alpha * dj_db

            
            #Save total cost J at each iteration
            if i < 100000:
                self.J_history.append(self.computeCost())     # History of total cost
                self.p_history.append([self.w, self.b])       # History of parameters [w, b]

            #Print the result every ten time we interate
            if i % (self.iterations / 10) == 0:
                print(f"Iteration {i}: Cost {self.J_history[-1]} ",
                    f"dj_dw: {dj_dw}, dj_db: {dj_db} ",
                    f"w: {self.w}, b: {self.b}")
                
        x_mean = np.mean(np.array(range(self.df.shape[0])) + 1)  # mean of uncentered xTrain
        self.b = -self.w * x_mean + self.b     # Adjusted intercept
                
        return self.w, self.b, self.J_history, self.p_history

for year in years:
    linearRegressionObj = linearRegression("AAPL", year, 10000, 0.0001)
    w_final, b_final, J_hist, p_hist = linearRegressionObj.gradientDescent()

    # Plot using slope and adjusted intercept
    graph.storePlotData(linearRegressionObj.stock, linearRegressionObj.year, w_final, b_final)

graph.showGraph()

# python -m linearRegression.model