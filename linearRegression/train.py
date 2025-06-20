from linearRegression.model import linearRegression
from stockPrediction.graph import graph
from stockPrediction.graph.plotData import plotData
from datetime import datetime
import pandas as pd
import numpy as np

currentYear = datetime.now().year
years = [(currentYear) - x for x in list(range(11))]

# Loop through each year and train the model
for year in years:
    filePath = f"stockPrediction/data/csv/AAPL/{year} AAPL.csv"
    df = pd.read_csv(filePath)

    # Prepare features and labels
    X = np.arange(1, len(df) + 1)
    y = df['Close'].values

    # Instantiate and train model
    model = linearRegression(X, y, iterations=10000, alpha=0.0001)
    w, b, J_hist, p_hist = model.gradientDescent()

    # Plot result
    plotObj = plotData("AAPL", year)
    plotObj.getLinearRegressionPlotData(w, b)
    graph.storePlotData(plotObj)

graph.showGraph()

# python -m linearRegression.train

