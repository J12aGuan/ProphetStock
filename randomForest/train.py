from randomForest.forest import randomForest
from datetime import datetime
import pandas as pd
import numpy as np

currentYear = datetime.now().year
years = [(currentYear) - x for x in list(range(11))]

# Loop through each year and train the model
for year in years:
    filePath = f"stockPrediction/data/csv/AAPL/{year} AAPL.csv"
    df = pd.read_csv(filePath)

    # Remove the first two row
    df.dropna(inplace=True)

    # Prepare features and labels
    feature_cols = ['High', 'Low', 'Open', 'Volume', 'PrevClose', 'Range', 'MA3']
    X = df[feature_cols].values
    y = df['Close'].values

    splitRatio = 0.8
    splitIndex = int(len(X) * splitRatio)

    # Split X
    X_train = X[:splitIndex]        # 80%
    X_test = X[splitIndex:]         # 20%

    # Split y
    y_train = y[:splitIndex]        # 80%
    y_test = y[splitIndex:]         # 20%


    y_result = []
    randomForestObj = randomForest(num_trees=20).fit(X_train, y_train)
    for X_individual in X_test:
        y_pred = randomForestObj.predict(X_individual)
        y_result.append(y_pred)

    print(y_result)

# python -m randomForest.train