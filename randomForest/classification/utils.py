import numpy as np
import operator
from randomForest.classification.model import Leaf

# Row Bagging
def rowBagging(X_train, y_train):
    num_data = len(X_train)
    indices = np.random.choice(num_data, num_data, replace=True)
    X_bagged = [X_train[index] for index in indices]
    y_bagged = [y_train[index] for index in indices]

    return X_bagged, y_bagged

def classify(datapoint, tree):
    if isinstance(tree, Leaf):
        return max(tree.labels.items(), key=operator.itemgetter(1))[0]
    
    if datapoint[tree.feature] < tree.threshold:
        return classify(datapoint, tree.left)
    else:
        return classify(datapoint, tree.right)