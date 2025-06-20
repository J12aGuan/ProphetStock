from collections import Counter
import numpy as np
import math
import random

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value


class internalNode:
    def __init__(self, feature, threshold, leftBranch, righBranch):
        self.feature = feature
        self.threshold = threshold
        self.left = leftBranch
        self.right = righBranch

class buildTree:
    def __init__(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain

    def split(self, dataset, labels, columnIndex, threshold):
        # Store the subsets of data and labels after splitting
        left_x_subsets, left_y_subsets = [], []
        right_x_subsets, right_y_subsets = [], []

        # Get all unique values from the specified column
        # This avoids repeated splits on the same value
        counts = list(set([data[columnIndex] for data in dataset]))  # Loop the 2D array and store unique values from the target column
        counts.sort()  # Sort the unique values to keep output consistent

        for i in range(len(dataset)):
            if(dataset[i][columnIndex] < threshold):
                left_x_subsets.append(dataset[i])
                left_y_subsets.append(labels[i])
            else:
                right_x_subsets.append(dataset[i])
                right_y_subsets.append(labels[i])

        return (left_x_subsets, right_x_subsets), (left_y_subsets, right_y_subsets)

    def gini(self, labels):
        impurity = 1
        labelCounts = Counter(labels)
        
        for label in labelCounts:
            probOfLabel = labelCounts[label] / len(labels)
            impurity -= probOfLabel ** 2

        return impurity

    def informationGain(self, startingLabels, splitLabels):
        # parent_labels: all labels before split
        # split_labels: [left_labels, right_labels]
        infoGain = self.gini(startingLabels)
        
        for subset in splitLabels:
            infoGain -= self.gini(subset) * len(subset) / len(startingLabels)

        return infoGain

    def featureBagging(self, dataset, labels):
        bestGain = 0
        bestFeature = None
        bestThreshold = None

        numFeatures = len(dataset[0])
        k = round(math.sqrt(len(dataset[0]))) # typically sqrt for classification

        # At each split, randomly pick sqaure root number of total features to consider (rounded)
        # replace = False means no same feature will appear twice
        feature_indices = np.random.choice(numFeatures, k, replace = False)

        for feature_idx in feature_indices:
            # Get unique sorted values for this feature
            values = sorted(set([row[feature_idx] for row in dataset]))

            # Try midpoints between consecutive values as thresholds
            for i in range(1, len(values)):
                threshold = (values[i - 1] + values[i]) / 2
                (left_x_subsets, right_x_subsets), (left_y_subsets, right_y_subsets) = self.split(dataset, labels, feature_idx, threshold)
                if not left_y_subsets or not right_y_subsets:
                    continue

                gain = self.informationGain(labels, [left_y_subsets, right_y_subsets])
                if gain > bestGain:
                    bestGain = gain
                    bestFeature = feature_idx
                    bestThreshold = threshold
        
        return bestGain, bestFeature, bestThreshold
    
    def build(self, dataset=None, labels=None, value=None):
        # 1. If this is the first call (i.e., root of the tree), use the full training data
        if dataset is None and labels is None:
            dataset = self.xTrain
            labels = self.yTrain

        # Find the best feature to split on using feature bagging
        bestGain, bestFeature, bestThreshold = self.featureBagging(dataset, labels)

        # 2. Base case: if no meaningful gain from splitting, return a Leaf node
        #    Leaf stores the most common label and stops further recursion
        if bestGain < 1e-7 or bestThreshold is None:
            value = Counter(labels).most_common(1)[0][0]  # Majority label
            return Leaf(labels, value)

        # 3. Otherwise, split the dataset and labels based on bestFeature and bestThreshold
        (left_x_subsets, right_x_subsets), (left_y_subsets, right_y_subsets) = self.split(dataset, labels, bestFeature, bestThreshold)
        
        leftBranch = self.build(left_x_subsets, left_y_subsets)
        rightBranch = self.build(right_x_subsets, right_y_subsets)

        # 5. Return an internal node containing:
        #    - bestFeature used for splitting at this level
        #    - bestThreshold used for splitting the left and right branch
        #    - all child branches (Leaf or internalNode)
        return internalNode(bestFeature, bestThreshold, leftBranch, rightBranch)



        


                