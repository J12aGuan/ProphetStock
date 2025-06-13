from collections import Counter
import numpy as np
import math
import random

class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value


class internalNode:
    def __init__(self, feature, branches, value, labels=None):
        self.feature = feature
        self.branches = branches
        self.value = value
        self.labels = Counter(labels) if labels else None

class buildTree:
    def __init__(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain

    def split(self, dataset, labels, column):
        # Store the subsets of data and labels after splitting
        x_subsets = []
        y_subsets = []

        # Get all unique values from the specified column
        # This avoids repeated splits on the same value
        counts = list(set([data[column] for data in dataset]))  # Loop the 2D array and store unique values from the target column
        counts.sort()  # Sort the unique values to keep output consistent

        for value in counts:
            # Prepare new containers for this specific split
            new_x_subset = []
            new_y_subset = []

            # Loop through each row of the dataset
            for i in range(len(dataset)):
                if dataset[i][column] == value:
                    # Add the entire row to this feature subset
                    new_x_subset.append(dataset[i])
                    # Add the corresponding label to the label subset
                    new_y_subset.append(labels[i])

            # Add the resulting subsets to the master list
            x_subsets.append(new_x_subset)
            y_subsets.append(new_y_subset)

        # Return all the subsets of features and labels grouped by unique values in the split column
        return x_subsets, y_subsets

    def gini(self, labels):
        impurity = 1
        labelCounts = Counter(labels)

        for label in labelCounts:
            probOfLabel = labelCounts[label] / len(labels)
            impurity -= probOfLabel ** 2

        return impurity

    def informationGain(self, startingLabels, splitLabels):
        infoGain = self.gini(startingLabels)

        for subset in splitLabels:
            infoGain -= self.gini(subset) * len(subset) / len(startingLabels)

        return infoGain

    def featureBagging(self, dataset, labels):
        bestGain = 0
        bestFeature = 0

        numFeatures = len(dataset[0])
        k = round(math.sqrt(len(dataset[0]))) # typically sqrt for classification

        # At each split, randomly pick sqaure root number of total features to consider (rounded)
        # replace = False means no same feature will appear twice
        feature_indices = np.random.choice(numFeatures, k, replace = False)

        for feature_idx in feature_indices:
            x_subsets, y_subsets = self.split(dataset, labels, feature_idx)
            gain = self.informationGain(x_subsets, y_subsets)
            if gain > bestGain:
                bestGain, bestFeature = gain, feature_idx
        
        return bestGain, bestFeature
    
    def build(self, dataset=None, labels=None, value=None):
        # Find the best feature to split on using feature bagging
        bestGain, bestFeature = self.featureBagging(dataset, labels)

        # 1. If this is the first call (i.e., root of the tree), use the full training data
        if dataset is None and labels is None:
            dataset = self.xTrain
            labels = self.yTrain

        # 2. Base case: if no meaningful gain from splitting, return a Leaf node
        #    Leaf stores the most common label and stops further recursion
        if bestGain < 0.00000001:
            value = Counter(labels).most_common(1)[0][0]  # Majority label
            return Leaf(labels, value)

        # 3. Otherwise, split the dataset and labels based on bestFeature
        x_subsets, y_subsets = self.split(dataset, labels, bestFeature)
        branches = []

        # 4. Recursively build subtrees for each subset (one per unique feature value)
        for i in range(len(x_subsets)):
            # Recurse on each subset; store the feature value that led to this branch
            branch = self.build(x_subsets[i], y_subsets[i], x_subsets[i][0][bestFeature])
            branches.append(branch)

        # 5. Return an internal node containing:
        #    - bestFeature used for splitting at this level
        #    - all child branches (Leaf or internalNode)
        #    - value: the feature value from the parent node that led here
        return internalNode(bestFeature, branches, value, labels)



        


                