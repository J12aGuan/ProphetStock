from collections import Counter

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
        print(labelCounts)

        for label in labelCounts:
            probOfLabel = labelCounts[label] / len(labels)
            impurity -= probOfLabel ** 2

        return impurity
    
    def informationGain(self, startingLabels, splitLabels):
        infoGain = self.gini(startingLabels)

        for subset in splitLabels:
            infoGain -= self.gini(subset) * len(subset) / len(startingLabels)

        return infoGain
    
    
                