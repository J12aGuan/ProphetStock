from randomForest.utils import rowBagging, classify
from randomForest.model import buildTree

class randomForest():
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

    def fit(self, dataset, labels):
        for i in range(self.num_trees):
            # Row bagging inside the loop — each tree gets its own random sample
            X_bagged, y_bagged = rowBagging(dataset, labels)

            buildTreeObj = buildTree(X_bagged, y_bagged)
            tree = buildTreeObj.build()
            self.trees.append(tree)

        return self
    
    def predict(self, X_individual):
        votes = []
        for tree in self.trees:
            vote = classify(X_individual, tree)
            votes.append(vote)
        y_pred = max(votes, key=votes.count)

        return y_pred