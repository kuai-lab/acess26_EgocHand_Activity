# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score

# class CWClassifier:
#     def __init__(self):
#         self.model = LogisticRegression()

#     def train(self, X, y):
#         self.model.fit(X.reshape(-1, 1), y)

#     def predict(self, X):
#         return self.model.predict(X.reshape(-1, 1))

#     def evaluate(self, X, y_true):
#         y_pred = self.predict(X)
#         return accuracy_score(y_true, y_pred)
