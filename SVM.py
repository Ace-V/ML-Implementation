import numpy as np

class SVM:
    def __init__(self,learning_rate=0.001,lambda_=0.01,n_iters=1000):
        self.lr=learning_rate
        self.lambda_param = lambda_
        self.n_iters = n_iters
        self.w=None
        self.b=None
    def fiting(self,X,y):
        n_smaples,n_features=X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i , xi in enumerate(X):
                cond=y_[i]*(np.dot(xi, self.w) - self.b) >= 1
                if cond:
                    self.w-=self.lr*(2*self.lambda_param*self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(xi, y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self,X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

        # Load dataset
    X, y = datasets.make_blobs(n_samples=100,n_features=2, centers=2,cluster_std=1.05, random_state=6)
    y = np.where(y == 0, -1, 1) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

        # Train SVM
    clf = SVM()
    clf.fiting(X_train, y_train)

        # Predict
    predictions = clf.predict(X_test)
    print("Predictions:", predictions)