import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def calculate_gradient(X, y, theta):
    m = y.size
    return X.T @ (sigmoid(X @ theta) - y) / m

def gradient_D(X, y, alpha=0.1, num=100, tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # BIAS
    theta = np.zeros(X_b.shape[1])
    
    for i in range(num):
        gradient = calculate_gradient(X_b, y, theta)
        theta -= alpha * gradient
        
        if np.linalg.norm(gradient) < tol:
            break  # early stopping
    
    return theta

def predict(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # BIAS
    return sigmoid(X_b @ theta)

def predict1(X, theta, threshold=0.5):
    return (predict(X, theta) >= threshold).astype(int)

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_s = scaler.fit_transform(X_train)
X_st = scaler.transform(X_test)
theta_ = gradient_D(X_s, y_train, alpha=0.1)

y_pred_train = predict1(X_s, theta_)  
y_pred_test = predict1(X_st, theta_)   

# Calculate accuracies
trainacc = accuracy_score(y_train, y_pred_train)
testacc = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {trainacc:.4f}")
print(f"Test Accuracy: {testacc:.4f}")