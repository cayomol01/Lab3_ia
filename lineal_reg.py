import numpy as np


def Cost(theta, X, y):
    h = X @ theta
    return np.sum((h - y) ** 2) / (len(X))


def Gradient(theta, X, y, ):
    h = X @ theta
    #print(f"((X.T @ (h - y)) / len(X)).shape: {((X.T @ (h - y)) / len(X)).shape}")
    return 2*(X.T@ (h - y))/ len(X)

def Descent(X, y, theta_0, lr=0.0000001, th = 0.0000001,iter=10000):
    theta = theta_0
    costs = []
    thetas = []

    iterations = 0

    for i in range(iter):
        #print(f"theta: {theta.shape}")
        theta -= lr * Gradient(theta, X, y)
        costs.append(Cost(theta, X, y))
        thetas.append(theta.copy())

    return theta, costs, thetas
