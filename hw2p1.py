#!!HW WORKED ON WITH CLASSMATES: JIO HONG & BEN CERRATO!!

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#functions and their gradients for testing
def f1(x):
    return x[0]**2 + x[1]**2

def df1(x):
    return np.array([2*x[0], 2*x[1]])

def f2(x):
    return 10**6 * x[0]**2 + x[1]**2

def df2(x):
    return np.array([2 * 10**6 * x[0], 2*x[1]])

def f3(x):
    return sum(xi**2 for xi in x)

def df3(x):
    return np.array([2*xi for xi in x])

#fefine hessian matrices for the functions
def d2f1(x):
    return np.array([[2, 0], [0, 2]])

def d2f2(x):
    return np.array([[2 * 10**6, 0], [0, 2]])

def d2f3(x):
    #assuming f3 is a function from R^n to R, its Hessian is 2I where I is the identity matrix
    return 2 * np.eye(len(x))

#gradient descent
def gradient_descent(f, df, x0, alpha, tolerance=1e-6, max_iterations=1000):
    #performs gradient descent on a given function f
    x = x0
    for i in range(max_iterations):
        grad = df(x)
        x_new = x - alpha * grad
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x

#implementing gradient descent with optimal step size for quadratic functions
def gradient_descent_optimal(f, df, A, b, x0, tolerance=1e-6, max_iterations=1000):
    #performs gradient descent with optimal step size for quadratic functions
    x = x0
    for i in range(max_iterations):
        grad = df(x)
        alpha = np.dot(grad, grad) / np.dot(np.dot(grad, A), grad) if np.dot(np.dot(grad, A), grad) != 0 else 1
        x_new = x - alpha * grad
        if np.linalg.norm(x_new - x) < tolerance:
            break
        x = x_new
    return x

if __name__ == "__main__":
    #test gradient descent
    print("Testing Gradient Descent with constant step size...")
    x0 = np.array([1.0, 1.0])
    print("Result for f1:", gradient_descent(f1, df1, x0, 0.01))
    print("Result for f2:", gradient_descent(f2, df2, x0, 0.000001))

    #test gradient descent with optimal step size
    print("Testing Gradient Descent with optimal step size for quadratic functions...")
    print("Result for f2:", gradient_descent_optimal(f2, df2, np.array([[10**6, 0], [0, 1]]), np.array([0, 0]), x0))