#!!HW WORKED ON WITH CLASSMATES: JIO HONG & BEN CERRATO!!

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#define the functions and their gradients for testing
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

#define hessian matrices for the functions
def d2f1(x):
    return np.array([[2, 0], [0, 2]])

def d2f2(x):
    return np.array([[2 * 10**6, 0], [0, 2]])

def d2f3(x):
    #assuming f3 is a function from R^n to R, its hessian is 2I where I is the identity matrix
    return 2 * np.eye(len(x))

#newton's method
def newtons_method(f, df, d2f, x0, tolerance=1e-6, max_iterations=1000):
    #performs Newton's method on a given function f
    x = x0
    for i in range(max_iterations):
        grad = df(x)
        Hessian = d2f(x)
        delta_x = np.linalg.solve(Hessian, -grad)
        x_new = x + delta_x
        if np.linalg.norm(delta_x) < tolerance:
            break
        x = x_new
    return x

if __name__ == "__main__":
    #test newton's method
    print("Testing Newton's Method...")
    x0 = np.array([1.0, 1.0])
    print("Result for f1 using Newton's Method:", newtons_method(f1, df1, d2f1, x0))
    print("Result for f2 using Newton's Method:", newtons_method(f2, df2, d2f2, x0))
    print("Result for f3 using Newton's Method:", newtons_method(f3, df3, d2f3, np.array([1.0, 1.0, 1.0])))