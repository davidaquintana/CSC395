
# hw6p3.py
# CSC 395 Spring 2024
# Name:
# Date:

import math
import numpy as np
import matplotlib.pyplot as plt

def test_quad(x):
    return (x-2)*(x-3)

def lagrange(x, x_vals, y_vals):
    n = len(x_vals)
    total = 0
    for i in range(n):
        term = y_vals[i]
        for j in range(n):
            if i != j:
                term *= (x - x_vals[j]) / (x_vals[i] - x_vals[j])
        total += term
    return total

def chebyshev_nodes(a, b, n):
    return [0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) / (2 * n) * np.pi) for k in range(n)]

if __name__ == "__main__":

    # Set the domain and number of points.
    a = 0
    b = 5
    n = 75

    # Set the number of points for plotting, the plotting x values,
    # and the plot window.
    n_plot = 1000
    x_plot = np.linspace(a, b, n_plot)
    window = [a, b, -4, 10]  # specific for test_quad.
    
    # Create linearly spaced x values and sample the test function.
    x_lin = np.linspace(a, b, n)
    y_lin = np.array([test_quad(x) for x in x_lin])

    # Obtain the y values from the Lagrange Polynomial Interpolation.
    l_plot = [lagrange(x, x_lin, y_lin) for x in x_plot]

    # Plot the original data and the interpolated polynomial.
    plt.scatter(x_lin, y_lin)
    plt.plot(x_plot, l_plot)
    plt.axis(window)
    plt.show()

    # Now define the Chebyshev nodes.
    x_cheb = chebyshev_nodes(a, b, n)
    y_cheb = np.array([test_quad(x) for x in x_cheb])

    # Obtain the y values from the Lagrange Polynomial Interpolation.
    l_cheb = [lagrange(x, x_cheb, y_cheb) for x in x_plot]

    # Plot the original data and the interpolated polynomial.
    plt.scatter(x_cheb, y_cheb)
    plt.plot(x_plot, l_cheb)
    plt.axis(window)
    plt.show()
