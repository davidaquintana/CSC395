
# hw6p2.py
# CSC 395 Spring 2024
# Name:
# Date:

import numpy as np
import random
import matplotlib.pyplot as plt

def noisy_data(func, a, b, N, sig):
    # Define random x values in our domain, then sort.
    x_vals = [a + (b - a) * random.random() for ii in range(N)]
    x_vals.sort()

    # Obtain a small additive noise for each data point.
    noise = [np.random.normal(0, sig) for ii in range(N)]

    # Set y = func(x) + noise, and return.
    y_vals = [func(x_vals[ii]) + noise[ii] for ii in range(N)]
    return x_vals, y_vals

def test_quad(x):
    return (x - 2) * (x - 3)

def poly_fit(x_vals, y_vals, n):
    # Create the Vandermonde matrix for polynomial fitting
    J = np.vander(x_vals, N=n+1, increasing=True)
    
    # Convert y_vals to a numpy array
    y = np.array(y_vals)
    
    # Solve the normal equations using numpy's linear algebra solver
    JTJ = J.T.dot(J)
    JTy = J.T.dot(y)
    xi = np.linalg.solve(JTJ, JTy)
    
    return xi.tolist()

if __name__ == "__main__":
    # Set the domain [a, b] and number of points n.
    a = 0
    b = 5
    N = 100

    # Set the order of the polynomial to fit.
    n = 2

    # Set the number of points for plotting and 
    # the std dev of the noise.
    N_lin = 1000
    sig = 0.5

    # Create the noisy data and the linearly spaced x values for
    # plotting.
    (xx, yy) = noisy_data(test_quad, a, b, N, sig)
    x_lin = np.linspace(a, b, N_lin)

    # Call the poly_fit function to perform the fit.
    xi = poly_fit(xx, yy, n)
    print(xi)
    
    # Plot the results and save them.
    plt.scatter(xx, yy)
    ff = [sum([xi[ii] * pow(x, ii) for ii in range(n + 1)]) for x in x_lin]
    plt.plot(x_lin, ff)
    plt.show()
