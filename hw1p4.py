import numpy as np
import matplotlib.pyplot as plt

def bisection(f, x, xx, trials, tol, actual_root):
    errors = []  # initialize list to store errors
    if np.sign(f(x)) == np.sign(f(xx)):
        return ["Zeroes not found"], errors
    for _ in range(trials):
        midPt = (x + xx) / 2
        error = np.abs(midPt - actual_root)
        errors.append(error)  # append current error
        if np.abs(f(midPt)) < tol or error < tol:
            return midPt, errors
        elif np.sign(f(x)) == np.sign(f(midPt)):
            xx = midPt
        else:
            x = midPt
    return midPt, errors

def secant(f, x, xx, trials, tol, actual_root):
    errors = []  # initialize list to store errors
    for _ in range(trials):
        xn = xx - f(xx) * ((xx - x) / (f(xx) - f(x)))
        error = np.abs(xn - actual_root)
        errors.append(error)  # append current error
        if np.abs(f(xn)) < tol or error < tol:
            return xn, errors
        x, xx = xx, xn
    return xn, errors

def newtonsMethod(f, df, x0, trials, tol, actual_root):
    x = x0
    errors = []  # initialize list to store errors
    for _ in range(trials):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return "error: divide by zero", errors
        x_new = x - fx / dfx
        error = np.abs(x_new - actual_root)
        errors.append(error)  # append current error
        if np.abs(fx) < tol or error < tol:
            return x, errors
        x = x_new
    return x, errors

def plot_errors(errors, method_name):
    plt.loglog(range(1, len(errors) + 1), errors, marker='o', label=method_name)
    plt.xlabel('Iteration')
    plt.ylabel('Absolute Error')
    plt.title('Error Convergence')
    plt.legend()
    plt.grid(True, which="both", ls="--")

def part1():
    f = lambda x: (x - 1) * (x - 2) * (x - 3)
    df = lambda x: 3 * x**2 - 12 * x + 11
    actualRoot= 1

    x0Newton= 0.1  
    x0_bisection = 0  
    xx_bisection = 1.5
    x0_secant = 0.1  
    xx_secant = 1.5

    _, errorsNewton = newtonsMethod(f, df, x0Newton, 1000, 1e-6, actualRoot)
    _, errorsBisection = bisection(f, x0_bisection, xx_bisection, 1000, 1e-6, actualRoot)
    _, errorsSecant = secant(f, x0_secant, xx_secant, 1000, 1e-6, actualRoot)

    plt.figure(figsize=(10, 6))
    plot_errors(errorsNewton, 'Newton\'s Method for Part 1')
    plot_errors(errorsBisection, 'Bisection Method for Part 1')
    plot_errors(errorsSecant, 'Secant Method for Part 1')
    plt.show()

def part2():
    f = lambda x: (x - 1)**3
    df = lambda x: 3 * (x - 1)**2
    actualRoot = 1

    x0Newton = 0.1
    x0_bisection = 0.5  
    xx_bisection = 1.5
    x0_secant = 0.1
    xx_secant = 1.5

    _, errorsNewton = newtonsMethod(f, df, x0Newton, 1000, 1e-6, actualRoot)
    _, errorsBisection = bisection(f, x0_bisection, xx_bisection, 1000, 1e-6, actualRoot)
    _, errorsSecant = secant(f, x0_secant, xx_secant, 1000, 1e-6, actualRoot)

    plt.figure(figsize=(10, 6))
    plot_errors(errorsNewton, 'Newton\'s Method for Part 2')
    plot_errors(errorsBisection, 'Bisection Method for Part 2')
    plot_errors(errorsSecant, 'Secant Method for Part 2')
    plt.show()

part1()
part2()
