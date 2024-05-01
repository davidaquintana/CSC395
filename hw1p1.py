import numpy as np
import math

# bisection
def bisection(f, x, xx, trials, tol):
    # does not cross y axis
    if np.sign(f(x)) == np.sign(f(xx)):
        return("Zeroes not found")
    #mid point
    midPt = (x + xx)/2
    if np.abs(f(midPt)) < tol:
        return midPt
        return x
    # iteration
    elif trials == 0:
        return("ran out of trials to run")
    elif np.sign(f(x)) == np.sign(f(midPt)):
        return bisection(f, xx, midPt, trials -1, tol)
    elif np.sign(f(xx)) == np.sign(f(midPt)):
        return bisection(f, x, midPt, trials -1,tol)

# secant method    
def secant(f, df, x, xx, trials, tol):
    # equation given in class
    xn = xx - f(xx)/((f(xx) - (f(x))/(xx-x)))
    if np.abs(f(xn)) < tol:
        return xn
    # max iteration
    elif trials == 0:
        return("ran out of trials")
    else:
        return secant(f, df, xx, xn, trials - 1, tol)

#newton's method    
def newtonsMethod(f, df, x0, trials, tol):
    x = x0
    # max iteration
    for _ in range(trials):
        fx = f(x)
        if np.abs(fx) < tol:
            return x
        dfx = df(x)
        # divide by zero error
        if dfx == 0:
            return "error: divide by zero"
        x_new = x - fx / dfx
        if np.abs(x - x_new) < tol:  # additional check for insignificant change
            return x
        x = x_new
    return "max limits reached"

def main():
    
    print("Problem 1: ")
    f = lambda x: (x**3) - (3*x) + 1
    df = lambda x: 3*(x**2) - 3

    bisectionHw1a = bisection(f, 1, 2, 1000, .001)
    print(bisectionHw1a) # -> 1.5322265625
    bisectionHw1b = bisection(f, 0, 1, 1000, .001)
    print(bisectionHw1b) # -> 0.34765625
    secantHw1a = secant(f, df, 0, 1, 1000, .001)
    print(secantHw1a) # -> 0.347224243566683
    secantHw1b = secant(f, df, 1, 2, 1000, .001)
    print(secantHw1b) # -> 1.5321655588050014
    newtonsMethodHw1 = newtonsMethod(f, df, 0, 1000, .001)
    print(newtonsMethodHw1) # -> 0.3472222222222222

    print("Problem 2: ")
    rootTwo = lambda x: (x**2) - 2
    dRootTwo = lambda x: (2*x)
    rootThree = lambda x: (x**3) - 3
    dRootThree = lambda x: 2*(x**2)

    bisectionHw2a = bisection(rootTwo, 0, 1, 1000, .001)
    print(bisectionHw2a) # -> Zeroes not found; the zero is out of range of our guess, so the code did not find a zero
    bisectionHw2b = bisection(rootThree, 1, 2, 1000, .001)
    print(bisectionHw2b) # -> 1.4423828125; in this case i made the range appropriate to show that a zero is found
    secantHw2a = secant(rootTwo, dRootTwo, 0, 1, 1000, .001)
    print(secantHw2a) # -> 1.4142301368693015
    secantHw2b = secant(rootThree, dRootThree, 0, 1, 1000, .001)
    print(secantHw2b) # -> 1.4423997406881874
    newtonsMethodHw2a = newtonsMethod(rootTwo, dRootTwo, 1, 1000, .001)
    print(newtonsMethodHw2a) # -> 1.4142156862745099 newton's method does find the root because the initial guess is appropriate to do so
    newtonsMethodHw2b = newtonsMethod(rootThree, dRootThree, 0, 1000, .001)
    print(newtonsMethodHw2b) # -> error: divide by zero; newton's method does not find the root because the initial guess is not appropriate to do so

    print("Problem 3: ")
    f2 = lambda x: x - math.cos(x)
    df2 = lambda x: 1 + math.sin(x)

    bisectionHw3 = bisection(f2, 0, 1, 1000, .001)
    print(bisectionHw3) # -> 0.7392578125
    secantHw3 = secant(f2, df2, 0, 1, 1000, .001)
    print(secantHw3) # -> 0.7395660306611712
    newtonsMethodHw3 = newtonsMethod(f2, df2, 0, 1000, .001)
    print(newtonsMethodHw3) # -> 0.7391128909113617

if __name__ == "__main__":
    main()