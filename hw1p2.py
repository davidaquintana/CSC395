import numpy as np
import math

def newtonsMethod(f, df, x0, trials, tol):
    x = x0
    for _ in range(trials):
        fx = f(x)
        if np.abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0:
            return "error: divide by zero"
        x_new = x - fx / dfx
        if np.abs(x - x_new) < tol:  # additional check for insignificant change
            return x
        x = x_new
    return "max limits reached"

def part1and2():
    f = lambda x: (x**3) - (3*x) + 1
    df = lambda x: 3*(x**2) - 3
    ans1 = newtonsMethod(f, df, -0.8, 1000, .001)
    print(ans1) # -> 1.5321383803144402
    ans2 = newtonsMethod(f, df, 0, 1000, .001)
    print(ans2) # -> 0.3472222222222222
    # the method find the root at (1.532, 0) as opposed to my initial guess of (.347, 0)
    # because the x0 = -0.8 the iterative process leads the method closer to 1.532
    # when x0 is anywhere between 0 and 0.5, the guess is closer to the root of .347, so newton's method picks that root

def part3():
    f = lambda x: (math.exp(x) / (1+ math.exp(x))) - 1/2
    df = lambda x: (math.exp(x) / (1 + math.exp(x))**2) * math.exp(x)
    ans1 = newtonsMethod(f, df, -3, 1000, .0001)
    print(ans1) # -> 0.00016060664231754007
    ans2 = newtonsMethod(f, df, -2, 1000, .0001)
    print(ans2) # -> 1.7864521578941743e-07
    # the root of this function is 0; however, based on our x0s we are not getting that answer
    # this has to do with the initial guesses of -3 and -2, they  are likely leading the 
    # iterations towards the convergence on the logistic function's behavior, not directly indicating the root at x = 0
    # which is a specific characteristic of the function rather than a numerical root to find through approximation.

def part4():
    f = lambda x: math.sin(x) - x
    df = lambda x: math.cos(x) -1
    ans1 = newtonsMethod(f, df, 0, 1000, .0001)
    print(ans1) 
    ans2 = newtonsMethod(f, df, .1, 1000, .0001)
    print(ans2)
    # i have two examples showing, one with an ideal x0 guess (x = 0) and another with a less ideal x0 (x = .1)
    # the root of this equation is 0, so having the x0 at 0 allows Newton's method to correctly guess the root
    # while .1 is not the guess, newton's method find the closest value given the tolerance provided
    # this is different than the cos example because there is only one root of 0, as opposed to the sin equation
    # having multiple roots and the x0 being more important to finding a certian root
part4()

# !CITATION: REPRESENT PIECEWISE WAS FOUND ONLINE!
def part5():
    f1 = lambda x: abs(x) ** (1/3)
    df1 = lambda x: (1/3)*abs(x)**(-2/3) if x != 0 else None
    f2 = lambda x: abs(x) ** (2/3)
    df2 = lambda x: (2/3)*abs(x)**(-1/3) if x != 0 else None
    f3 = lambda x: abs(x) ** (4/3)
    df3 = lambda x: (4/3)*abs(x)**(1/3) if x != 0 else None

    # adjusted derivative functions to handle x < 0 case correctly
    def adjusted_df1(x):
        if x > 0:
            return (1/3) * (x ** (-2/3))
        elif x < 0:
            return -(1/3) * ((-x) ** (-2/3))
        else:
            return None  # derivative does not exist at x = 0

    def adjusted_df2(x):
        if x > 0:
            return (2/3) * (x ** (-1/3))
        elif x < 0:
            return -(2/3) * ((-x) ** (-1/3))
        else:
            return None

    def adjusted_df3(x):
        if x > 0:
            return (4/3) * (x ** (1/3))
        elif x < 0:
            return -(4/3) * ((-x) ** (1/3))
        else:
            return None

    ans1 = newtonsMethod(f1, adjusted_df1, 0.1, 1000, .0001)
    print(ans1)
    ans2 = newtonsMethod(f2, adjusted_df2, 0.1, 1000, .0001)
    print(ans2)
    ans3 = newtonsMethod(f3, adjusted_df3, 0.1, 1000, .0001)
    print(ans3)

def main():
    part1and2()
    part3()
    part4()
    part5()

if __name__ == "__main__":
    main() 
