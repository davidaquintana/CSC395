# sir.py
# Eric A. Autry
# CSC 395 Spring 2024
# 02/01/24

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

"""
sirFunc: this function computes the RHS of the ODE for use in the solver.

y: the current solution vector
t: the current time
R0: the basic reproductive number, from the model

dydt: the RHS of the ODE
"""
def sirFunc(y, t, R0):
	ss, ii, rr = y
	dydt = [-R0*ss*ii, R0*ss*ii-ii, ii]
	return dydt


"""
sir: this function will solve the ODE using scipy's built in solver, and 
     returns the solution S,I,R at the time T given the initial vector and R0.

initVec: the initial values of S,I,R at time t=0
T: the final time to solve until
R0: the basic reproductive number, from the model

Returns the values S(T),I(T),R(T).
"""
def sir(initVec, T, R0):
	tt = [0, T]
	sol = odeint(sirFunc, initVec, tt, args=(R0,))
	return (sol[-1][0], sol[-1][1], sol[-1][2])


"""
sirForPlot: this solve the ODE over a time interval to give the solution for 
            use in plotting.

initVec: the initial values of S,I,R at time t=0
T: the final time to solve until
N: the number of intermediate points to use for plotting
R0: the basic reproductive number, from the model

sol: the matrix representing the solution
tt: the vector of timesteps used, corresponding to the matrix sol
"""
def sirForPlot(initVec, T, N, R0):
	tt = np.linspace(0, T, N)
	sol = odeint(sirFunc, initVec, tt, args=(R0,))
	return sol, tt

"""
plotSIR: will generate a plot of the SIR solution for the given inputs.

initVec: the initial values of S,I,R at time t=0
T: the final time to solve until
N: the number of intermediate points to use for plotting
R0: the basic reproductive number, from the model

Creates and displays a corresponding plot, no return value.
"""
def plotSIR(initVec, T, N, R0):
	sol, tt = sirForPlot(initVec, T, N, R0)
	plt.plot(tt, sol[:,0], 'b', label='S')
	plt.plot(tt, sol[:,1], 'r', label='I')
	plt.plot(tt, sol[:,2], 'g', label='R')
	plt.legend(loc='best')
	plt.xlabel('t')
	plt.grid()
	plt.show()
	return

def derv1(R0, sFin, iFin):
    return ((R0 * sFin * iFin) - iFin)

def getDi2dt2(r0, sFin, iFin):
    return (- (R0 ** 2) * sFin * (iFin ** 2)) + ((R0 * sFin - 1) ** 2) * iFin

def newtonWithVal(dfx, d2fx, x0, maxIter, tol, init, R0):
    #at the x value, finds the new SIR
    sFin, iFin, rFin = sir(init, x0 - (dfx/d2fx), R0)
    #only the iFin changes
    
    #base case
    if(maxIter == 0):
        print("max iterations hit")
        return x0, iFin
    elif d2fx == 0:
        return "divide by 0"
    elif (abs(x0 - (dfx/d2fx) - x0) < tol):
        print("tolerance hit")
        return x0, iFin
    
    #with the new SIR, find the first derivative
    newDfx = derv1(R0, sFin, iFin)

    #with the new SIR, find the second derivative
    newD2fx = getDi2dt2(R0, sFin, iFin)

    #recursively calls function with new x value
    return newtonWithVal(newDfx, newD2fx, x0 - (dfx/d2fx), maxIter - 1, tol, init, R0)


if __name__ == "__main__":

	# Example scenario with initial infection of 1% of population and R0=5.
	# Solving until time T=4, with 1000 intermediate steps for plotting.
	init = [.99, 0.01, 0]
	T = 4
	R0 = 5
	N = 1000

	# Get the values at time T=4.
	sFin, iFin, rFin = sir(init, T, R0)
	print("T=4, R0=5, I0=0.01")
	print(sFin, iFin, rFin)
	print()

	# Get the values at time T=1.5.
	sFin, iFin, rFin = sir(init, 1.5, R0)
	print("T=1.5, R0=5, I0=0.01")
	print(sFin, iFin, rFin)

	# Plot the solution through time T=4.
	plotSIR(init, T, N, R0)

#example test cases

#Example initial infection of 1% and R0 = 3. Solving until T = 3, with 1000 itermediate steps
T = 3
R0 = 3
sFin, iFin, rFin = sir(init, T, R0)
initPoint = T
maxIter = 900
tolerance = 0.0001

print("with a R0 of 3, T at infections, max infections is: ")
print(newtonWithVal(derv1(R0, sFin, iFin), getDi2dt2(R0, sFin, iFin), initPoint, maxIter, tolerance, init, R0))


#Example initial infection with 1% and R0 = 7. Solving until T = 7, with 1000 itermediate steps
T = 7
R0 = 7
sFin, iFin, rFin = sir(init, T, R0)
initPoint = T
maxIter = 900
tolerance = 0.0001

print("with a R0 of 7, T at infections, max infections is: ")
print(newtonWithVal(derv1(R0, sFin, iFin), getDi2dt2(R0, sFin, iFin), initPoint, maxIter, tolerance, init, R0))