# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 12:58:55 2019

@author: jakob
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import urllib.request


def read(shortcode):
    """
    Function that reads in the data given by requesting the url. It requires
    internet to function, if not available use np.loadtxt and specify skiprows
    and max_rows manually.
    Returns array of fit data and unoscillated flux.
    """
    
    url = "https://www.hep.ph.ic.ac.uk/~ms2609/CompPhys/neutrino_data/"+str(shortcode)+".txt"
    # Request teh datafile from url
    urllib.request.urlretrieve(url, str(shortcode)+".txt")
    data = urllib.request.urlopen(url)
    count = 0
    pos = []
    # Read teh datafile to determine where the two datasets starts and stops
    for line in data:
        count += 1
        if line == b'\n':  # Finds where data starts and ends form empty line
            pos.append(count)
        
    fit_data = np.loadtxt(str(shortcode)+".txt", delimiter="\n", skiprows=pos[0],
                      max_rows=(pos[1]-pos[0]))
    u_flux = np.loadtxt(str(shortcode)+".txt", delimiter="\n", skiprows=pos[2])
    return fit_data, u_flux


def Prob_osc(E, theta, m_sq, L=295):
    return 1 - (np.sin(2*theta))**2 * (np.sin(1.267*m_sq*L/E))**2
    

def NLL(theta):
    """
    Function that finds the negative log-likelihood as funciton of a number theta, m_sq
    is kept constant for this function.
    """
    # Number of observed events given by data to fit
    k = fit_data
    # Predicted event rate foudn by combining unoscillated flux and prob
    lam = Prob_osc(E[1:], theta, m_sq)*u_flux
    sum = 0
    for i in range(len(lam)):
        # Stirling approx breaks down for ki = 0
        if k[i] == 0:
            sum += lam[i]
        else:
            sum += lam[i] - k[i] + k[i]*np.log(k[i]/lam[i])
    return sum


def data_plot(theta, m_sq):
    """
    Function that plots the results as binned histograms. The energy bins is
    defined outside the funciton.
    """
    # Create one entry for every bin and weight it from the data set
    fig, ax = plt.subplots()
    ax.hist(E[0:-1], bins=E, weights=fit_data, histtype='step',
            color='black', label='Detector data')
    ax.hist(E[1:], bins=E, weights=Prob_osc(E[1:], theta, m_sq)*u_flux,
            histtype='step', color='blue', label='Best fit Oscillated Prediction')
    ax.hist(E[1:], bins=E, weights=u_flux,
            histtype='step', color='red', label='Unoscillated Prediction')
    ax.set_xlabel("Neutrino Energy (GeV)")
    ax.set_ylabel("Events / (GeV)")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend()
    plt.show()
    
    
def parabola_minimiser(f, start, stop, validate=False):
    """
    Function that finds the minimum of a parabola in several iterations. The inital 
    range is given and then the next iteration use the preceeding points.
    Start, stop needs to be in a range where the minimum is and f has positive curvature.
    Returns the three points thatdefine parabola x0, x1, x2 and minimum x3. 
    """
    
    tol = 3e-8  # Tolerance, not smaller than sqrt(epsilon)
    max_it = 100  # Maximum number of iterations
    # Initially spread points out to cover whole range
    x0, x2 = start, stop
    x1 = (stop - start)/2
    
    x3i = 1000  # Initialisation to ensure it doesn't exit at first iteration
    count = 0
    all_x3 = []  # array of all minima
    if validate == True:
        # Makes a plot of all the minima that were foudn to check for convergence
        fig2, ax2 = plt.subplots()
        theta_arr = np.linspace(0, np.pi/2, 2000)
        NLL_theta_arr = np.zeros(len(theta_arr))
        for i in range(len(theta_arr)):
            NLL_theta_arr[i] = NLL(theta_arr[i])
        ax2.plot(theta_arr, NLL_theta_arr)
    
    while count < max_it:
        # Loop through parabolic minimiser until converged on rached max_it
        y0, y1, y2 = f(x0), f(x1), f(x2)
        # Find position of minimum from parabola
        x3 = 0.5*((x2*x2 - x1*x1)*y0 + (x0*x0 - x2*x2)*y1 + (x1*x1 - x0*x0)*y2) \
                /((x2 - x1)*y0 + (x0 - x2)*y1 + (x1 - x0)*y2)
        # Find difference in minimum points between iterations to determine convergence
        all_x3.append(x3)
        if abs(x3 - x3i) < tol:
            print("Minimum x = %.7f found in %.i iterations" %(x3, count))
            print("Minimum NLL value: %.7f" %f(x3))
            if validate == True:
                fig1, ax1 = plt.subplots()
                ax1.plot(np.arange(0, count +1, 1), all_x3)
            return x0, x1, x2, x3
        
        if validate == True:
            # Plots points on NLL curve to validate that global minimum is found
            ax2.plot(x3, f(x3), 'x')
        x3i, y3 = x3, f(x3)  # Housekeeping
        xpoints = [x0, x1, x2, x3]
        fpoints = [y0, y1, y2, y3]
        # Keep x for three smallest f(x)
        del xpoints[np.argmax(fpoints)]
        xpoints.sort()
        x0, x1, x2 = xpoints
        count += 1
    print("Minimum could not be found for %.i iterations" % max_it)
    

def secant_method(f, x0, x1, max_it, tol):
    """
    Numerical root solver based on the secant method. Finds roots for function 
    f, by using inital points x0 and x1 and iterate through until convergence.
    Reeturns first root
    """
    count = 0
    while count < max_it:
        x2 = x1 - f(x1)*(x1 - x0) / (f(x1) - f(x0))  # Perform the algorhitm
        if abs(x2 - x1) < tol:  # Convergence condition
            return x2
        x0 = x1  # Housekeeping
        x1 = x2
        count += 1
    raise RuntimeError("Root not found")


def NLL_shifterror(xmin):
    """
    Finds roots of function = NLL(theta) - (NLL(theta_min) + 0.5), using secant
    method. Initial guess x0, x1, set both above or below minimum, to converge
    to each root.
    """
    tol = 3e-8
    max_it = 100
    def NLLshift(x):  # Shift funciton to have roots at deltaNLL = 0.5
        return NLL(x) - (NLL(xmin) + 0.5)
    
    # Find root to the right of min
    theta_plus = -xmin + secant_method(NLLshift, (xmin+0.01), (xmin+0.07), max_it, tol)        
    # Find root to the left of min
    theta_minus = xmin - secant_method(NLLshift, (xmin-0.01), (xmin-0.07), max_it, tol)
            
    sigma = (theta_plus + theta_minus)/2
    print("Error from Secant method = %.7f (theta_plus = %.7f, theta_minus = %.7f)" 
          %(sigma, theta_plus, theta_minus))
    return sigma
    

def NLL_curverr(f, x0, x1, x2):
    """
    Function that finds error of f by using a parabola estimate from points
    x0, x1, x2 around its minimum.
    Returns standard deviation.
    """
    y0, y1, y2 = f(x0), f(x1), f(x2)
    # Curvature of parabola from lagrange polynomial
    a = y0/((x0-x1)*(x0-x2)) + y1/((x1-x0)*(x1-x2)) + y2/((x2-x0)*(x2-x1))
    sigma = 1/np.sqrt(2*a)
    print("Standard deviation from parabola curvature = %.7f" %sigma)
    return sigma


fit_data, u_flux = read('jrt3817')
E = np.arange(0, 10.05, 0.05)  # 10.05 as not including endpoint

# Initial guesses
theta = np.pi/5
m_sq = 2.7e-3

theta0, theta1, theta2, thetamin = parabola_minimiser(NLL, 0, 0.8, validate=False)
NLL_shifterror(thetamin)
NLL_curverr(NLL, theta0, theta1, theta2)

data_plot(thetamin, m_sq)


