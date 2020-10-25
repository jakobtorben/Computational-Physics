# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:31:50 2019

@author: jakob
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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
    urllib.request.urlretrieve(url, str(shortcode)+".txt")
    # Request teh datafile from url
    data = urllib.request.urlopen(url)
    count = 0
    pos = []
    for line in data:
        count += 1
        if line == b'\n':  # Finds where data starts and ends form empty line
            pos.append(count)
        
    fit_data = np.loadtxt(str(shortcode)+".txt", delimiter="\n", skiprows=pos[0],
                      max_rows=(pos[1]-pos[0]))
    u_flux = np.loadtxt(str(shortcode)+".txt", delimiter="\n", skiprows=pos[2])
    return fit_data, u_flux



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
    #ax.hist(E[1:], bins=E, weights=u_flux,
    #        histtype='step', color='red', label='Unoscillated Prediction')
    ax.set_xlabel("Neutrino Energy (GeV)")
    ax.set_ylabel("Events / (GeV)")
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.legend()
    plt.show()


def Prob_osc(E, theta, m_sq, L=295):
    return 1 - (np.sin(2*theta))**2 * (np.sin(1.267*m_sq*L/E))**2


def NLL(theta, m_sq):
    """
    Function that calculates NLL for numbers theta and m_sq
    """
    k = fit_data
    # Find lambda from oscillation probability and unoscillated flux
    lam = Prob_osc(E[1:], theta, m_sq)*u_flux
    sum = 0
    for j in range(len(lam)):
        # Stirling approx breaks down for ki = 0
        if k[j] == 0:
            sum += lam[j]
        else:
            sum += lam[j] - k[j] + k[j]*np.log(k[j]/lam[j])
    return sum


def NLL_arr2d(theta, m_sq):
    """
    Function that calculates NLL for numbers theta and m_sq with array input
    """
    if type(theta) == np.ndarray: N = len(theta)
    if type(m_sq) == np.ndarray: N = len(m_sq)
    ki = fit_data
    NLL = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
                # Find lambda from oscillation probability and unoscillated flux
                lam = Prob_osc(E[1:], theta[i, j], m_sq[i, j])*u_flux
                sum = 0
                for k in range(len(lam)):
                    # Stirling approx breaks down for ki = 0
                    if ki[k] == 0:
                        sum += lam[k]
                    else:
                        sum += lam[k] - ki[k] + ki[k]*np.log(ki[k]/lam[k])
                    NLL[i, j] = sum
    return NLL


def parabola_minimiser(f, start, stop, kwargs, var):
    """
    Function that finds the minimum of single parabola. The inital 
    range is given where start, stop needs to be in a range where the minimum is and f has positive curvature.
    Returns the three points thatdefine parabola x0, x1, x2 and minimum x3. 
    """
    # Initially spread points out to cover whole range
    x0, x2 = start, stop
    x1 = (stop-start)/2

    kwargs[var] = x0
    y0 = f(**kwargs)
    kwargs[var] = x1
    y1 = f(**kwargs)
    kwargs[var] = x2
    y2 = f(**kwargs)

    # Find position of minimum from parabola
    x3 = 0.5*((x2*x2 - x1*x1)*y0 + (x0*x0 - x2*x2)*y1 + (x1*x1 - x0*x0)*y2) \
            /((x2 - x1)*y0 + (x0 - x2)*y1 + (x1 - x0)*y2)
    kwargs[var] = x3
    y3 = f(**kwargs)

    xpoints = [x0, x1, x2, x3]
    ypoints = [y0, y1, y2, y3]
    del xpoints[np.argmax(ypoints)]  # Keep x for three smallest f(x)
    xpoints.sort()
    x0, x1, x2 = xpoints
    return x0, x1, x2, x3


def univariate(theta, thetamin, thetamax, m_sq, m_sqmin, m_sqmax, validate=False):
    """ 
    Funciton that minimises two-dimensional function by using parabolic minimiser 
    successively. If validate= True plots convergene plots and surface plots of opints.
    Retruns thetamin, m_sqmin and standard deviations: thetaerr, m_sqerr
    """
    
    tol = 3e-8  # Tolerance, not smaller than sqrt(epsilon)
    max_it = 1000  # Maximum number of iterations
    
    kwargs = {"theta": theta}
    m_sq0, m_sq1, m_sq2, m_sqmin = parabola_minimiser(NLL, m_sqmin, m_sqmax, kwargs, 'm_sq')
    kwargs = {"m_sq": m_sq}
    theta0, theta1, theta2, thetamin = parabola_minimiser(NLL, thetamin, thetamax, kwargs, 'theta')
    if validate == True:
        all_thetamin, all_m_sqmin = [],[]
        
        m_sq = np.linspace(0, 5e-3, 50)
        theta = np.linspace(0, np.pi/2, 50)
        X,Y = np.meshgrid(theta, m_sq) # grid of point
        Z = NLL_arr2d(X, Y) # evaluation of the function on the grid
        
        fig, ax = plt.subplots()
        surf = ax.contourf(X, Y, Z, 30, cmap=cm.RdBu)
        fig.colorbar(surf)
        
    count = 0
    while count < max_it:
        thetamin_i, m_sqmin_i = thetamin, m_sqmin 

        kwargs ={"m_sq": m_sqmin}
        theta0, theta1, theta2, thetamin = parabola_minimiser(NLL, theta0, theta2, kwargs, 'theta')
        kwargs = {"theta": thetamin}
        m_sq0, m_sq1, m_sq2, m_sqmin = parabola_minimiser(NLL, m_sq0, m_sq2, kwargs, 'm_sq')
        
        if validate == True:
            all_thetamin.append(thetamin)
            all_m_sqmin.append(m_sqmin)
            ax.plot(thetamin, m_sqmin, 'x', color='black')
            
        # Find difference between iterations using pythagoras
        delta = np.sqrt((thetamin-thetamin_i)**2 + (m_sqmin - m_sqmin_i)**2)
        count += 1
        if delta < tol:
            if validate == True:
                plt.figure()
                plt.plot(np.arange(0, count, 1), all_thetamin)
                plt.figure()
                plt.plot(np.arange(0, count, 1), all_m_sqmin)
            break
    kwargs ={"m_sq": m_sqmin}
    thetaerr = NLL_curverr(NLL, theta0, theta1, theta2, kwargs, 'theta')
    kwargs = {"theta": thetamin}
    m_sqerr = NLL_curverr(NLL, m_sq0, m_sq1, m_sq2, kwargs, 'm_sq')
        
    if count == max_it:
        print("Minimasation did not converge in %.i iterations" %max_it)
    else:
        print("Minimum found in %.i iterations" %count)
        print("thetamin = %.7f ± %.7f" %(thetamin, thetaerr))
        print("m_sqmin = %.7f ± %.7f" %(m_sqmin, m_sqerr))
        return thetamin, m_sqmin, thetaerr, m_sqerr


def NLL_curverr(f, x0, x1, x2, kwargs, var):
    """
    Function that finds error of f by using a parabola estimate from points
    x0, x1, x2 around its minimum.
    Returns standard deviation.
    """
    # y points found by assigning the variable of f for x0, x1 and x2
    kwargs[var] = x0
    y0 = f(**kwargs)
    kwargs[var] = x1
    y1 = f(**kwargs)
    kwargs[var] = x2
    y2 = f(**kwargs)
    
    # Curvature of parabola from lagrange polynomial
    a = y0/((x0-x1)*(x0-x2)) + y1/((x1-x0)*(x1-x2)) + y2/((x2-x0)*(x2-x1))
    sigma = 1/np.sqrt(2*a)
    return sigma


fit_data, u_flux = read('jrt3817')
# Initial guesses
theta = np.pi/5
m_sq = 2.8e-3
E = np.arange(0, 10.05, 0.05)  # 10.05 as not including endpoint

thetamin, m_sqmin, thetaerr, m_sqerr = univariate(theta, 0, 0.8, m_sq, 0, 0.007,
                                                  validate=False)
data_plot(thetamin, m_sqmin)


