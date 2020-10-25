# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 21:31:50 2019

@author: jakob
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import random
from matplotlib import cm
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


def data_plot(theta, m_sq, cs=1):
    """
    Function that plots the results as binned histograms. The energy bins is
    defined outside the funciton.
    """
    # Create one entry for every bin and weight it from the data set
    fig, ax = plt.subplots()
    ax.hist(E[0:-1], bins=E, weights=fit_data, histtype='step',
            color='black', label='Detector data')
    if cs !=1 :
        ax.hist(E[1:], bins=E, weights=Prob_osc(E[1:], theta, m_sq)*cs*E[1:]*u_flux,
            histtype='step', color='blue', label='Best fit Oscillated Prediction')
    else:
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
    return 1 - (np.sin(2*theta))**2 * (np.sin(1.267*m_sq*L/(E)))**2


def NLL_point2d(theta, m_sq):
    """
    Function that calculates NLL for numbers theta and m_sq
    """
    ki = fit_data
    lam = Prob_osc(E[1:], theta, m_sq)*u_flux
    sum = 0
    for i in range(len(lam)):
        # Stirling approx breaks down for ki = 0
        if ki[i] == 0:
            sum += lam[i]
        else:
            sum += lam[i] - ki[i] + ki[i]*np.log(ki[i]/lam[i])
    return sum


def NLL_point3d(theta, m_sq, cs):
    """
    Function that calculates NLL for numbers theta, m_sq and cross section cs with array input
    """
    ki = fit_data
    lam = Prob_osc(E[1:], theta, m_sq)*u_flux*cs
    sum = 0
    for i in range(len(lam)):
        # Stirling approx breaks down for ki = 0
        if ki[i] == 0:
            sum += lam[i]
        else:
            sum += lam[i] - ki[i] + ki[i]*np.log(ki[i]/lam[i])
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

    
def sim_anneal(f, kwargs, limits, plot_convcheck=True, plot_surf=False):
    """
    Function that implements simulated annealing for general number of arguments.
    Dictionaries are used as different arguments are used in different orders.
    Surface plot only for two-dimensional minimisation
    Returns minimum points for each paramter, its uncertainty and NLL value at min
    """
    
    cycles = 500
    trials = 10
    # Relative step size between [0,1]
    step = 0.01
    # Initial temperature
    T = 1
    # Final temperature
    Tf = 1e-8
    # Fraction to reduce the temperature between cycles
    reduce = (Tf/T)**(1/(cycles-1))
    
    if plot_surf == True:
        m_sq = np.linspace(0, 5e-3, 30)
        theta = np.linspace(0, np.pi/2, 30)
        X,Y = np.meshgrid(theta, m_sq) # grid of point
        Z = NLL_arr2d(X, Y) # evaluation of the function on the grid
        fig1, ax1 = plt.subplots()
        surf = ax1.contour(X, Y, Z, 30, cmap=cm.RdBu)
        cb = fig1.colorbar(surf, shrink=0.8, pad=0.02, fraction=0.1)
        cb.set_label('Negative Log-Likelihood')
        ax1.set_xlabel('θ23')
        ax1.set_ylabel('m23\N{SUPERSCRIPT TWO}')
        fig1.tight_layout()
    
    perturb = {}
    all_kwargs = {}
    for k in kwargs.keys():
        # Set limit to random numbers for every variable as frac of inital guess
        perturb[k] = limits[k]*step
        all_kwargs[k] = np.zeros(cycles + 1)
        all_kwargs[k][0] = kwargs[k]
    
    accepted = 0  # Initialise the number accepted perturbation
    # Inital energy of unperturbed function
    Ei = f(**kwargs)
    E = np.zeros(cycles + 1)
    E[0] = Ei
    Ep = {}  # Perturbed energy
    p_kwargs = {}
    c_kwargs = kwargs.copy()
    
    for i in range(cycles):
        for j in range(trials):
            # Iterates through keyword names of dictionary to perturb every variable
            for k in kwargs.keys():
                # Adjust limits to increase accuracy as minimum is approached
                lim = perturb[k]*(cycles-i)/(cycles)
                # Perturb current kwargs by a random number within limits
                p_kwargs[k] = c_kwargs[k] + random.uniform(-lim, lim)
                # Ensure that point is still within func limits [0, lim]
                p_kwargs[k] = max(min(p_kwargs[k], limits[k]), 0)

            Ep = f(**p_kwargs)  # Perturbed energy
            Ec = f(**c_kwargs)  # Current unperturbed energy
            #print(Ep, Ec)
            deltaE = abs(Ep - Ec)
            if Ep > Ec:
                pacc = np.exp(-deltaE/T)
                if pacc > random.random():
                    accept = True
                else:
                    accept = False
            else:
                #  Accept with probability 1 for positive deltaE
                accept = True
            if accept == True:
                c_kwargs = p_kwargs.copy()
                Ec = Ep
                accepted += 1
                    
        # Record all the best variable values after every cycle                                            
        for k in kwargs.keys():
            all_kwargs[k][i+1] = c_kwargs[k]
        # Record funciton value after every cycle
        E[i+1] = f(**c_kwargs)
        # Reduce the temperature before next cycle
        T *= reduce
        if plot_surf == True:
            ax1.plot(c_kwargs['theta'], c_kwargs['m_sq'], 'o', color='black',
                    markersize=2.5)
    error = {}
    points = c_kwargs.copy()  # Avoid changing c_kwargs that will be returned
    for k in kwargs.keys():
        error[k] = NLL_curverr(f, points[k]*0.9, points[k],
             points[k]*1.1, points, k)
        
    if plot_convcheck == True:
        plt.figure()
        plt.plot(range(0, cycles+1), E)
        plt.xlabel('Iteration')
        plt.ylabel('NLL value')
        for k in kwargs.keys():
            plt.figure()
            plt.plot(range(cycles+1), all_kwargs[k])
            plt.xlabel("Iteration")
            plt.ylabel(str(k))
            
    print("Accepted: %.i out of %.i" %(accepted, cycles*trials))
    print("Best solution: ", c_kwargs)
    print("Uncertainty estimate from parabola curvature: ", error)
    print("Best function value: ", f(**c_kwargs))
    return c_kwargs, error, f(**c_kwargs)


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
# Define energy bins, 10.05 as not including endpoint
E = np.arange(0, 10.05, 0.05)

limits = {"theta": np.pi/2, "m_sq": 5e-3}
kwargs = {"theta": 0.3, "m_sq": 2e-3}
c_kwargs, error, NLL_value = sim_anneal(NLL_point2d, kwargs, limits, plot_convcheck=False, plot_surf=False)
data_plot(**c_kwargs)

limits = {"theta": np.pi/2, "m_sq": 5e-3, "cs": 3}
kwargs = {"theta": np.pi/5, "m_sq": 2.7e-3, "cs": 2}
c_kwargs, error, NLL_value = sim_anneal(NLL_point3d, kwargs, limits, plot_convcheck=False)
data_plot(**c_kwargs)





