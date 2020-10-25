# Computational-Physics
Extracting Neutrino Oscillation Parameters from a Log-Likelihood fit

Each file concerns different methods of minimising. To see the minimisation for the particular method, open the respective file and run. The extraction of the data to fit requires an internet connection as it requests teh file directly from the website. All variables and functions are defined and ran at the bottom of the script.

one_dimensional_min.py
- Finds the minimum of NLL as a function of theta and associated error estimates. Plots a best fit prediction using the obtained value.
- By setting validate=True, plots of the minimum values are plotted on a NLL plot and for every iteration to ensure convergence.

univariate.py
- Finds the minimum of NLL as a function of theta and m_sq, using successive parabolic minimisation and associated error estimates. Plots a best fit prediction using the obtained values.
- By setting validate=True, plots of the minimum values are plotted on a NLL plot and for every iteration to ensure convergence

Simultaneous_minimisation.py
- Finds the minimum of NLL as a function of theta, m_sq and cross section using simulated annealing and associated error estimates. Plots a best fit prediction using the obtained values.
- By setting plot_convcheck=True, plots of the minimum values for every iteration are plotted to ensure convergence.
- By setting plot_surf=True, plots of the minimum values at every point are plotted on a NLL ontour plot (only suported for 2d).


<b> Highlight results </b>

![contours](https://github.com/jakobtorben/Computational-Physics/blob/main/Figures/contour.png?raw=true)

Contour plot that shows the path of the simulated annealing method
in two dimensions, in it’s search for the global minimum. The two minima
are indistinguishable in the probability and have similar NLL values.

![3d parameters](https://github.com/jakobtorben/Computational-Physics/blob/main/Figures/three_dim.png?raw=true)

An updated best fit of the oscillation prediction where the cross-section
interaction has been included. The parameters are estimated using simulated
annealing for theta_(23), delta m_(sq) and simga_(cs) with 5000 cycles and 10 trials each. The
oscillated prediction shows agreement with the detector data and gives the
best estimate on the parameters.
