# Parameter Constraints with $H(z)$ and MCMC sampling

This repository contains a Markov Chain Monte Carlo pipeline to constrain certain cosmological parameters: the Hubble Constant ($H_0$) and matter density parameter ($\Omega_m$) using observational $H(z)$ data from published cosmic chronometers papers. The analysis computes full parameter posteriors by evaluating a Gaussian likelihood and parameter priors from published values of $H_0$ and $\Omega_m$. Off-diagonal correlation elements are made using `RectBivariateSpline` on known covariance blocks.

## This repository contains:
1. `analysis.py` which builds the covariance matrices and runs the 4 MCMC fits
2. `plotting.py` which makes all the plots by running `analysis.py` first
3. `Hz_data.csv` which is the dataset with the $H(z)$ values used in this project
4. `requirements.txt` which lists the packages needed to run this 
5. `README.md` which is the file you're reading right now!

