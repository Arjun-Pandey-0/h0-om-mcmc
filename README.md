# Parameter Constraints with $H(z)$ and MCMC sampling

This repository contains a Markov Chain Monte Carlo pipeline to constrain certain cosmological parameters: the Hubble Constant ($H_0$) and matter density parameter ($\Omega_m$) using observational $H(z)$ data from published cosmic chronometers papers. The analysis computes full parameter posteriors by evaluating a custom Gaussian likelihood and parameter priors from published literature values. Off-diagonal correlation elements are made using `RectBivariateSpline` on known covariance blocks.

## Requirements
1. numpy>=1.21.0
2. scipy>=1.8.0
3. pandas>=1.4.0
4. matplotlib>=3.5.0
5. seaborn>=0.11.0
6. emcee>=3.1.0
7. pyccl>=3.0.0

This repo contains:
1. `mcmc.py` which the data was analyzed
2. `Hz_dat.csv` which is the dataset with the $H(z)$ values used in this project
3. `README.md` which is the file you're reading right now!
