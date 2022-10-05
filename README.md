# GAMes (Genetic Algorithm for multi-Messenger)

GAMes is a Python package that provides utility functions for optimizing the physical parameters of AGNs/blazars. GAMes works by iteratively comparing photon and neutrino spectra obtained through simulations with observational data 

The software has been tested using the AM3 radiation model (S. Gao 2017) to obtain the blazar SEDs.

Xavier Rodrigues 2019

## Prerequisites

[Python 3.7]

[DEAP] (https://github.com/DEAP/deap) -- evolutionary computation tools for Python

[AM3] -- the AM3 libraries are necessary to make a new Python AM3 extension module besides the one given (e.g. for changing the blazar parameters to be optimized)

