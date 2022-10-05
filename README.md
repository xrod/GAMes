# GAMMes (Genetic Algorithm for Multi-Messenger)

Python package that provides utility functions for optimizing the physical parameters of AGNs/blazars by fitting to observational data the Spectral Energy Distribution (SED) of photons and neutrinos obtained by a theoretical model.

The software has been tested using the AM3 radiation model (S. Gao 2017) to obtain the blazar SEDs.

Xavier Rodrigues 2019

## Prerequisites

[Python 3.7]

[DEAP] (https://github.com/DEAP/deap) -- evolutionary computation tools for Python

[AM3] -- the AM3 libraries are necessary to make a new Python AM3 extension module besides the one given (e.g. for changing the blazar parameters to be optimized)

