# genetic-algorithm-for-blazars
Genetic algorythm in Python that optimizes the physical parameters of AGNs/blazars by comparing the Spectral Energy Distribution (SED) obtained by a radiation model to the observational data.

The software has been tested using the AM3 radiation model (S. Gao 2017) to obtain the blazar SEDs.

Xavier Rodrigues 2018

## Prerequisites

[Python 2.7]
[DEAP] (https://github.com/DEAP/deap) -- evolutionary computation tools for Python
[AM3] -- the AM3 libraries are necessary to make a new Python AM3 extension module besides the one given (e.g. for changing the blazar parameters to be optimized)

