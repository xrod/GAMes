#!/usr/bin/env python2.7

import am3_ext as am3 # Module to perform the radiation modeling
import muon_fit_ext as mf # Module to compute the expected number of neutrinos is IceCube
import h5py
from deap import base, creator, tools
import random
import time
import multiprocessing
from multiprocessing import Pool
import subprocess
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter
import matplotlib.ticker as mtick
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.offsetbox import AnchoredText
import matplotlib.colors as colors
import matplotlib.patches as patches
import seaborn as sns
from skimage import data
from skimage import io, img_as_float
from skimage.transform import rescale, resize, downscale_local_mean


from scipy.integrate import simps
from scipy.integrate import quad, tplquad, dblquad
from scipy.interpolate import PchipInterpolator
import scipy as scp
import numpy as np
import copy

import os
import sys
import time
sys.path.append(os.getcwd())

exec_file = "RunAGNModel" # executable that performs the AGN simulation 

info_prefix = "txs885"+"sed" # control files printed by the simulation program
h5_prefix = "res_txs885" # control files to be printed
job_prefix = "t885" # job prefix for qsub 
prefix = "txs885"+"sed" # data files printed by AM3
direc = "/lustre/fs23/group/that/xr/ana/am3/" # input directory (with the simulation output) 

BIN_X = 550
BIN_P = 194

# Constants in cgs units

z_pks = 0.361
z_txs = 0.3365

c = 3e10 
h_bar = 1.0545726e-27
r_0 = 2.81794e-13 
m_e = 9.1095e-28
m_p = 1.67e-24 
e = 4.8032e-10 
wien = 2.89e-1 # cm.K
hc = 1.9878225197022526e-16
boltzmann = 1.38e-16 # erg/K
sigma_thomson = 8*np.pi/3*r_0**2
sigma_pp = 7e-26
alpha = e**2/h_bar/c

# Conversion factors

ev_to_hz = 1./(h_bar*2*np.pi)/624.15e9
Mpc2cm = 3e24
microJy2cgs = 1.5e-3

# Evolution parameters

CXPB = 0.5 # Crossover prob
MUTPB = 0.33 # Mutation prob 
INDPB = 1./9
ETA = 20 # How similar the mutant will be to the progenitors

# Limits of the fit parameter ranges when mutating
b_lims = [-.3,2.] # 50 
r_blob_lims = [14.,17.]#[15.,17.] # 15.9
f_elec_lims = [-3.,3.]  # -2
f_prot_lims = [-2.,3.] # 3
lorentz_lims = [10.,50.]#[5,50] # 18
PRO_inj_max_lims = [128.,128.0001] # if we want maximum to hit the broad line (re-calculate for specific gamma)
e_inj_max_lims = []#[50.,400.] # 53
e_inj_index_lims = []#[.6,1.6] # 1.2
lum_thermal_lims = [43.,45.]#[41.,47.]
dust_temp_lims = [4.9e5,5.1e5]# [5e4,2e5]#[1000,70000] # VARY
t_simul_lims = []#[100.,100+1e-7]#[1000,70000]
PRO_inj_index_lims = [120,140]#[1.5,2.5] # VARY
frac_esc_lims = [-5.,-3.]#[1000,70000]  # proton escape rate
r_blr_lims = [16.5,17.5]#[1000,70000]  # proton escape rate


BOUND_LOW = [b_lims[0], r_blob_lims[0], f_elec_lims[0], f_prot_lims[0], lorentz_lims[0], lum_thermal_lims[0], dust_temp_lims[0],frac_esc_lims[0], r_blr_lims[0]]
BOUND_UP  = [b_lims[1], r_blob_lims[1], f_elec_lims[1], f_prot_lims[1], lorentz_lims[1], lum_thermal_lims[1], dust_temp_lims[1],frac_esc_lims[1], r_blr_lims[1]]
 
def plot_data():
    plt.scatter(all_data_x, all_data_y, c=(1,0,0,0.2),s=2)

###################
# Physics functions
###################

def lumdistance(z):
    '''
    returns luminosity distance in Mpc
    par z: source redshift
    '''
    return (1.0+z)*(1.312146e28)*\
        (1.02351*z-0.285939*z**2.0+0.0521128*z**3.0-0.00528826*z**4.0+0.000224442*z**5.0)

def PlanckDistribution(en, temperature):
    '''
    e [eV], T[K]
    return: E^2 dN/dE (a.u.)
    '''
    E = en/624.15e9
    res = E**3/hc/(np.exp(E/boltzmann/temperature)-1)
    return res

def gammagamma_coeff(e,et,nt):
    '''
    Returns photon-photon absorption coefficient
    par e: photon energy [eV]
    par et: target photon energy [eV]
    par nt: target photon density [eV^-1 cm^-3]
    '''
    M = m_e*3e10**2*624.15e9
    e = e.reshape(1,e.size)
    et = et.reshape(et.size,1)
    nt = nt.reshape(nt.size,1)
    beta = ((e*et-M*M)/(e*et+M*M))**0.5
    sigma = .5*np.pi*r_0**2*(1-beta**2)*((3-beta**4)*np.log((1+beta)/(1-beta))-2*beta*(2-beta**2))

    integrand = sigma*nt*(1-(M*M)/(e*et))
    cond = (et*e<M)
    integrand = np.ma.masked_where(cond,integrand)

    res = np.trapz(integrand,et,axis=0)
    return res.reshape(res.size)

def gamma_absorption_exact(e,et,nt,R):
    '''
    par: e, et [eV]
    par: nt [eV^-1 cm^-3]
    par: R [cm]
    '''
    tau = gammagamma_coeff(e,et,nt)*R
    tau[tau<1e-15] = 1e-15

    return 1./tau*(1.-np.exp(-tau))


#############################
# Functions for data analysis
#############################

def not_empty(individual):
    myfile = direc + prefix + make_string(individual,"_") + ".dat"
    if os.stat(myfile).st_size:
        return True
    return False

def ff(arr1,arr2,thresh=None):
    thresh = np.max(np.fabs(arr2))/10. if thresh is None else thresh

    arr1 = arr1*np.array([np.fabs(arr1)>thresh]).reshape(arr1.size)
    arr2 = arr2*np.array([np.fabs(arr2)>thresh]).reshape(arr2.size)

    cond1 = np.logical_and(arr1,arr2==0)
    cond2 = np.logical_and(arr2,arr1==0)
    cond3 = arr1*arr2 < 0

    cond = cond1 + cond2 + cond3

    return cond + 1

#####################
# Evolution functions
#####################


def calc_chisquare_txs(data_x, data_y, sigma_y, is_uplim, model_x, model_y):
    """ Returns chi squared of model compared to data points
    
    Keyword arguments:
    individual-- the set of blazar parameters to be evaluated
    data_x, data_y-- [energy_arr [eV], lum_arr [erg/s] of the data (obs frame)
    weights_y-- weights for chi^2 test
    model_x, model_y-- [energy_arr [eV], lum_arr [erg/s] of the model (obs frame)
    """
    model_interp_y = 10**np.interp(np.log10(data_x),np.log10(model_x),np.log10(model_y))

    # chi_prelim = np.nansum((model_interp_y-data_y)**2*weights_y)
    # if not chi_prelim or np.isnan(chi_prelim):
    #     return (chi_prelim, 0, [0], [0], [0], 0, [0],[0])

    model_points = model_interp_y
    data_points = data_y

    chi2 = 0.
    chi2_max = 0.
    for y, xd, yd, iul, s in zip(model_interp_y, data_x, data_y, is_uplim, sigma_y):

        chival = (y-yd)**2/s**2*(1-(y<yd)*iul)
        chival = 1e5 if np.isinf(chival) or np.isnan(chival) else chival
        chi2 += chival
        chi2_max += yd**2/s**2

    norm = 5./chi2_max # Relative norm between SED and neutrinos

    return [chi2*norm]


def fix_wrong_string(stg):
    return int(stg[0]=="e")*"1" + stg


def get_thermal_BHF(xSRF, rBLR, L, lor, temp, thomson = 0.01):
    '''
    Reurns x[eV] and y[s^-1]
    y == disk emission spectrum in the BLR
    '''
    xdist = xSRF # bug fixed 20190116: was xSRF/lor
    pdist = PlanckDistribution(xdist, temp)/xdist # EdN/dE
    norm = L*624.15e9/np.trapz(pdist,xdist)
    pdist *= norm
    return pdist

def get_thermal_SRF(xSRF, rBLR, L, lor, temp, thomson = 0.01):
    '''
    Reurns x[eV] and y[cm^-3]
    y == disk emission spectrum in the BLR
    '''
    dist_BHF = get_thermal_BHF(xSRF,rBLR,L,lor,temp,thomson)
    dist_SRF = dist_BHF*thomson*lor/(4*rBLR**2*3e10)
    
    return dist_SRF

def get_BLR_absorption(xSRF, rBLR, L, lor, temp, thomson=0.01):
    planckSRF = get_thermal_SRF(xSRF,rBLR,L,lor,temp,thomson)
    return gamma_absorption_exact(xSRF, xSRF, planckSRF/xSRF, rBLR)


def evaluate(individual, data_x, data_y, sigma_y, is_uplim, dx=0.1, xi=-280):
    """ Returns array with chi-squared values with same size as data_x
    
    Keyword arguments:
    individual-- the set of blazar parameters to be evaluated
    data_x, data_y-- [log10(energy_arr [GeV]), log10(flux_arr [erg/cm^2/s])
    data_weights-- array of weights for chi2 test
    sed_file-- name of file with array of ln(dL/dE [s^-1])
    dx-- AM3 parameter
    xi-- AM3 parameter
    """	

    # sed_y = np.loadtxt(filename, converters = {0: fix_wrong_string}) # from AM3: s^-1
    # data_arr = np.loadtxt(filename)

    ind_string = make_string(individual,"_")
    ind_info = ind_string.strip("_").split("_")

    filename = direc + prefix + ind_string + ".dat"
    # Exctract individual's info
    logB = float(ind_info[0])
    logRblob = float(ind_info[1])
    logFRACe = float(ind_info[2])
    logFRACp = float(ind_info[3])
    lorentz = float(ind_info[4])
    PRO_inj_max = float(ind_info[5])
    e_inj_max = float(ind_info[6])
    e_inj_index = float(ind_info[7])
    log_thermal_luminosity = float(ind_info[8]) # change with parameters
    temperature = float(ind_info[9])
    f_escape = 10**float(ind_info[10]) # change with parameters
    rBLR = 10**float(ind_info[11]) # change with parameters

    if os.path.exists(filename):
    	content = np.loadtxt(filename,converters={0: fix_wrong_string})
    else:
    	content = np.full(3+2*BIN_X+BIN_P+1,0.)

    nu_y = content[3+BIN_X:3+BIN_X+BIN_P]
    sed_y = content[3+BIN_X+BIN_P:3+2*BIN_X+BIN_P]

    sed_x = np.exp((np.arange(sed_y.size)+xi)*dx)*m_e*c*c*624.15e9*lorentz # eV
    nu_x = np.exp((np.arange(nu_y.size))*dx)*1e9*lorentz # eV

    ## Photon absorption in the broad line region (BLR)
    gg_arr = get_BLR_absorption(sed_x, rBLR, 10**log_thermal_luminosity, lorentz, temperature)
    sed_y *= gg_arr
    
    chi2 = 1e5
    if not np.all(sed_y==0.):
        sed_y *= sed_x/624.15e9*(4./3*np.pi*(10**logRblob)**2*3e10)*lorentz**3*sigma_thomson**-1.5 # erg/s
        chi2 = calc_chisquare_txs(data_x,data_y,sigma_y,is_uplim,sed_x,sed_y)[0]

    nu_fit=5.
    if not np.all(nu_y==0.):
        try:
            nu_number = mf.get_muon_number(lorentz, 10**logRblob, list(nu_y), nu_y.size)
            nu_number /= 3.
            nu_fit = (nu_number-5.)**2/5.
        except:
            print "Error in neutrino fit routine:", make_string(individual,"_")
            nu_fit = 1e3

    res = chi2 + nu_fit 
    res = 1e5 if np.isnan(res) else res

    # print "%s SED chi2: %.2e nu chi2: %.2e"%(filename, chi2, nu_fit) 
    print "Chi2, nufit ==", chi2, nu_fit
    return [res]


# Caution: all_data_x and all_data_y MUST be set properly
def eval_wrapper(individual):
    return toolbox.evaluate(individual, all_data_x, all_data_y, sigma_y, is_uplim)


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

def round_random(low,high):
    x = random.uniform(low,high)
    return round(x,4)

# Here we must register the attributes of our individuals, ie the parameters that
#  will be scanned, and register them in the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_b",            round_random,BOUND_LOW[0],BOUND_UP[0])
toolbox.register("attr_r_blob",       round_random,BOUND_LOW[1],BOUND_UP[1])
toolbox.register("attr_f_elec",       round_random,BOUND_LOW[2],BOUND_UP[2])
toolbox.register("attr_f_prot",       round_random,BOUND_LOW[3],BOUND_UP[3])
toolbox.register("attr_lorentz",      round_random,BOUND_LOW[4],BOUND_UP[4])
# toolbox.register("attr_pro_max",      round_random,BOUND_LOW[5],BOUND_UP[5])
# toolbox.register("attr_e_max",      random.randint,BOUND_LOW[6],BOUND_UP[6])
# toolbox.register("attr_e_index",      round_random,BOUND_LOW[7],BOUND_UP[7])
toolbox.register("attr_therm_lum",    round_random,BOUND_LOW[5],BOUND_UP[5])
toolbox.register("attr_dust_temp",    round_random,BOUND_LOW[6],BOUND_UP[6])
toolbox.register("attr_frac_esc",    round_random,BOUND_LOW[7],BOUND_UP[7])
toolbox.register("attr_r_blr",    round_random,BOUND_LOW[8],BOUND_UP[8])

# Now we register an individual, which is a set of blazar parameters. All
# the attributes registered above must be passed to the attribute list
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_b,
                  toolbox.attr_r_blob,
                  toolbox.attr_f_elec,
                  toolbox.attr_f_prot,
                  toolbox.attr_lorentz,
                  toolbox.attr_therm_lum,
                  toolbox.attr_dust_temp,
                  toolbox.attr_frac_esc,
                  toolbox.attr_r_blr,
                  ),
                  n=1)
                 
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Register population of individuals
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=ETA, indpb=INDPB)
toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("evaluate", evaluate)

def run_script(string):
    subprocess.call(string, shell=True)
    return

def update_backup(current_file,backup_file):
    if os.path.exists(backup_file):
    	os.remove(backup_file)
    subprocess.call("cp "+current_file+" "+backup_file, shell=True)

    return



def run_individual(individual, i, gen):
    job_name = job_prefix+"_%i_%i"%(gen, i+1)
    qsub_options = "qsub -l h_rt=00:15:00 -l h_rss=2G -N %s -j y -b y -o /afs/ifh.de/group/that/work-xrod/ana/NC/AM3/branches/X_branch/evolution/logs2 "%(job_name)
    location = "/afs/ifh.de/group/that/work-xrod/ana/NC/AM3/branches/X_branch/"
    executable = exec_file + make_string(individual, " ")
    script =  qsub_options + location + executable
    run_script(script)  
    return

def await_files(filenames):
    
    tac = time.time()

    njobs = 1
    while njobs:
        time.sleep(5)

        njobs = int(subprocess.check_output("echo $(qstat -u xavier | grep %s | wc -l)"%job_prefix, shell=True))
        njobs_running = int(subprocess.check_output("echo $(qstat -u xavier | grep %s | grep '  r  ' | wc -l)"%job_prefix, shell=True))

        print "Awaiting %i jobs, %s running."%(njobs, njobs_running)

    tic = time.time()
    print "Generation calculated. Time: %f min"%((tic-tac)/60.)


def format_zero(par, fmt="%.4f"):
    if round(par,4)==0:
        return fmt%0
    return fmt%par

def break_string(filename,prefix,sep="_"):
    stripped_string = filename.strip(prefix)
    stripped_string = stripped_string.strip(".dat")
    
    return np.array([float(s) for s in stripped_string.strip("_")]) 


# Here one chooses the parameters that go into the fit, and calculate those that are fixed
def make_string(individual,sep="_"):

    par_arr = tuple(individual)
    res = sep+format_zero(par_arr[0])
    res+= sep+format_zero(par_arr[1])
    res+= sep+format_zero(par_arr[2])
    res+= sep+format_zero(par_arr[3])
    res+= sep+format_zero(par_arr[4])
    ## Either calcualte max proton energy so as to interact with the broad line photons...
    # epmaxGev = 0.1887/(10.1e-9*float(par_arr[4]))
    # PROinjmax = 10*np.log(epmaxGev)

    ## ... or to interact with the thermal photons from the disk emission 
    epmaxGev = 0.1887 * wien / float(par_arr[4]) / float(par_arr[6]) / hc / 624.15
    PROinjmax = 10*np.log(epmaxGev)
    res+= sep+format_zero(PROinjmax)

    # Calculate max electron injection energy so as to get synchrotron peak at 3 eV
    B = float(10**par_arr[0])
    lorentz = float(par_arr[4])
    optimal_gamma_synch = np.sqrt((1.69e20*(3/624.15e9)/lorentz*(1+z_txs)/B))
    e_inj_max = 10*np.log(optimal_gamma_synch)
    res+= sep+format_zero(int(e_inj_max)) 
    
    res+= sep+format_zero(1.0)
    res+= sep+format_zero(par_arr[5])

    ## Either calculate the accretion disk temperature to match 
    ## the maximum ptoton energy...
    # e_p_max_GeV = np.exp(0.1*float(par_arr[5]))
    # dust_temperature = 0.1887/e_p_max_GeV/lorentz*wien/hc/624.15 // 0.1887 has been calculated precisely 
    # res+= sep+format_zero(dust_temperature)

    #... or set it as a parmeter to scan
    res+= sep+format_zero(par_arr[6]) # disk temperature
    res+= sep+format_zero(par_arr[7])
    res+= sep+format_zero(par_arr[8])

    return res



def from_string(stg, sep="_"):
    arr = np.array([float(s) for s in stg.strip(sep).split(sep)])
    return arr



def add_to_matrix(h5, filename, individual, generation):
    '''Add file info to the h5 file with all the results. 

    filename: string of file name, including directory and .dat extension 
    individual: tuple with parameters 
    gen: generation number, starting from 0
    '''
    gen = "g"+str(generation)
    ind = make_string(individual,"_")

    try:# if ind not in h5[gen]:
        h5[gen].create_dataset(ind, data = [1])

        try: # if os.path.exists(filename):
        	content = np.loadtxt(filename,converters={0: fix_wrong_string})
        except IOError:
        	print "Error in AM3: File", filename[35:], "does not exist"
        	content = np.full(3+2*BIN_X+BIN_P+1,0.)

        sed_internal = content[3:3+BIN_X] # in-source density 
        nu = content[3+BIN_X:3+BIN_X+BIN_P]
        sed_observed = content[3+BIN_X+BIN_P:3+2*BIN_X+BIN_P]

        try: # if ind not in h5["seds"]:
            h5["seds"].create_dataset(ind, data=sed_observed)
            h5["photon_density"].create_dataset(ind, data=sed_internal)
            h5["neutrinos"].create_dataset(ind, data=nu)
        except RuntimeError:
        	pass

    except RuntimeError:
        val = np.array(h5[gen][ind])[0]
        del h5[gen][ind]
        print "Repeated parameter set:", val+1
        h5[gen].create_dataset(ind, data = [val + 1])


def main(population, h5file, data_x, data_y, data_weights, dx=0.1, xi=-280,MUTPB=MUTPB):
    """ Main evolution function.

    Keyword arguments:
    population-- population of individuals to evolve
    data-- [ln(energy_arr[GeV]), ln(flux_arr [erg/cm^2/s])
    data_weights-- chi2_weights_array]
    dx-- AM3 parameter
    xi-- AM3 parameter
    """

    tac0 = time.time()

    fg=open(info_prefix+"_info.txt",'w')
    np.savetxt(fg,[NGEN,NIND,10],fmt="%d")
    np.savetxt(fg,[MUTPB,CXPB,ETA],fmt="%.2f")
    fg.close()

    # Calculate SEDs of the whole population in parallel

    filenames = [direc + prefix + make_string(ind,"_") + ".dat" for ind in population] 

    for i, individual in zip(np.arange(len(population)), population):
        if not os.path.exists(filenames[i]):
            run_individual(individual, i, 0)
        else:
            print "File already exists:", filenames[i]
            
    await_files(filenames)

    print "Adding info to matrix. Do not modify HDF5 file."
    h5file = h5py.File(direc + h5_prefix+".h5", "a")

    # Add SEDs and neutrino spectra to the h5 datafile for later evaluation
    for filename, individual in zip(filenames, population):
        add_to_matrix(h5file, filename, individual, 0)

    # Evaluate the entire population
    print "Beginning evaluation."
    fitnesses = map(eval_wrapper, population)



    # Give the fit to each individual and add fit to the h5 datafile
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
        indname = make_string(ind,"_")
        try:
            h5file["fits"].create_dataset(indname, data=fit)
        except:
            pass

    # Delete temprary files written by the simulation software
    for filename in filenames:
    	if os.path.exists(filename):
    		os.remove(filename)   


    h5file.close()
    print "HDF5 file closed."

    f=open(evol_tracking_file,'w')
    for ind in population:
        np.savetxt(f,ind,fmt="%.4f")
    f.close()

    for g in range(NGEN):

        print "Generation", g, "\n"
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = map(toolbox.clone, offspring)

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
            	try:
                	toolbox.mutate(mutant)
                except Exception as e:
                	print "Exception raised in toolbox.mutate:"
                	print e

                del mutant.fitness.values


        # Evaluate the new individuals (only the different ones)
        new_individuals = [ind for ind in offspring if not ind.fitness.valid]
        new_filenames = [direc + prefix + make_string(ind,"_") + ".dat" for ind in new_individuals] 

        for i, filename, individual in zip(np.arange(len(new_individuals)), new_filenames, new_individuals):
            if not os.path.exists(filename):
                run_individual(individual, i, g+1)
            else:
                print "File already exists:", filename
        await_files(new_filenames)

        # Add entire new population to h5 file
        print "Adding info to matrix. Do not modify HDF5 file."
        h5file = h5py.File(direc + h5_prefix+".h5", "a")
        offspring_filenames = [direc + prefix + make_string(ind,"_") + ".dat" for ind in offspring]
        h5file.create_group("g"+str(g+1))
        [add_to_matrix(h5file, filename, indiv, g+1) for filename, indiv in zip(offspring_filenames, offspring)]
       
        fitnesses = map(eval_wrapper, new_individuals)

        for ind, fit in zip(new_individuals, fitnesses):
            ind.fitness.values = fit
            indname = make_string(ind,"_")
            try:
                h5file["fits"].create_dataset(indname, data=fit)
            except:
                pass

        # Remove temporary files of the new individuals
        [os.remove(filename) for filename in new_filenames if os.path.exists(filename)]

        h5file.close()
        print "HDF5 file closed."

        update_backup(direc + h5_prefix+".h5",direc + h5_prefix+"_backup.h5")

        # The population is entirely replaced by the offspring
        population[:] = offspring

        print "Total time:", (time.time()-tac0)/60., "min."

        # Write population into file to keep track of evolution
        f=open(evol_tracking_file,'a')
        for ind in population:	
            np.savetxt(f,ind,fmt='%.4f')

        f.close()    

        # remove logs
        subprocess.call("rm -rf /afs/ifh.de/group/that/work-xrod/ana/NC/AM3/branches/X_branch/evolution/logs2", shell=True)
        subprocess.call("mkdir /afs/ifh.de/group/that/work-xrod/ana/NC/AM3/branches/X_branch/evolution/logs2", shell=True)

    tic0 = time.time()
    print "Done. Total time:", (tic0-tac0)/60., "min."
    return population

d_lum_pks = lumdistance(z_pks)
d_lum_txs = 1750.*Mpc2cm


# Load observational data

# PKS 1510
data_x1, data_y1, data_llim1, data_ulim1 = np.loadtxt("../../../../../../docs/data/sed_pks_1210.4552_MJD_54343.dat",unpack=True)
data_x2, data_y2, data_llim2, data_ulim2 = np.loadtxt("../../../../../../docs/data/sed_pks_1210.4552_MJD_55230.dat",unpack=True)
data_x3, data_y3, data_llim3, data_ulim3 = np.loadtxt("../../../../../../docs/data/sed_pks_1210.4552_MJD_55766_incl_xray_weighted.dat",unpack=True)
data_x4, data_y4, data_llim4, data_ulim4 = np.loadtxt("../../../../../../docs/data/sed_pks_1210.4552_MJD_55790.dat",unpack=True)

# Blazar sequence
data_x_seq_bl, data_y_seq_bl = [], []
data_x_seq_fs, data_y_seq_fs = [], []
for L in np.arange(41.5, 51.5, 1):
    name_bllac = "../../../../../../docs/data/sed_sequence_bllac_%.1f.dat"%L
    name_fsrq = "../../../../../../docs/data/sed_sequence_fsrq_%.1f.dat"%L
    bl_x, bl_y = np.loadtxt(name_bllac,unpack=True) if os.path.exists(name_bllac) else (None, None)
    fs_x, fs_y = np.loadtxt(name_fsrq,unpack=True) if os.path.exists(name_fsrq) else (None, None)
    data_x_seq_bl.append(bl_x)
    data_y_seq_bl.append(bl_y)
    data_x_seq_fs.append(fs_x)
    data_y_seq_fs.append(fs_y)

# Minimal data (5 points)
data_x_41, data_y_41 = np.loadtxt("../../../../../../docs/data/points_41.5.dat",unpack=True)

# TXS 0506+056 data
data_x_txs, data_y_txs, ymax_txs, ymin_txs = np.loadtxt("../../../../../../docs/data/txs_hist_4.dat",unpack=True)

# Convert x arrays from Hz to eV
data_x1 += np.log10((1.+z_pks)/ev_to_hz)
data_x2 += np.log10((1.+z_pks)/ev_to_hz)
data_x3 += np.log10((1.+z_pks)/ev_to_hz)
data_x4 += np.log10((1.+z_pks)/ev_to_hz)

# Convert x arrays from GeV to eV
for i in np.arange(len(data_x_seq_bl)):
    try:
        data_x_seq_bl[i] += 9 
    except:
        pass
    try:
        data_x_seq_fs[i] += 9
    except:
        pass
data_x_41 +=9 
data_x_txs *= 1e9   

# Convert photon flux [erg/s/cm^2] to luminosity [erg/s]
data_y1 += np.log10(4*np.pi*d_lum_pks**2)
data_y2 += np.log10(4*np.pi*d_lum_pks**2)
data_y3 += np.log10(4*np.pi*d_lum_pks**2)
data_y4 += np.log10(4*np.pi*d_lum_pks**2)
data_y_txs *= 4*np.pi*d_lum_txs**2
ymin_txs   *= 4*np.pi*d_lum_txs**2
ymax_txs   *= 4*np.pi*d_lum_txs**2
data_x_txs *= 1 + z_txs

## Uncomment if Blazar Sequence data to be used
# all_data_x = data_x_41 #data_x_seq_bl[0] #np.concatenate((data_x1,data_x2,data_x3))
# all_data_y = data_y_41 #data_y_seq_bl[0] #np.concatenate((data_x1,data_x2,data_x3))

## Uncomment if TXS 0506+056 data to be used
pointselec = [1,2,3,4,5,6,7]
all_data_x = data_x_txs[pointselec]
all_data_y = data_y_txs[pointselec]
all_max_y = ymax_txs[pointselec]
all_min_y = ymin_txs[pointselec]

sigma_y = (all_max_y-all_min_y)/2.

## Uncomment to make ad hoc adjustment to give preference to Xray constraints
## over other wavebands
# sigma_y[0]*=2

is_uplim = all_data_y==0


## Start with random population and run evolution
NGEN = 50 # number of generations to simulate
NIND = 10000 # number of individuals in every generation
poptest = toolbox.population(n=NIND)
evol_tracking_file = info_prefix + "_evol_track.txt"

## Run evolution

a = raw_input("Start evolution?")
if a=='n':
    exit()

# Create HDF5 file for output of the entire evolution
h5file = h5py.File(direc + h5_prefix+".h5", "w")
h5file.create_group("g0")
h5file.create_group("fits")
h5file.create_group("seds")
h5file.create_group("photon_density")
h5file.create_group("neutrinos")
h5file.create_group("neutrino_fits")
h5file.close()

popfinal = main(poptest,h5file,all_data_x, all_data_y, np.full(data_x_seq_bl[0].size,1.))
print "Evol complete."
exit()















##############
# Plot results
##############

# pop = toolbox.population(n=NIND)
# pop = main(pop, all_data_x, all_data_y, np.full(all_data_x.size,1.))
# print "Done."

# plt.scatter([ind[0] for ind in pop], [ind[1] for ind in pop])
# plt.show()


# # Load simulation results

# h5data = h5py.File(direc + h5_prefix + ".h5", "r")
# all_gens = np.array([k for k in sorted(h5data.iterkeys()) if k[0]=="g"])
# print h5data.keys()


# # Plot the 20 best individuals in the last generation
# G = "0"
# minpages = 20

# inds = np.array(h5data[G].keys())
# chis = np.array([h5data['fits'][ind][0] for ind in inds])

# print inds
# print chis

# # print "Min chi2:", np.min(chis)

# with PdfPages("plots"+info_prefix[:-3]+"_best_fits_%s.pdf"%G) as pdf:
#     for j in np.arange(min(minpages,NIND)):
#         i = np.argsort(chis)
#         ind = inds[np.argsort(chis)][j]
#         print j, i, ind, np.argsort(chis)
        
#         y_sed_array = np.array(h5data['seds'][ind])
#         x_sed_array = np.exp((np.arange(y_sed_array.size)-280.)*0.1)*m_e*c*c*624.15e9 # eV, BHF

#         pars = from_string(ind)
 		
#         logB = pars[0]
#         logRblob = pars[1]
#         logFRACe = pars[2]
#         logFRACp = pars[3]
#         lorentz = pars[4]
#         PRO_inj_max = pars[5]
#         e_inj_max = pars[6]
#         e_inj_index = pars[7]
#         log_thermal_luminosity = pars[8]
#         temperature = pars[9]
#         print logRblob
#         density_to_lum = 4.*np.pi*(10**logRblob)**2*3e10
#         x_arr, y_arr = x_sed_array/(1+z), y_sed_array*density_to_lum*lorentz**3*sigma_thomson**-1.5
#         ax1 = plt.subplot(122)
#         plt.loglog(x_arr,y_arr)
#         plot_data()

#     	ax2 = plt.subplot(122)
#     	chivals = calc_chisquare_txs(all_data_x,all_data_y,sigma_y,is_uplim,sed_x,sed_y)[3]   
#     	pltscatter(chivals)
    
#     plt.xlim([1e-6,1e16])
#     plt.ylim([1e20,1e70])
#     pdf.savefig()
#     plt.close()

#       strtxt = r"$B$=%.1f G"%d[G,i,:][0]+"\n"+r"$R$=1e%.1f cm"%d[G,i,:][1]+"\n"+r"$f_{\rm elec}$=1e%.1f"%d[G,i,:][2]+"\n"+r"$\gamma^{\rm max}_e$=1e%.1f"%d[G,i,:][3]+"\n"+r"$s_e$=%.1f"%d[G,i,:][4]

#       plt.text(5,31,strtxt,verticalalignment='bottom',horizontalalignment='center')
#
#       chi_info = calc_chisquare(data_x_seq_bl[0],data_y_seq_bl[0],data_y_seq_bl[0]*0+1,x_log_arr,y_log_arr, S0, S1, S2)
#       strtxtchi = "chi2 = %.1e"%(chi_info[0])
#
#       plt.text(13,39,strtxtchi,verticalalignment='bottom',horizontalalignment='center')
#
#       plt.text(-5,31,"%.1f %d %.1f %.1f"%(chi_info[0],chi_info[1],chi_info[2],chi_info[3]),verticalalignment='bottom',horizontalalignment='center')
#       plt.xlabel(r"$E_\gamma$ [eV]")
#       plt.ylabel(r"$E_\gamma dL_\gamma/dE_\gamma$ [erg/s]")
#
#        chi_result = calc_chisquare(all_data_x,all_data_y,all_data_y*0+1,x_log_arr,y_log_arr, S0, S1, S2)
#        chi = chi_result[0]
#        LEN = chi_result[1]
#        BSdatax0 = all_data_x
#        BSdatay0 = all_data_y
#        model_x_points  = chi_result[2]
#        model_y_points  = chi_result[3]
#        gradie_model  = chi_result[4]
#        gradie2_model = chi_result[5]
#        chi_arr       = chi_result[6]
#        chi2_grd      = chi_result[7]
#        chi2_grd2     = chi_result[8]
#
#        ax2=plt.subplot(211)
#        plt.plot(model_x_points, model_y_points,'b')
#        plt.scatter(BSdatax0, BSdatay0,color='r')
#
#        plt.ylabel("log Spectrum (erg/s)")
#
#        plt.xlim([-6,16])
#        plt.ylim([30,40])       
#
#        # ax4=plt.subplot(412)
#        # plt.plot(BSdatax0,np.gradient(BSdatay0) ,'r')
#        # plt.plot(BSdatax0,gradie_model ,'b')
#        # plt.ylabel("1st deriv.")
#        # plt.xlim([-6,16])
#
#        # ax5=plt.subplot(413)
#        # plt.ylabel("2nd deriv.")
#        # gradie2_data = np.gradient(np.gradient(BSdatay0))
#        # plt.plot(BSdatax0,s_golay(gradie2_data,5) ,'r')
#        # plt.plot(BSdatax0,s_golay(gradie2_model,5) ,'b')
#        # plt.xlim([-6,16])
#
#        ax6=plt.subplot(212)
#        plt.scatter(all_data_x, chi_arr , color='r')
#        plt.scatter(BSdatax0, chi2_grd , color='g')
#        #plt.plot(BSdatax0, chi2_grd2,color='b')
#        plt.xlim([-6,16])
#
#        plt.text(0.1,0.05,"CHI: %.1e"%(chi_result[9]),transform=ax2.transAxes)
#        plt.text(0.1,0.9,"CHI TOTAL: %.1e"%(chi),transform=ax2.transAxes)
#        # plt.text(0.1,0.05,"CHI: %.1e"%(np.nansum(chi2_grd)),transform=ax4.transAxes)
#        # plt.text(0.1,0.05,"CHI: %.1e"%(np.nansum(chi2_grd2)),transform=ax5.transAxes)
#
    # pdf.savefig()
    # plt.close()
#
#    print "Printed plots: plots"+info_prefix[:-3]+"_best_fits_%s.pdf"%G



## Make scatter plots

#G = all_gens[-2]
#
#with PdfPages("scatter"+info_prefix[:-3]+"_%s.pdf"%G) as pdf:
#
#    for gen in h5data.keys():
#        if gen[0]=='g':
#            par_lists    = [from_string(k) for k in h5data[gen].keys()]
#            number_list  = [h5data[gen][k][0] for k in h5data[gen].keys()]
#            chi_list  = [h5data['fits'][k][0] for k in h5data[gen].keys()]
#
#
#           B_arr      = np.array([list[0] for list in par_lists])
#            R_arr      = np.array([list[1] for list in par_lists])
#            f_e_arr    = np.array([list[2] for list in par_lists])
#            emax_e_arr = np.array([list[6] for list in par_lists])
#            alpha_arr  = np.array([list[7] for list in par_lists])
#
#            print B_arr[0], R_arr[0], len(number_list), number_list
#            ax1 = plt.subplot(211)
#            plt.scatter(B_arr,R_arr,alpha=.5)
#            ax1.axis([BOUND_LOW[0],BOUND_UP[0],BOUND_LOW[1],BOUND_UP[1]])
#            ax2 = plt.subplot(212)
#            plt.scatter(np.exp(emax_e_arr/10.),alpha_arr,alpha=0.5)
#
#            ax2.axis([np.exp(BOUND_LOW[3]/10.),np.exp(BOUND_UP[3]/10.),BOUND_LOW[4],BOUND_UP[4]])
#            ax1.set_xlabel(r"$B$")
#            ax1.set_ylabel(r"$R$")
#            ax2.set_xlabel(r"$Ee_max$")
#            ax2.set_ylabel(r"$alpha$")
#            ax2.set_xscale('log')
#            ax2.set_yscale('linear')
#            plt.subplots_adjust(hspace=0.4)
#            pdf.savefig()
#            plt.close()
#print "Scatter plot done."

#h5data.close()

