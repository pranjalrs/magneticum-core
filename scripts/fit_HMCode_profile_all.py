'''Same as fit_HMCode_profile.py but fits all profiles instead of 
the mean profile in each mass bin
'''
import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from schwimmbad import MPIPool

import astropy.units as u
import astropy.cosmology.units as cu
import getdist
from getdist import plots
import glob
import emcee

import sys
sys.path.append('../src/')

from analytic_profile import Profile
import post_processing

def likelihood(x, mass_list, z=0):
    for i in range(len(fit_par)):
        lb, ub = bounds[fit_par[i]]
        if x[i]<lb or x[i]>ub:
            return -np.inf

    model_pars = x[:-num_rbins]
    fitter.update_param(fit_par[:-num_rbins], model_pars)

    mvir = mass_list*u.Msun/cu.littleh
    ## Get profile for each halo
    Pe, r = fitter.get_Pe_profile_interpolated(mvir, z=z)
    
    chi2 = 0 

    for i in range(len(Pe)):
        interpolator = interp1d(r, Pe[i].value)
        Pe_theory = interpolator(r_sim[i])  # Get theory Pe at measured r values
        
        num = np.log(Pe_sim[i] / Pe_theory)**2
        denom = sigmalnP_sim[i]**2 + x[len(model_pars):]**2
        chi2 += -0.5*np.sum(num/denom)  # Sum over radial bins
    
    return -chi2

bounds = {'f_H': [0.65, 0.85],
        'gamma': [1.1, 5],
        'alpha': [0.1, 1.5],
        'log10_M0': [10, 17],
        'M0': [1e10, 1e17],
        'beta': [0.4, 0.8],
        'eps1_0': [-.8, .8],
        'eps2_0': [-.8, .8]}

fid_val = {'f_H': 0.75,
        'gamma': 1.2,
        'alpha': 0.5,
        'log10_M0': 14,
        'M0': 1e14,
        'beta': 0.6,
        'eps1_0': 0.2,
        'eps2_0': -0.1}

std_dev = {'f_H': 0.2,
        'gamma': 0.2,
        'alpha': 0.2,
        'log10_M0': 2,
        'M0': 1e12,
        'beta': 0.2,
        'eps1_0': 0.02,
        'eps2_0': 0.02}

#####-------------- Parse Args --------------#####

parser = argparse.ArgumentParser()
parser.add_argument('--test', type=bool, default=False)
args = parser.parse_args()

test = args.test

#####-------------- Load Data --------------#####
files = glob.glob('../output_data/Profiles_median/Box1a/Pe_Pe_Mead_Temp_matter_cdm_gas_z=0.00*')


Pe_sim= []
r_sim = []
sigmalnP_sim = []
Mvir_sim = []

for f in files:
    this_prof_data = joblib.load(f)
    
    for d in this_prof_data:
        this_prof_r = d['fields']['Pe_Mead'][1]/d['rvir']
        mask = this_prof_r<1
        this_prof_r = this_prof_r[mask]
        this_prof_field = d['fields']['Pe_Mead'][0][mask]
        this_sigma_lnP = d['fields']['Pe'][3][mask]
        
        Pe_sim.append(this_prof_field)
        r_sim.append(this_prof_r)
        sigmalnP_sim.append(this_sigma_lnP)
    
    Mvir_sim.append(this_prof_data['mvir'])

# Now we need to sort halos in order of increasing mass
# Since this is what the scipy interpolator expects
Mvir_sim = np.concatenate(Mvir_sim, dtype='float32')
sorting_indices = np.argsort(Mvir_sim)

Pe_sim = np.array(Pe_sim, dtype='float32')[sorting_indices]
r_sim = np.array(r_sim, dtype='float32')[sorting_indices]
sigmalnP_sim = np.array(sigmalnP_sim, dtype='float32')[sorting_indices]
Mvir_sim = Mvir_sim[sorting_indices]

num_rbins = r_sim.shape[1]
sigma_intr = np.zeros(num_rbins)

#####-------------- Adding intr. scatter to fit parameters --------------#####
fit_par = ['gamma', 'alpha', 'log10_M0', 'beta', 'eps1_0', 'eps2_0']
par_latex_names = ['\Gamma', '\\alpha', '\log_{10}M_0', '\\beta', '\epsilon_1', '\epsilon_2']

fit_par += [f'sigma_intr_{i+1}' for i in range(num_rbins)]
par_latex_names += ['\sigma^{\mathrm{intr}}_'+f'{{i+1}}' for i in range(num_rbins)]

for i in range(num_rbins):
    bounds[f'sigma_intr_{i+1}'] =  [0.03, 0.8]
    fid_val[f'sigma_intr_{i+1}'] =  0.3
    std_dev[f'sigma_intr_{i+1}'] =  0.1

#####-------------- Prepare for MCMC --------------#####
fitter = Profile(use_interp=True)

starting_point = [fid_val[k] for k in fit_par]
std = [std_dev[k] for k in fit_par]

ndim = len(fit_par)
nwalkers= 2 * ndim
nsteps = 3000

p0_walkers = emcee.utils.sample_ball(starting_point, std, size=nwalkers)

for i, key in enumerate(fit_par):
    low_lim, up_lim = bounds[fit_par[i]]

    for walker in range(nwalkers):
        while p0_walkers[walker, i] < low_lim or p0_walkers[walker, i] > up_lim:
            p0_walkers[walker, i] = np.random.rand()*std[i] + starting_point[i]


#####-------------- RUN MCMC --------------#####
if test is False:
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        print('Running MCMC with MPI...')
        sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, pool=pool, args=[Mvir_sim])
        sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

else:
    print('Running MCMC...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, likelihood, args=[Mvir_sim])
    sampler.run_mcmc(p0_walkers, nsteps=nsteps, progress=True)

#####-------------- Plot and Save --------------#####
save_path = 'emcee_output/fit_Pe_all/'
walkers = sampler.get_chain()
chain = sampler.get_chain(discard=int(0.8*nsteps), flat=True)

log_prob_samples = sampler.get_log_prob(discard=int(0.8*nsteps), flat=True)

all_samples = np.concatenate((chain, log_prob_samples[:, None]), axis=1)
np.savetxt(f'{save_path}/all_samples.txt', chain)



fig, ax = plt.subplots(len(fit_par), 1, figsize=(10, 1.5*len(fit_par)))
ax = ax.flatten()

for i in range(len(fit_par)):
    ax[i].plot(walkers[:, :, i])
    ax[i].set_ylabel(f'${par_latex_names[i]}$')
    ax[i].set_xlabel('Step #')

plt.savefig(f'{save_path}/trace_plot.pdf')

plt.figure()

gd_samples = getdist.MCSamples(samples=chain, names=fit_par, labels=par_latex_names)
g = plots.get_subplot_plotter()
g.triangle_plot(gd_samples, axis_marker_lw=2, marker_args={'lw':2}, line_args={'lw':1}, title_limit=2)
plt.savefig(f'{save_path}/triangle_plot.pdf')

########## Compare best-fit profiles ##########
c = ['r', 'b', 'g', 'k']

bins = [13.5, 14, 14.5, 15]
# Fiducial HMCode profiles
fitter.update_param(fit_par, gd_samples.getMeans())

# Randomly pick 5 profiles
nhalo = 5
inds = np.random.choice(Mvir_sim, nhalo, replace=False)
Pe_bestfit, r_bestfit = fitter.get_Pe_profile_interpolated(Mvir_sim[inds]*u.Msun/cu.littleh, z=0)

plt.figure(figsize=(7, 5))

lines = [None]*(len(nhalo)+1)

for i, j in zip(range(nhalo), inds):
    bin_label = f'{(bins[i]):.1f}$\leq\log$M$_{{vir}}<${(bins[i+1]):.1f}'
    plt.plot(r_bestfit[i], Pe_bestfit[i].value, c=c[i], ls='--')
    lines[i] = plt.errorbar(r_sim[j], Pe_sim[j], yerr=sigmaP_sim[j], c=c[i], label=f'log10 M = {np.log10(Mvir_sim[j]):.3f}')
    

plt.xscale('log')
plt.yscale('log')

lines[i+1], = plt.loglog([], [], '--', c='k', label='Best Fit')

legend1 = plt.legend(handles=lines, title='Box1a', fontsize=12, frameon=False)
# legend2 = plt.legend(handles=line_analytic, labels=['Best Fit'], fontsize=12, frameon=False, loc='lower center')

plt.gca().add_artist(legend1)
# plt.gca().add_artist(legend1)

plt.xlabel('$r/R_{\mathrm{vir}}$')
plt.ylabel('$P_e$ [keV/cm$^3$]');

plt.ylim([2e-6, 1.2e-2])
plt.savefig(f'{save_path}/best_fit.pdf')