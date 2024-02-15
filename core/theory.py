import numpy as np

import camb
from classy import Class
import Pk_library as PKL
import hmcode  # This is the full Python implementation of the Fortran HMCode
import pyhmcode  # This the Python wrapper for the Fortran HMCode

def get_CLASS_Pk(k_sim, input_dict=None, binned=False, z=0.0):
	'''Returns Pk for WMAP7 cosmology, optionally a dictionary of cosmo
	parameters can also be passed
	'''
	zmax = 0.1

	if z>0.1:
		zmax = z + 0.2

	params = {
		'output': 'mPk', #which quantities we want CLASS to compute
		'H0':0.704*100, #cosmology paramater
		'z_max_pk' : str(zmax),
		'non linear' : 'halofit',#option for computing non-linear power spectum
		'P_k_max_1/Mpc':30.0,
		'sigma8': 0.809, #cosmology paramater
		'n_s': 0.963 , #cosmology paramater
		'Omega_b': 0.0456, #cosmology paramater
		'Omega_cdm': 0.272-0.0456 #cosmology paramater
	}
	
	if input_dict is not None:
		params['Omega_cdm'] = input_dict['Om0']-input_dict['Ob0']
		params['Omega_b'] = input_dict['Ob0']
		params['sigma8'] = input_dict['sigma8']
		params['H0'] = input_dict['H0']

	#initialize Class instance and set parameters
	cosmo = Class()
	cosmo.set(params)
	cosmo.compute()
	#now create second Class instance with parameters changed to linear power spectrum model
	params['non linear'] = 'none'
	cosmo_lin = Class()
	cosmo_lin.set(params)
	cosmo_lin.compute()
	# get P(k) at redhsift z=0
	Nk = 2000
	Plin = np.zeros(Nk)
	Pnl = np.zeros(Nk)
	h = cosmo.h() # get reduced Hubble for conversions to 1/Mpc
	kvec = np.exp(np.linspace(np.log(1.e-3),np.log(10.),Nk))
	for k in range(Nk):
		Pnl[k] = cosmo.pk(kvec[k]*h, z)*h**3 #using method .pk(k,z), convert to Mpc/h units
		Plin[k] = cosmo_lin.pk(kvec[k]*h, z)*h**3

	if binned is True:
		# Binning
		bin_edges = np.concatenate((k_sim, [k_sim[-1]+(k_sim[-1]-k_sim[-2])]))
		bins = np.digitize(kvec, bin_edges)-1
		P_theory_binned = np.zeros_like(k_sim)
		for i in range(len(k_sim)):
			P_theory_binned[i] = np.mean(Pnl[bins==i])

		return P_theory_binned, k_sim
		
		return Pk, k

	return Pnl, kvec

def get_pyHMCode_Pk(camb_pars=None, z=[0.], fields=None, return_halo_terms=False, **kwargs):
	'''
	Calculate the nonlinear matter power spectrum using the HMcode.

	Parameters:
	- camb_pars (optional): CAMB parameters used to compute the linear matter power spectrum.
	- z (list): Redshifts at which to compute the power spectrum.
	- fields (optional): List of fields to include in the calculation.
	- return_halo_terms (bool): Whether to return the halo terms in addition to the total power spectrum.
	- **kwargs: Additional keyword arguments to customize the HMcode calculation.

	Returns:
	- Pk_hm (array): The total nonlinear matter power spectrum.
	- ks (array): The wavenumbers corresponding to the power spectrum.

	If return_halo_terms is True, the function also returns:
	- Pk_hm_1halo (array): The 1-halo term of the power spectrum.
	- Pk_hm_2halo (array): The 2-halo term of the power spectrum.
	'''
	# Compute CAMB results
	r = camb.get_results(camb_pars, z)
	ks, zs, Pk_lin = r.get_linear_matter_power_spectrum(nonlinear=False)

	# Need these Omegas
	# Note that the calculation of Omega_v is peculiar, but ensures flatness in HMcode
	# (very) small differences between CAMB and HMcode otherwise
	omv = r.omega_de+r.get_Omega("photon")+r.get_Omega("neutrino") 
	omm = camb_pars.omegam

	# Setup HMcode internal cosmology
	c = pyhmcode.Cosmology()

	# Set HMcode internal cosmological parameters
	c.om_m = omm
	c.om_b = camb_pars.omegab
	c.om_v = omv
	c.h = camb_pars.h
	c.ns = camb_pars.InitPower.ns
	c.sig8 = r.get_sigma8_0()

	if r.num_nu_massive == 0.0:
		c.m_nu = 0.0
	else:
		raise NotImplementedError('Setting massive neutrinos is not implemented yet')

	# Set the linear power spectrum for HMcode
	c.set_linear_power_spectrum(ks, zs, Pk_lin)

	mode = pyhmcode.HMx2020_matter_pressure_w_temp_scaling
	hmod = pyhmcode.Halomodel(mode, verbose=False)

	for key in kwargs:
		hmod.__setattr__(key, kwargs[key])

	if fields is None:
		fields = [pyhmcode.field_electron_pressure]

	if return_halo_terms is False:
		Pk_hm = pyhmcode.calculate_nonlinear_power_spectrum(c, hmod, fields, verbose=False, return_halo_terms=False)
		return Pk_hm, ks

	elif return_halo_terms is True:
		Pk_hm, Pk_hm_1halo, Pk_hm_2halo = pyhmcode.calculate_nonlinear_power_spectrum(c, hmod, fields, verbose=False, return_halo_terms=True)
		return Pk_hm, Pk_hm_1halo, Pk_hm_2halo, ks


def get_suppresion_hmcode(input_cosmo=None, zs=[0.], T_AGNs=None):
	'''Matter power supression based on hmcode-the full Python
	implementation of the Fortran HMCode.
	'''
	zs = np.array(zs)

	# Wavenumbers [h/Mpc]
	kmin, kmax = 1e-3, 3e1
	nk = 128
	k = np.logspace(np.log10(kmin), np.log10(kmax), nk)

	# AGN-feedback temperature [K]
	if T_AGNs is None:
		T_AGNs = np.power(10, np.array([7.6, 7.8, 8.0, 8.3]))

	# Redshifts

	camb_results  = build_CAMB_cosmology(input_cosmo=input_cosmo, zs=zs)

	Pk_lin_interp = camb_results.get_matter_power_interpolator(nonlinear=False).P
	Pk_nonlin_interp = camb_results.get_matter_power_interpolator(nonlinear=True).P

	# Arrays for CAMB non-linear spectrum
	Pk_CAMB = np.zeros((len(zs), len(k)))
	for iz, z in enumerate(zs):
		Pk_CAMB[iz, :] = Pk_nonlin_interp(z, k)

	Rk_feedback = []
	for T_AGN in T_AGNs:
		Pk_feedback = hmcode.power(k, zs, camb_results, T_AGN=T_AGN, verbose=False)
		Pk_gravity = hmcode.power(k, zs, camb_results, T_AGN=None)
		Rk = Pk_feedback/Pk_gravity
		Rk_feedback.append(Rk)

	return Rk_feedback, k

# Halo masses [Msun/h] (for halo model only)
Mmin, Mmax = 1e0, 1e18
nM = 256
M = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)

def build_CAMB_cosmology(input_cosmo=None, zs=[0.]):
	'''
	Builds a cosmology using the CAMB library.

	Parameters:
		input_cosmo (dict): Dictionary containing input cosmological parameters.
							If None, default cosmological parameters are used.
							Default is None.
		zs (list): List of redshifts at which to calculate the linear matter power spectrum.
				   Default is [0.].

	Returns:
		results: The results object obtained from running CAMB with the specified cosmological parameters.
	'''
	# default cosmology is WMAP7
	Omega_b = 0.0456
	Omega_c = 0.272 - Omega_b
	Omega_k = 0.0
	h = 0.704
	ns = 0.963
	sigma_8 = 0.809
	w0 = -1.
	wa = 0.
	m_nu = 0.
	set_sigma8 = True
	As = 2e-9

	if input_cosmo is not None:
		Omega_b = input_cosmo['Ob0']
		Omega_c = input_cosmo['Om0'] - input_cosmo['Ob0']
		h = input_cosmo['H0']/100
		sigma_8 = input_cosmo['sigma8']

	# CAMB
	kmax_CAMB = 200.

	# Sets cosmological parameters in camb to calculate the linear power spectrum
	pars = camb.CAMBparams(WantTransfer=True, WantCls=False, Want_CMB_lensing=False, 
						DoLensing=False, NonLinearModel=camb.nonlinear.Halofit(halofit_version='mead2020'))
	wb, wc = Omega_b*h**2, Omega_c*h**2

	# This function sets standard and helium set using BBN consistency
	pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
	pars.set_dark_energy(w=w0, wa=wa, dark_energy_model='ppf')
	pars.InitPower.set_params(As=As, ns=ns, r=0.)
	pars.set_matter_power(redshifts=zs, kmax=kmax_CAMB) # Setup the linear matter power spectrum
	Omega_m = pars.omegam # Extract the matter density

	# Scale 'As' to be correct for the desired 'sigma_8' value if necessary
	if set_sigma8:
		results = camb.get_results(pars)
		sigma_8_init = results.get_sigma8_0()
		print('Running CAMB')
		print('Initial sigma_8:', sigma_8_init)
		print('Desired sigma_8:', sigma_8)
		scaling = (sigma_8/sigma_8_init)**2
		As *= scaling
		pars.InitPower.set_params(As=As, ns=ns, r=0.)

	sigma_8 = results.get_sigma8_0()
	print('Final sigma_8:', sigma_8)

	# Run
	results = camb.get_results(pars)

	return results


def get_Pk_Pylians(cube, box_size, calc_delta, MAS, savefile=None):
	if calc_delta is False:
		if isinstance(cube, str):
			delta = np.load(cube) # this is the 3D data

		else:
			delta = cube
		Pk = PKL.Pk(delta, box_size, 0, MAS, True)

		return Pk.k3D, Pk.Pk[:,0]