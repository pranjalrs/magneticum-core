from numba import jit
from numba.core import types
from numba.typed import Dict
import numpy as np
import scipy.integrate
import scipy.interpolate

import astropy.units as u
import astropy.constants as const
import astropy.cosmology.units as cu

from interpolator import ProfileInterpolator

class HaloProfile():
	def __init__(self, **kwargs) -> None:
		## Cosmology
		self.omega_m = 0.272
		self.omega_b = 0.0456
		self.h = 0.704
		self.H0 = self.h * 100 *u.km/u.second/u.Mpc
		
		## Output quantity units
		self.units_rho = u.Msun/u.kpc**3 * cu.littleh**2
		self.units_Pe = u.keV/u.cm**3 * cu.littleh**2
		self.units_Temp = u.K

		## HMCode
		self.f_H = 0.76
		self.alpha = 0.8471
		self.M0 = 10**13.5937 * u.Msun/cu.littleh
		self.beta = 0.6
		## Halo concentration
		# epsilon1 = epsilon1_0 + epsilon1_1 * z
		self.eps1_0, self.eps1_1 = -0.1065, -0.1073
		self.eps2_0, self.eps2_1 = 0., 0.
		self.HMcode_rescale_A = 1#1.2989607249999999

		## Non-thermal pressure support
		self.alpha_nt = 0.0
		self.n_nt = 0.0

		## For irho = 0
		self.gamma = 1.177  # Polytropic index for bound gas profile
		self.gamma_T = 2  # Slope for KS temperature profile for low-mass halos
#		self.b = 0.0  # Slope for alpha = alpha * (M/M0)^b
#		self.d = 0.0  # Slope for gamma = gamma * (M/M0)^d

		## For irho = 1
		self.a = 0  # gamma= gamma*(M/M0)^a

		## For irho = 2
#		self.gamma_0 = 0.5
#		self.gamma_1 = -0.05
#		self.gamma_2 = 0. # For redshift scaling, disabled
#		self.beta_0 = 4.7
#		self.beta_1 = 0.05
#		self.beta_2 = 0.  # For redshift scaling, disabled
#		self.eta = 1.3

		##### Check these parameters before producing profiles ####
		## Choose Profile
		# 0 For HMCode profile
		# 1 for mass scaling of gamma
		# 2 for mass scaling + modified scaling similar to Gupta 2015
		self.irho = 0

		## Choose mass-concentration relation
		# 0 for Duffy 2008 (used in HMx)
		# 1 for (Magneticum) Ragagnin 2021
		# 2 for concentration as free parameter
		self.imass_conc = 0
		self.conc_param = 8
		self.rs = 0.5

		self.lognorm_rho = -2
		## Are you going to run MCMC?
		## This enables an interpolator for the profile
		self.use_interp = False  # set to True for computing profiles using interpolation
		self.zs = (0.,)  # Only used if use_interp is True
		self.mmin = 1e13 # These are used only for profile interpolation
		self.mmax = 1e16 # If no units supplied assumed in Msun/h

		##### Some class settings ####
		self.verbose = False
		self.interp_error_tol = 0.001 # %
		self.update_param(list(kwargs.keys()), list(kwargs.values()))

	def __str__(self):
		info = ""
		irho_dict = {0:'default HMCode', 1:'mass dependent gamma', 2:'e-GNFW'}
		cosmo_pars = ['omega_m', 'omega_b', 'h']
		global_halo_pars = ['f_H', 'alpha', 'HMcode_rescale_A', 'M0', 'beta',
							'eps1_0', 'eps1_1', 'eps2_0', 'eps2_1']
		prof_halo_pars = {0: ['gamma'],
						1: ['gamma', 'a'],
						2: ['gamma_0', 'gamma_1', 'gamma_2', 'beta_0', 
							'beta_1', 'beta_2', 'eta']}
		derived_halo_pars = ['mu_e', 'mu_p']
		misc_pars = ['use_interp', 'zs', 'mmin', 'mmax', 'interp_error_tol']

		info += 'Cosmology\n'
		info += '---------\n'
		for item in cosmo_pars:
			info += f'{item} = {self.__getattribute__(item)}\n'
		info += '\n'

		info += 'Global Halo Parameters\n'
		info += '----------------------\n'
		for item in global_halo_pars:
			info += f'{item} = {self.__getattribute__(item)}\n'
		info += '\n'

		info += 'Profile Halo Parameters\n'
		info += '-----------------------\n'
		info += f'irho={self.irho}: Using {irho_dict[self.irho]} profile\n'

		for item in prof_halo_pars[self.irho]:
			info += f'{item} = {self.__getattribute__(item)}\n'
		info += '\n'

		info += 'Derived Parameters\n'
		info += '----------------------\n'
		for item in derived_halo_pars:
			info += f'{item} = {self.__getattribute__(item)}\n'
		info += '\n'

		info += 'Misc. Parameters\n'
		info += '----------------\n'
		if self.use_interp is False: info += f'use_interp = False'
		else:
			for item in misc_pars:
				info += f'{item} = {self.__getattribute__(item)}\n'
		info += '\n'

		info += f'verbose = {self.verbose}'  

		return info

	def __repr__(self):
		return self.__str__()

	def update_param(self, names, values):
		"""Updates class attributes and derived parameters

		Parameters
		----------
		names : list
			Attribute names
		values : array
			Updated values
		"""
		for i in range(len(names)):
			if names[i]=='M0':
				self.__setattr__(names[i], values[i]*u.Msun/cu.littleh)

			elif names[i] not in self.__dict__.keys():
				if names[i]=='log10_M0':
					self.__setattr__('M0', 10**values[i]*u.Msun/cu.littleh)

				else:
					print(f'Unkown attribute `{names[i]}`! Ignoring..')
					continue

			else:
				self.__setattr__(names[i], values[i])
		self._update_derived_param()


	def get_rho_dm_profile_interpolated(self, M, z, r_bins=None):
		if '_rho_dm_prof_interpolator' not in self.__dict__:
			raise Exception('Attempting to use interpolator without initiliazing! \n Please set `use_interp` to True')

		if r_bins is None:
			r_bins = np.logspace(np.log10(0.1), np.log10(1), 200)

		this_z_rho_dm_interp = self._rho_dm_prof_interpolator[z]

		M = M.to(u.Msun/cu.littleh)

		this_rho_dm_profile = this_z_rho_dm_interp.eval(M, r_bins)
		this_rho_dm_profile *= self._rho_dm_prof_interpolator_units

		return_profiles = this_rho_dm_profile

		return return_profiles, r_bins

	def get_rho_dm_profile(self, M, z, r_bins=None):
		if r_bins is None:
			r_bins = np.logspace(np.log10(0.1), np.log10(1), 200)

# 		r_virial = self.get_rvirial(M, z)
		c_M = self.get_concentration(M, z)
# 		rs = r_virial/c_M

# 		fcdm = 1 - self.omega_b/self.omega_m
# 		Mcdm = M*fcdm
#		norm = 4*np.pi*rs**3 * (np.log(1 + c_M) - c_M/(1+c_M))

		num =  10**self.lognorm_rho
		denom = (r_bins*c_M) * (1 + r_bins*c_M)**2
		rho_cdm = num/denom * cu.littleh**2 * u.Msun/u.kpc**3

# 		x = r_bins
# 		Anfw = np.log(1+c_M) - c_M/(1+c_M)
# 		mean_rho = 10**self.lognorm_rho
# 		denom = 3 * Anfw * x * (1/c_M + x)**2
# 		rho_cdm = mean_rho/denom * u.Msun/u.kpc**3  #Mcdm/norm
		return rho_cdm, r_bins


	def get_Pe_profile_interpolated(self, M, z=0, r_bins=None, return_rho=False, return_Temp=False):
		"""_summary_

		Parameters
		----------
		M : np.array
			1D array of masses with appropriate astropy units
			Needs to be strictly increasing
		z : int, optional
			redshift, by default 0
		r_bins : np.array, optional
			1D array of radial bins in terms of r/Rvir, by default None

		Returns
		-------
		_type_
			_description_
		"""
		if '_Pe_prof_interpolator' not in self.__dict__:
			raise Exception('Attempting to use interpolator without initiliazing! \n Please set `use_interp` to True')

		if r_bins is None:
			r_bins = np.logspace(np.log10(0.1), np.log10(1), 200)

		# elif 


		this_z_Pe_interp = self._Pe_prof_interpolator[z]

		M = M.to(u.Msun/cu.littleh)

		this_Pe_profile = this_z_Pe_interp.eval(M, r_bins[0])
		this_Pe_profile *= self._Pe_prof_interpolator_units

		return_profiles = {}

		return_profiles['Pe'] = this_Pe_profile

		if return_rho is True:
			this_rho_profile = self._rho_prof_interpolator[z].eval(M, r_bins[1])
			this_rho_profile *= self._rho_prof_interpolator_units

			return_profiles['rho_gas'] = this_rho_profile

		if return_Temp is True:
			this_Temp_profile = self._Temp_prof_interpolator[z].eval(M, r_bins[2])
			this_Temp_profile *= self._Temp_prof_interpolator_units

			return_profiles['Temp'] = this_Temp_profile

		return return_profiles, r_bins


	def get_Pe_profile(self, M, z=0, r_bins=None, return_rho=False, return_Temp=False):
		"""Computes pressure profile for a given mass from 0.1-1Rvir

		Parameters
		----------
		M : float
			Virial Mass (in Msun/h)
		z : float, optional
			Redshift, by default 0
		Returns
		-------
		_type_
			r as a fraction of virial radius
		"""

		if r_bins is None:
			r_bins = np.logspace(np.log10(0.1), np.log10(1), 200)

		r_virial = self.get_rvirial(M, z)

		this_profile = self.get_Pe(M, r_bins*r_virial, z=z, return_rho=return_rho, return_Temp=return_Temp)
		
		return this_profile, r_bins


	def get_Pe(self, M, r, z, return_rho=False, return_Temp=False):
		"""Returns the electron pressure based on eq. 40
		in Mead et. al. 2020

		Arguments passed to the function must have associated astropy units
		Parameters
		----------
		M : float
			Halo Mass in Mass unit/h
		r : float
			radial distance
		a : float
			scale factor

		Returns
		-------
		float
			in keV/cm^3
		"""
		r_virial = self.get_rvirial(M, z)
		c_M = self.get_concentration(M, z)

		rho_bnd = self.get_rho_bnd(M, r, r_virial=r_virial, c_M=c_M, z=z)
		Temp_g = self.get_Temp_g(M, r, r_virial=r_virial, c_M=c_M, z=z)
		P_e = rho_bnd * const.k_B*Temp_g/const.m_p/self.mu_e
		factor_nt = self._get_factor_nonthermal(M, r, r_virial)

		P_e = P_e.to(self.units_Pe)*factor_nt

		return_profiles = (P_e,)
		if return_rho is True:
			return_profiles += (rho_bnd.to(self.units_rho),) # Keep in Msun/kpc^3; otherwise interpolation error is large

		if return_Temp is True:
			return_profiles += (Temp_g.to(self.units_Temp),)

		return return_profiles

	def _get_factor_nonthermal(self, M, r, r_virial):
		Rnt = self.alpha_nt * (r/r_virial)**self.n_nt	
		return np.maximum(0, 1-Rnt)


	def get_rho_bnd(self, M, r, r_virial, c_M, z):
		rho_bnd = self._get_rho_bnd_wrapper(M, r, r_virial=r_virial, c_M=c_M)

		norm = self.get_norm(self._get_rho_bnd_wrapper, M, r_virial=r_virial, c_M=c_M)

		return rho_bnd*self.get_f_bnd(M)*M / norm


	def _get_rho_bnd_wrapper(self, M, r, r_virial, c_M):
		'''Eq. 35
		'''

		r_s = r_virial/c_M

		params = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
		M0 = 1e13*u.Msun/cu.littleh

		if self.irho == 0:
			params['gamma'] = self.gamma

			return self._get_rho_bnd((r/r_s).decompose(), M, params, irho=0)

		if self.irho == 1:
			params['gamma'] = self.gamma
			params['a'] = self.a

			return self._get_rho_bnd((r/r_s).decompose(), M, params, irho=1)

		if self.irho == 2:
			params = {'gamma_0': self.gamma,
						'gamma_1': self.gamma_1,
						'gamma_2': self.gamma_2,
						'beta_0': self.beta_0,
						'beta_1': self.beta_1,
						'beta_2': self.beta_2,
						'eta': self.eta} 

			return self._get_rho_bnd((r/r_s).decompose(), (M/M0).decompose(), params, irho=2, c_M=c_M)

	@staticmethod
	@jit(nopython=True)
	def _get_rho_bnd(x, m, params, irho, c_M=None):
		if irho == 0:
			gamma = params['gamma']
			return np.power((np.log(1+x) / x ), 1/(gamma-1) )

		elif irho == 1:
			gamma = params['gamma']*m**params['a']  # Scaling gamma with mass
			return (np.log(1+x) / x )**(1/(gamma-1))

		# elif irho == 2:
		#     gamma_prime = params['gamma_0']*m**params['gamma_1']
		#     beta_prime = params['beta_0']*m**params['beta_1']

		#     num = c_M**gamma_prime * (1 + c_M**params['eta'])**((beta_prime-gamma_prime)/params['eta'])
		#     denom = (c_M*x)**gamma_prime * (1 + (c_M*x)**params['eta'])**((beta_prime-gamma_prime)/params['eta'])

		#     return num/denom

	def get_norm(self, profile, M, r_virial, c_M):
		r_unit = r_virial.unit
		integrand = lambda r: 4*np.pi*r**2*profile(M, r*u.Mpc/cu.littleh, r_virial=r_virial, c_M=c_M)
		rrange = np.linspace(1e-6, r_virial.value, 2000)  # Integration range 0, 1Rvir
		y = integrand(rrange)
		return scipy.integrate.simpson(y, rrange)*r_unit**3


	def get_Temp_g(self, M, r, r_virial, c_M, z):
		'''Gas temperature
		Eq. 38
		'''
		T_v = self._get_Temp_virial(M, r_virial, z=z)
		r_s = r_virial/c_M
		x = (r/r_s).decompose()

		f_r = np.log(1 + x)/x


		return T_v * (f_r)**(1/(self.gamma_T-1))

	def _get_Temp_virial(self, M, r_virial, z):
		'''Eq. 39
		Tv = G * m_p * mu_p /(a * rvirial) /(3/2 * kB) * M
		'''

		return self.alpha*(const.G*const.m_p*self.mu_p*(1+z)/r_virial/(3/2*const.k_B)*M).to(u.K)

	def get_delta_v(self, z):
		'''Eq. 22
		'''
		omega_at_z = self.omega_m*(1+z)**3 / (self.omega_m*(1+z)**3 + (1- self.omega_m))  # Omega0*(1+z)^3/E(z)^2
		return 1/omega_at_z * (18*np.pi**2 -82*(1-omega_at_z) - 39*(1-omega_at_z)**2)


	def get_rvirial(self, M, z):
		"""_summary_

		Parameters
		----------
		M : float
			Halo mass in Mass units/h
		z : float, optional
			redshift, by default 0.0

		Returns
		-------
		float
			virial radius in Mpc/h
		"""
		delta_v = self.get_delta_v(z)
		rho_crit = 2.7554e11 * u.Msun/u.Mpc**3 * cu.littleh**2  # In Msun * h**2 /Mpc**3
		rho_m = self.omega_m * rho_crit

		return ((M/ (4/3*np.pi * delta_v * rho_m))**(1/3)).to(u.Mpc/cu.littleh)#*self.h**(2/3)  #in Mpc


	def get_concentration(self, M, z):
		'''Eq. 33
		M should be in Msun/h
		'''
		## Concenetraion-Mass relation from Duffy et. al. 2008
		if self.imass_conc == 0:
			MSCALE = 2e12*u.Msun/cu.littleh
			c_M = 7.85 * (M/MSCALE)**(-0.081) * (1+z)**(-0.71)

		if self.imass_conc == 1:
			# Table 2 in arXiv:2011.05345
			MSCALE = 19.9e13*0.704*u.Msun/cu.littleh
			ap = 0.877
			a = 1/(1+z)
			c_M = np.exp(1.5)* (M/MSCALE)**(-0.04) * (a/ap)**(-0.52)
		
		if self.imass_conc == 2:
			return self.conc_param

		eps1 = self.eps1_0 + self.eps1_1*z
		eps2 = self.eps2_0 + self.eps2_1*z

		if eps1<= -1: 
			raise ValueError("eps1<-1 concentration for low mass halos is negative!")

		if eps2<= -1: 
			raise ValueError("eps2<-1 concentration for high mass halos is negative!")

		c_M_modified = c_M * (1 + eps1 + (eps2-eps1) * self.get_f_bnd(M)/ (self.omega_b/self.omega_m))
		c_M_modified = c_M_modified*self.HMcode_rescale_A

		return c_M_modified


	def get_f_bnd(self, M):
		"""Eq. 25
		"""
		return self.omega_b/self.omega_m * (M/self.M0)**self.beta/(1 + (M/self.M0)**self.beta)

	def _update_derived_param(self):
		# Derived params
		self.H0 = self.h * 100 *u.km/u.second/u.Mpc
		self.mu_e = 2/(1 + self.f_H)
		self.mu_p = 4/(3 + 5*self.f_H)

		# For faster evaluation pre-compute norm
		if self.use_interp is True:
			# self._init_norm_interp()
			self._init_prof_interpolator()

	def _init_prof_interpolator(self):
		self._rho_dm_prof_interpolator = {}
		self._Pe_prof_interpolator = {}
		self._rho_prof_interpolator = {}
		self._Temp_prof_interpolator = {}

		r_bins = np.logspace(np.log10(0.009), np.log10(1), 200)
		Mvirs = np.logspace(np.log10(self.mmin), np.log10(self.mmax), 50)*u.Msun/cu.littleh

		for z in self.zs:
			rho_dm_profs = []
			Pe_profs = []
			rho_profs = []
			Temp_profs = []
			for j, m in enumerate(Mvirs):
				this_rho_dm_prof, _ = self.get_rho_dm_profile(m, z, r_bins)
				rho_dm_profs.append(this_rho_dm_prof)

				temp, _ = self.get_Pe_profile(m, z, r_bins=r_bins, return_rho=True, return_Temp=True)
				this_Pe_prof, this_rho_prof, this_Temp_prof = temp[0], temp[1], temp[2]
				Pe_profs.append(this_Pe_prof.value)
				rho_profs.append(this_rho_prof.value)
				Temp_profs.append(this_Temp_prof.value)
			
			self._rho_dm_prof_interpolator[z] = ProfileInterpolator(Mvirs.value, r_bins, rho_dm_profs)
			self._Pe_prof_interpolator[z] = ProfileInterpolator(Mvirs.value, r_bins,Pe_profs)
			self._rho_prof_interpolator[z] = ProfileInterpolator(Mvirs.value, r_bins, rho_profs)
			self._Temp_prof_interpolator[z] = ProfileInterpolator(Mvirs.value, r_bins, Temp_profs)


	def _test_prof_interpolator(self, n=1000):
		if self.use_interp is False:
			print('`use_interp` set to False...Creating interpolator..')
			self._init_prof_interpolator()

		# Get r_bins
		r_bins = np.logspace(np.log10(0.5), np.log10(1), 100)

		# Now test at each redshift
		for z in self.zs:
			Ms = 10**np.random.uniform(np.log10(self.mmin), np.log10(self.mmax), n)*u.Msun/cu.littleh
			rho_dm_difference = 0.0
			Pe_difference, rho_difference, Temp_difference = 0., 0., 0.
				
			true_rho_dm_profiles = []
			true_Pe_profiles = []
			true_rho_profiles = []
			true_Temp_profiles = []

			for m in Ms:
				true_rho_dm_prof, _ = self.get_rho_dm_profile(m, z, r_bins=r_bins)
				true_rho_dm_profiles.append(true_rho_dm_prof.value)

				true_profs, _ = self.get_Pe_profile(m, z, r_bins=r_bins, return_rho=True, return_Temp=True)
				true_Pe_prof, true_rho_prof, true_Temp_prof = true_profs[0].value, true_profs[1].value, true_profs[2].value
				true_Pe_profiles.append(true_Pe_prof)
				true_rho_profiles.append(true_rho_prof)
				true_Temp_profiles.append(true_Temp_prof)

			true_rho_dm_profiles = np.concatenate(true_rho_dm_profiles)
			true_Pe_profiles = np.concatenate(true_Pe_profiles)
			true_rho_profiles = np.concatenate(true_rho_profiles)
			true_Temp_profiles = np.concatenate(true_Temp_profiles)

			interp_rho_dm_prof = np.concatenate(self._rho_dm_prof_interpolator[z].eval(Ms, r_bins))
			interp_Pe_prof = np.concatenate(self._Pe_prof_interpolator[z].eval(Ms, r_bins))
			interp_rho_prof = np.concatenate(self._rho_prof_interpolator[z].eval(Ms, r_bins))
			interp_Temp_prof = np.concatenate(self._Temp_prof_interpolator[z].eval(Ms, r_bins))

			mean_rho_dm_difference = np.sum(np.abs(interp_rho_dm_prof/true_rho_dm_profiles - 1))/n*100
			mean_Pe_difference = np.sum(np.abs(interp_Pe_prof/true_Pe_profiles - 1))/n*100
			mean_rho_difference = np.sum(np.abs(interp_rho_prof/true_rho_profiles - 1))/n*100
			mean_Temp_difference = np.sum(np.abs(interp_Temp_prof/true_Temp_profiles - 1))/n*100


			if mean_Pe_difference > self.interp_error_tol:
				raise Exception(f'Interpolation test failed for Pe profile with a mean frac. difference of {mean_Pe_difference:.4f}%. :(')

			if mean_rho_difference > self.interp_error_tol:
				raise Exception(f'Interpolation test failed for rho profile with a mean frac. difference of {mean_rho_difference:.4f}%. :(')

			if mean_Temp_difference > self.interp_error_tol:
				raise Exception(f'Interpolation test failed for Temperature profile with a mean frac. difference of {mean_Temp_difference:.4f}%. :(')

			if self.verbose is True:
				print(f'# of radial bins: {len(r_bins)}')
				print(f'Mean frac. difference between interpolated and true dark matter rho profile...')
				print(f'At z={z} is {mean_rho_dm_difference:.4f} %')

				print(f'Mean frac. difference between interpolated and true Pe profile...')
				print(f'At z={z} is {mean_Pe_difference:.4f} %')
				
				print(f'Mean frac. difference between interpolated and true rho profile...')
				print(f'At z={z} is {mean_rho_difference:.4f} %')
				
				print(f'Mean frac. difference between interpolated and true Temperature profile...')
				print(f'At z={z} is {mean_Temp_difference:.4f} %')

				print('Success!')