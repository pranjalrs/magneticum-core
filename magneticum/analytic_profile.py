from numba import jit
from numba.core import types
from numba.typed import Dict
import numpy as np
import scipy.integrate
import scipy.interpolate

import astropy.units as u
import astropy.constants as const
import astropy.cosmology.units as cu

class Profile():
    def __init__(self, **kwargs) -> None:
        ## HMCode
        self.f_H = 0.76
        self.gamma = 1.177  # Polytropic index for bound gas profile
        self.alpha = 0.8471
        self.M0 = 10**13.5937 * u.Msun/cu.littleh
        self.beta = 0.6
        self.HMCode_rescale_A = 1.2989607249999999

        ## For irho = 1
        self.a = 0  # gamma= gamma*(M/M0)^a

        ## For irho = 2
        self.gamma_0 = 0.5
        self.gamma_1 = -0.05
        self.gamma_2 = 0. # For redshift scaling, disabled
        self.beta_0 = 4.7
        self.beta_1 = 0.05
        self.beta_2 = 0.  # For redshift scaling, disabled
        self.eta = 1.3


        ## Halo concentration
        # epsilon1 = epsilon1_0 + epsilon1_1 * z
        self.eps1_0, self.eps1_1 = -0.1065, -0.1073
        self.eps2_0, self.eps2_1 = 0., 0.



        ## Cosmology
        self.omega_m = 0.272
        self.omega_b = 0.0456
        self.h = 0.704

        ##### Check these parameters before producing profiles ####
        ## Choose Profile
        # 0 For HMCode profile
        # 1 for mass scaling of gamma
        # 2 for mass scaling + modified scaling similar to Gupta 2015
        self.irho = 0
        ## Are you going to run MCMC?
        ## This enables an interpolator for the profile norm
        self.ifit = False  # set to True if doing fits
        self.zs = (0.,)  # Only used if ifit is True

        self.update_param(list(kwargs.keys()), list(kwargs.values()))

    def _update_derived_param(self):
        # Derived params
        self.H0 = self.h * 100 *u.km/u.second/u.Mpc
        self.mu_e = 2/(1 + self.f_H)
        self.mu_p = 4/(3 + 5*self.f_H)

        # For faster evaluation pre-compute norm
        if self.ifit is True:
            Mvirs = np.logspace(10, 16, 100)*u.Msun/cu.littleh
            self._norm_interpolate = {}

            norms = np.zeros((len(self.zs), len(Mvirs)))
            for i in range(len(self.zs)):
                norms = []
                for j, m in enumerate(Mvirs):
                    rvir = self.get_rvirial(m, z=self.zs[i])
                    c_M = self.get_concentration(m, z=self.zs[i])
                    this_norm = self.get_norm(self._get_rho_bnd_wrapper, m, rvir, c_M=c_M)
                    norms.append(this_norm.value)

                self._norm_interpolate[self.zs[i]] = scipy.interpolate.CubicSpline(Mvirs, norms, extrapolate=False)
            self._norm_interpolate_units = this_norm.unit

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
            if names[i] not in self.__dict__.keys():
                print(f'Unkown Attribute {names[i]} !')
                return

            if names[i]=='log10_M0':
                self.__setattr__(names[i], 10**values[i]*u.Msun/cu.littleh)

            elif names[i]=='M0':
                self.__setattr__(names[i], values[i]*u.Msun/cu.littleh)

            else:
                self.__setattr__(names[i], values[i])
        self._update_derived_param()

    
    def get_Pe_profile(self, M, z=0):
        """Computes pressure profile for a given mass from 0.1-1Rvir

        Parameters
        ----------
        M : float
            Virial Mass (in Msun/h)
        a : float, optional
            scale factor, by default 1

        Returns
        -------
        _type_
            r as a fraction of virial radius
        """
        rvir = self.get_rvirial(M, z)
        r_bins = np.logspace(np.log10(0.1), np.log10(1), 200)*rvir
        this_profile = self.get_Pe(M, r_bins, z=z)
        return this_profile, (r_bins/rvir).value


    def get_Pe(self, M, r, z):
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
        
        P_e = P_e.to(u.keV/u.cm**3, cu.with_H0(self.H0))

        return P_e

    def get_rho_bnd(self, M, r, r_virial, c_M, z):
        rho_bnd = self._get_rho_bnd_wrapper(M, r, r_virial=r_virial, c_M=c_M)

        if self.ifit is True:        
            norm = self.get_norm(self._get_rho_bnd_wrapper, M, r_virial=r_virial, c_M=c_M)
        
        elif self.ifit is False:
            z_index = np.where(z==self.zs)[0]
            norm = self._norm_interpolate[z_index](M)*self._norm_interpolate_units

        return rho_bnd*self.get_f_bnd(M)*M / norm


    def _get_rho_bnd_wrapper(self, M, r, r_virial, c_M):
        '''Eq. 35
        '''

        r_s = r_virial/c_M

        params = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
        M0 = 5e14*u.Msun/cu.littleh

        if self.irho == 0:
            params['gamma'] = self.gamma
            return self._get_rho_bnd((r/r_s).decompose(), M, params, irho=0)

        if self.irho == 1:
            params = {'gamma': self.gamma,
                        'a': self.a}
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

        # elif irho == 1:
        #     gamma = params['gamma']*m**params['a']  # Scaling gamma with mass
        #     return (np.log(1+x) / x )**(1/(gamma-1))

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

        if self.irho == 0 or self.irho == 1:
            f_r = np.log(1 + x)/x

        elif self.irho == 2:
            f_r = 1

        return T_v * f_r    


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
        MSCALE = 2e12*u.Msun/cu.littleh
        c_M = 7.85 * (M/MSCALE)**(-0.081) * (1+z)**(-0.71)

        eps1 = self.eps1_0 + self.eps1_1*z
        eps2 = self.eps2_0 + self.eps2_1*z

        c_M_modified = c_M * (1 + eps1 + (eps2-eps1) * self.get_f_bnd(M)/ (self.omega_b/self.omega_m))
        c_M_modified = c_M_modified*self.HMCode_rescale_A

        return c_M_modified


    def get_f_bnd(self, M):
        """Eq. 25
        """
        return self.omega_b/self.omega_m * (M/self.M0)**self.beta/(1 + (M/self.M0)**self.beta)
