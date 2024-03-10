class ProfileContainer():
    """
    A class representing a container for profile data.

    Attributes:
        mvir (float): The virial mass.
        rvir (float): The virial radius.
        profile (list): The profile data.
        profile_rescale (float): The rescaled profile data.
        rbins (list): The radial bins.
        sigma_prof (float): The profile standard deviation.
        sigma_lnprof (float): The natural logarithm of the profile standard deviation.
    """

    def __init__(self, mvir, rvir, profile, profile_rescale, rbins, sigma_prof, sigma_lnprof):
        self.mvir = mvir
        self.rvir = rvir
        self.profile = profile
        self.profile_rescale = profile_rescale
        self.rbins = rbins
        self.sigma_prof = sigma_prof
        self.sigma_lnprof = sigma_lnprof