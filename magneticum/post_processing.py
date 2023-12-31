'''
Functions for post-processing data from simulations
'''
import numpy as np

def get_mean_profile(halo_data, field, scaling='r500c'):

	profile = []
	radial_bins = []

	for i, halo in enumerate(halo_data):
		F, r_bins = halo[field], halo['r_bins']

		if scaling == 'r500c':
			mscale, rscale = halo['m500c'], halo['r500c']
		
		if scaling == 'rvir':
			mscale, rscale = halo['mvir'], halo['rvir']

		rscale_index = np.abs(r_bins - rscale).argmin()  # Get radial bin closest to r500c
		Fscale = F[rscale_index]

		if Fscale!=0:
			profile.append(F/Fscale)
			radial_bins.append(r_bins/rscale)

	return np.mean(profile, axis=0), np.mean(radial_bins, axis=0)


def get_mean_profile_all_fields(halo_data, r_name='rvir', rescale=True):
	"""_summary_

	Parameters
	----------
	halo_data : _type_
		_description_
	r_name : str, optional
		_description_, by default 'rvir'
	rescale : bool, optional
		_description_, by default True

	Returns
	-------
	dict
		mean_profile, r bins, sigma ln(profile), sigma profile
	"""
	fields = halo_data[0]['fields'].keys()

	mean_profile_dict = {k: [[], [], [], []] for k in fields}

	for field in fields:
		all_profiles = np.array([halo['fields'][field][0] for halo in halo_data])
		all_r_bins = np.array([halo['fields'][field][1] for halo in halo_data])
		all_rvir = np.array([halo[r_name] for halo in halo_data])

		mean_prof, mean_r, std, lnstd = get_mean_profile_one_field(all_profiles, all_r_bins, all_rvir, rescale)
		mean_profile_dict[field][0] = mean_prof
		mean_profile_dict[field][1] = mean_r
		mean_profile_dict[field][2] = std
		mean_profile_dict[field][3] = lnstd

	return mean_profile_dict

def get_mean_profile_one_field(profiles, r_bins, rvir, rescale):
	rescaled_profiles = []
	rescaled_r_bins = []

	for i in range(len(profiles)):
		this_profile = profiles[i]
		this_r_bins = r_bins[i]
		this_rvir = rvir[i]

		rscale_index = np.abs(this_r_bins - this_rvir).argmin()
		this_rscale = this_r_bins[rscale_index]
		this_profile_scale = this_profile[rscale_index]
        
		if rescale is True:
			this_rescaled_profile = this_profile/this_profile_scale

		elif rescale is False:
			this_rescaled_profile = this_profile

		if not np.any(np.isnan(this_rescaled_profile)):
			rescaled_profiles.append(this_rescaled_profile)
			rescaled_r_bins.append(this_r_bins/this_rscale)

	rescaled_profiles = np.array(rescaled_profiles)
	rescaled_r_bins = np.array(rescaled_r_bins)

	return np.mean(rescaled_profiles, axis=0), np.mean(rescaled_r_bins, axis=0), np.std(rescaled_profiles, axis=0), np.std(np.log(rescaled_profiles), where=~np.isinf(np.log(rescaled_profiles)), axis=0)