import argparse
import numpy as np
import os
from tqdm import tqdm

import MAS_library as MASL
import Pk_library as PKL
import g3read

parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--snap_dir', default='144', type=str)
parser.add_argument('--grid', default=1024, type=int)
parser.add_argument('--MAS', type=str)
parser.add_argument('--fold', default=1, type=int)
parser.add_argument('--threads', default=1, type=int)


def get_mass_cube(delta, ptype, fold):
	pos = []
	mass = []

	shot_noise_num = 0
	shot_noise_denom = 0

	for i in range(f.header.num_files):
		this_file = snap_path + str(i)

		for this in ptype:
			pos = np.array(g3read.read_new(this_file, ['POS '], [this])[this]['POS ']*1e-3)/fold

			if this in [0, 1, 4, 2]:
				mass = np.array(g3read.read_new(this_file, ['MASS'], [this])[this]['MASS']*1e10)

			elif this == 5:
				try:
					mass = np.array(g3read.read_new(this_file, ['BHMA'], [this])[this]['BHMA']*1e10)
				except FileNotFoundError:
					print(f'Block BHMA not found in file {i}')

			shot_noise_num += np.sum(mass)
			shot_noise_denom += np.sum(mass**2)

			pos, mass = pos.astype('float32'), mass.astype('float32')
			MASL.MA(pos, delta, BoxSize, MAS, W=mass, verbose=verbose)

	Neff = shot_noise_num**2/shot_noise_denom  # Effective number of particles
	return Neff



if __name__== '__main__':
	args = parser.parse_args()

	sim_box = args.box
	sim_name = args.sim
	snap_dir = args.snap_dir
	grid = args.grid
	threads = args.threads

	current_directory = os.getcwd()

	if 'pranjalrs' in current_directory:
		snap_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}.'

	if 'di75sic' in current_directory:
		snap_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}.'

	print(f'Path to snap shot files is: {snap_path}')

	f = g3read.GadgetFile(snap_path+'0')

	# density field parameters
	fold = args.fold
	grid    = grid   #the 3D field will have grid x grid x grid voxels
	BoxSize = f.header.BoxSize/1e3/fold #Mpc/h ; size of box
	z = f.header.redshift
	MAS     = args.MAS  #mass-assigment scheme
	verbose = True   #print information on progress
	threads = threads
	axis = 0

	# Initialise the density field
	delta = np.zeros((grid,grid,grid), dtype=np.float32)

	if 'dm' not in sim_name:
		get_mass_cube(delta, [0, 1, 4, 5], fold)

	else:
		if 'dm_hr' in sim_name:
			Neff = get_mass_cube(delta, [1], fold)

		else:
			Neff = get_mass_cube(delta, [1, 2], fold)

	delta /= np.mean(delta, dtype=np.float64)
	delta -= 1.0

	shot_noise = BoxSize**3/Neff
	Pk = PKL.Pk(delta, BoxSize, axis, MAS, verbose)

	header = f'''Power spectrum of the matter density field in the Magneticum simulation {sim_name} at redshift z={z:.2f}.
	The columns are: k [h/Mpc], Pk [Mpc^3/h^3](sim_name, z)
	Shot noise (not subtracted): {shot_noise}
	Fold: {fold} (Box size and particle positions are scaled by 1/fold)
	'''

	# Save Pk
	save_dir = f'../../magneticum-data/data/Pylians/Pk_matter/{sim_box}/'
	if not os.path.exists(save_dir): os.makedirs(save_dir)
	np.savetxt(f'{save_dir}/Pk_{sim_name}_z={z:.2f}_R{grid}_fold_{fold}.txt',
               np.column_stack((Pk.k3D, Pk.Pk[:, 0])), delimiter='\t', header=header)

	# Save delta
	cube_save_dir = f'../../cube/{sim_box}/delta_matter/'
	if not os.path.exists(cube_save_dir): os.makedirs(cube_save_dir)
	np.save(f'{cube_save_dir}/delta_matter_{sim_name}_z={z:.2f}_R{grid}_fold_{fold}.npy', delta, allow_pickle=False)