import argparse
import numpy as np
import os

import astropy.cosmology.units as cu
import astropy.units as u
import MAS_library as MASL
import Pk_library as PKL
import g3read

from dawn import Pk_tools


parser = argparse.ArgumentParser()
parser.add_argument('--box', default='Box1a', type=str)
parser.add_argument('--sim', default='mr_bao', type=str)
parser.add_argument('--snap', default='144', type=str)
parser.add_argument('--grid', default=1024, type=int)
parser.add_argument('--MAS', default='CIC', type=str)
parser.add_argument('--threads', default=1, type=int)
parser.add_argument('--fold', default=1, type=int)
parser.add_argument('--field', type=str)

def get_Pe_cube(Pe_cube, norm, little_h, fold, ptype=0):
	for i in range(f.header.num_files):
		this_snap = g3read.GadgetFile(snap_path + str(i))
		Pe = Pk_tools.get_field(ptype, this_snap, 'Pe', little_h=little_h)
		pos = this_snap.read_new('POS ', ptype)*1e-3/fold
		m_over_rho = np.array(this_snap.read_new('MASS', ptype))/np.array(this_snap.read_new('RHO ', ptype))

		pos, Pe = pos.astype('float32'), Pe.astype('float32')
		m_over_rho = m_over_rho.astype('float32')
		MASL.MA(pos, Pe_cube, BoxSize, MAS, W=Pe*m_over_rho, verbose=verbose)
		MASL.MA(pos, norm, BoxSize, MAS, W=m_over_rho, verbose=verbose)

		print(i)


def get_Pe_Mead_cube(Pe_cube, little_h, fold, ptype=0):
	"""
	Calculate the pressure field in a cube using the Pe_Mead method.

	Parameters:
	Pe_cube (numpy.ndarray): The cube to store the pressure field.
	little_h (float): The dimensionless Hubble parameter.
	fold (int): The folding factor for the positions.
	ptype (int, optional): The particle type to consider. Default is 0.

	Returns:
	None

	Notes:
	- The function also calculates shot noise and matter pressure shot noise.
	Shot Noise matter-pressure = L^3/(grid^3)^2 * sum_i (P_i * m_i)/ mean_mass
	Shot Noise pressure = L^3/(grid^3)^2 * sum_i (P_i^2)
	"""
	cell_volume = (BoxSize*u.Mpc/grid)**3

	shot_noise_pressure = 0
	short_noise_matter_pressure = 0
	total_mass = 0

	for i in range(f.header.num_files):
		this_snap = g3read.GadgetFile(snap_path + str(i))
		Pe = Pk_tools.get_field(ptype, this_snap, 'Pe_Mead', little_h, cell_volume)
		mass = np.array(g3read.read_new(this_snap, ['MASS'], [ptype])[ptype]['MASS']*1e10)
		pos = this_snap.read_new('POS ', ptype)*1e-3/fold


		pos, Pe = pos.astype('float32'), Pe.astype('float32')
		MASL.MA(pos, Pe_cube, BoxSize, MAS, W=Pe, verbose=verbose)


		shot_noise_pressure += np.sum(Pe**2)
		short_noise_matter_pressure += np.sum(Pe*mass)
		total_mass += np.sum(mass)
		print(i)

	mean_mass = total_mass/grid**3
	shot_noise_pressure = shot_noise_pressure * BoxSize**3/(grid**3)**2
	short_noise_matter_pressure = short_noise_matter_pressure/mean_mass * BoxSize**3/(grid**3)**2

	return shot_noise_pressure, short_noise_matter_pressure

def get_ne_Mead_cube(ne_cube, little_h, fold, ptype=0):
	cell_volume = (BoxSize*u.Mpc/cu.littleh/grid)**3
	for i in range(f.header.num_files):
		this_snap = g3read.GadgetFile(snap_path + str(i))
		Pe = Pk_tools.get_field(ptype, this_snap, 'ne_Mead', little_h, cell_volume)
		pos = this_snap.read_new('POS ', ptype)*1e-3/fold


		pos, Pe = pos.astype('float32'), Pe.astype('float32')
		MASL.MA(pos, ne_cube, BoxSize, MAS, W=Pe, verbose=verbose)

		print(i)

if __name__ == '__main__':
	args = parser.parse_args()

	fold = args.fold
	sim_box = args.box
	sim_name = args.sim
	snap_dir = args.snap
	grid = args.grid
	threads = args.threads
	save_cube = args.save_cube
	save_Pk = args.save_Pk
	field = args.field

	field_name = 'pressure' if 'Pe' in field else 'ne'
	current_directory = os.getcwd()

	if 'pranjalrs' in current_directory:
		snap_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}.'

	if 'di75sic' in current_directory:
		snap_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{sim_box}/{sim_name}/snapdir_{snap_dir}/snap_{snap_dir}.'

	print(f'Path to snap shot files is: {snap_path}')

	f = g3read.GadgetFile(snap_path+'0')

	# density field parameters
	grid    = grid   # the 3D field will have grid x grid x grid voxels
	BoxSize = f.header.BoxSize/1e3/fold # Mpc/h ; size of box
	redshift = f.header.redshift  # Redshift
	little_h = f.header.HubbleParam
	MAS     = args.MAS  # mass-assigment scheme
	verbose = True   # print information on progress
	threads = threads
	axis = 0


	cube = np.zeros((grid,grid,grid), dtype=np.float32)

	if field == 'Pe_wht_vol':
		norm = np.zeros((grid,grid,grid), dtype=np.float32)

		get_Pe_cube(cube, norm, z=redshift, little_h=little_h, fold=fold)
		cube[norm!=0] /= norm[norm!=0]
		del norm

	elif field == 'Pe_Mead':
		shot_noise_auto, shot_noise_cross = get_Pe_Mead_cube(cube, little_h=little_h, fold=fold)

	elif field == 'ne_Mead':
		get_ne_Mead_cube(cube, little_h=little_h, fold=fold)

	Pk = PKL.Pk(cube, BoxSize, axis, MAS, verbose)

	header = f'''Power spectrum of the {field_name} field in the Magneticum simulation {sim_name} at redshift z={redshift:.2f}.
	The columns are: k [h/Mpc], Pk [Mpc^3/h^3](sim_name, z)
	Shot noise {field_name} auto (not subtracted): {shot_noise_auto}
	Shot noise matter-{field_name} cross (not subtracted): {shot_noise_cross}
	Fold: {fold} (Box size and particle positions are scaled by 1/fold)
	'''

	# Save Pk
	save_dir = f'../../magneticum-data/data/Pylians/Pk_{field_name}/{sim_box}/'
	if not os.path.exists(save_dir): os.makedirs(save_dir)
	np.savetxt(f'{save_dir}/{field}_R{grid}_z={redshift:.4f}.txt', np.column_stack((Pk.k3D, Pk.Pk[:,0])), delimiter='\t', header=header)

	# Save cube
	cube_save_dir = f'../../cube/{sim_box}/{field_name}/'
	if not os.path.exists(cube_save_dir): os.makedirs(cube_save_dir)
	np.save(f'{cube_save_dir}/delta_matter_{sim_name}_z={redshift:.2f}_R{grid}_fold_{fold}.npy', cube, allow_pickle=False)