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
parser.add_argument('--save_cube', default=False, type=bool)
parser.add_argument('--save_Pk', default=True, type=bool)
parser.add_argument('--field', type=str)

def get_Pe_cube(Pe_cube, norm, little_h, ptype=0):
	for i in range(f.header.num_files):
		this_snap = g3read.GadgetFile(snap_path + str(i))
		Pe = Pk_tools.get_field(ptype, this_snap, 'Pe', little_h=little_h)
		pos = this_snap.read_new('POS ', ptype)*1e-3
		m_over_rho = np.array(this_snap.read_new('MASS', ptype))/np.array(this_snap.read_new('RHO ', ptype))

		pos, Pe = pos.astype('float32'), Pe.astype('float32')
		m_over_rho = m_over_rho.astype('float32')
		MASL.MA(pos, Pe_cube, BoxSize, MAS, W=Pe*m_over_rho, verbose=verbose)
		MASL.MA(pos, norm, BoxSize, MAS, W=m_over_rho, verbose=verbose)

		print(i)


def get_Pe_Mead_cube(Pe_cube, little_h, ptype=0):
	cell_volume = (BoxSize*u.Mpc/grid)**3
	for i in range(f.header.num_files):
		this_snap = g3read.GadgetFile(snap_path + str(i))
		Pe = Pk_tools.get_field(ptype, this_snap, 'Pe_Mead', little_h, cell_volume)
		pos = this_snap.read_new('POS ', ptype)*1e-3


		pos, Pe = pos.astype('float32'), Pe.astype('float32')
		MASL.MA(pos, Pe_cube, BoxSize, MAS, W=Pe, verbose=verbose)

		print(i)

def get_ne_Mead_cube(ne_cube, little_h, ptype=0):
	cell_volume = (BoxSize*u.Mpc/cu.littleh/grid)**3
	for i in range(f.header.num_files):
		this_snap = g3read.GadgetFile(snap_path + str(i))
		Pe = Pk_tools.get_field(ptype, this_snap, 'ne_Mead', little_h, cell_volume)
		pos = this_snap.read_new('POS ', ptype)*1e-3


		pos, Pe = pos.astype('float32'), Pe.astype('float32')
		MASL.MA(pos, ne_cube, BoxSize, MAS, W=Pe, verbose=verbose)

		print(i)

if __name__ == '__main__':
	args = parser.parse_args()

	box = args.box
	sim = args.sim
	snap_dir = args.snap
	grid = args.grid
	threads = args.threads
	save_cube = args.save_cube
	save_Pk = args.save_Pk
	field = args.field

	current_directory = os.getcwd()

	if 'pranjalrs' in current_directory:
		snap_path = f'/xdisk/timeifler/pranjalrs/magneticum_data/{box}/{sim}/snapdir_{snap_dir}/snap_{snap_dir}.'

	if 'di75sic' in current_directory:
		snap_path = f'/dss/dssfs02/pr62go/pr62go-dss-0001/Magneticum/{box}/{sim}/snapdir_{snap_dir}/snap_{snap_dir}.'

	print(f'Path to snap shot files is: {snap_path}')

	f = g3read.GadgetFile(snap_path+'0')

	# density field parameters
	grid    = grid   # the 3D field will have grid x grid x grid voxels
	BoxSize = f.header.BoxSize/1e3 # Mpc/h ; size of box
	redshift = f.header.redshift  # Redshift
	little_h = f.header.HubbleParam
	MAS     = args.MAS  # mass-assigment scheme
	verbose = True   # print information on progress
	threads = threads
	axis = 0

	Pe_cube = np.zeros((grid,grid,grid), dtype=np.float32)

	if field == 'Pe_wht_vol':
		norm = np.zeros((grid,grid,grid), dtype=np.float32)

		get_Pe_cube(Pe_cube, norm, z=redshift, little_h=little_h)
		Pe_cube[norm!=0] /= norm[norm!=0]
		del norm

	elif field == 'Pe_Mead':
		get_Pe_Mead_cube(Pe_cube, little_h=little_h)

	elif field == 'ne_Mead':
		get_ne_Mead_cube(Pe_cube, little_h=little_h)

	if save_cube is True:
		#np.save(f'/xdisk/timeifler/pranjalrs/cube/{sim_box}_{Pe_field}_{MAS}_R{grid}.npy', Pe_cube)
		np.save(f'../../cube/{box}/{sim}/{field}_R{grid}_z={redshift:.4f}.npy', Pe_cube)

	if save_Pk is True:
		Pk = PKL.Pk(Pe_cube, BoxSize, axis, MAS, verbose)

		if 'Pe' in field:
			np.savetxt(f'../../../magneticum-data/data/Pylians/Pk_pressure/{box}/{field}_R{grid}_z={redshift:.4f}.txt', np.column_stack((Pk.k3D, Pk.Pk[:,0])), delimiter='\t')

		elif 'ne' in field:
			np.savetxt(f'../../magneticum-data/data/Pylians/Pk_electron_density/{box}/{field}_R{grid}_z={redshift:.4f}.txt', np.column_stack((Pk.k3D, Pk.Pk[:,0])), delimiter='\t')
