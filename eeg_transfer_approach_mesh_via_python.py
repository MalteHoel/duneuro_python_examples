#!/bin/python3

import numpy as np
import time

duneuropy_path='/home/malte/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# optionally add simbiopy for analytical EEG solutions
simbiopy_path = '/home/malte/duneuro/simbiosphere/build/src'
sys.path.append(simbiopy_path)
import simbiopy as sp

# load data
print('Loading data')
mesh_data = np.load('example_tet_mesh_data.npz')
nodes = mesh_data['nodes']
elements = mesh_data['elements']
labels = mesh_data['labels']
conductivities = mesh_data['conductivities']
electrodes = mesh_data['electrodes']
dipoles_radial = mesh_data['dipoles_radial']
print('Data loaded')

# define dictionary containing the mesh
grid_cfg = {'nodes' : nodes, 'elements' : elements}
tensor_cfg = {'labels' : labels, 'conductivities' : conductivities}
volume_conductor_cfg = {'grid' : grid_cfg, 'tensors' : tensor_cfg}

driver_cfg = {'type' : 'fitted', 'solver_type' : 'cg', 'element_type' : 'tetrahedron', 'post_process' : 'true', 'subtract_mean' : 'true'}
solver_cfg = {'reduction' : '1e-14', 'edge_norm_type' : 'houston', 'penalty' : '20', 'scheme' : 'sipg', 'weights' : 'tensorOnly'}
driver_cfg['solver'] = solver_cfg
driver_cfg['volume_conductor'] = volume_conductor_cfg

print('Creating driver')
meeg_driver = dp.MEEGDriver3d(driver_cfg)
print('Driver created')

# set electrodes
print('Setting electrodes')
electrode_cfg = {'type' : 'closest_subentity_center', 'codims' : '3'}
electrodes = [dp.FieldVector3D(electrode) for electrode in electrodes]
meeg_driver.setElectrodes(electrodes, electrode_cfg)
print('Electodes set')

# compute transfer matrix
print('Computing transfer matrix')
transfer_solver_config = {'reduction' : '1e-14'}
eeg_transfer_config = {'solver' : transfer_solver_config}
eeg_transfer_matrix, eeg_transfer_computation_information = meeg_driver.computeEEGTransferMatrix(eeg_transfer_config)
print('Transfer matrix computed')

# wrap dipoles in duneuro data structure
print('Wrapping dipoles')
dipoles = [dp.Dipole3d(dipole) for dipole in dipoles_radial]
print('Dipoles wrapped')

# solve EEG forward problem, which means computing the leadfield
print('Solving EEG forward problem')
source_model_cfg = {'type' : 'localized_subtraction', 'restrict' : 'false', 'initialization' : 'single_element', 'intorderadd_eeg_patch' : '0', 'intorderadd_eeg_boundary' : '0', 'intorderadd_eeg_transition' : '0', 'extensions' : 'vertex vertex'}
driver_cfg['source_model'] = source_model_cfg
start_time = time.time()
numerical_solutions, computation_information = meeg_driver.applyEEGTransfer(eeg_transfer_matrix, dipoles, driver_cfg)
print('EEG forward problem solved')
print(f"Solving EEG forward problem took {time.time() - start_time} seconds")


##################
## Optional : Compute analytical solutions via simbio
##################

# compute EEG forward solutions analytically using simbiopy
# mm
print('Computing analytical EEG solutions')
center = [127, 127, 127]
# mm
radii = [92, 86, 80, 78]
# S/ mm
conductivities = [0.00043, 0.00001, 0.00179, 0.00033]
number_of_electrodes = len(electrodes)
number_of_dipoles = len(dipoles)
analytical_solutions = [None] * number_of_dipoles
electrodes_simbio = [np.array(electrode).tolist() for electrode in electrodes]
for count, dipole in enumerate(dipoles):
  analytical_solution = sp.analytic_solution(radii, center, conductivities, electrodes_simbio, np.array(dipole.position()).tolist(), np.array(dipole.moment()).tolist())
  mean = sum(analytical_solution) / len(analytical_solution)
  analytical_solution = [x - mean for x in analytical_solution]
  analytical_solutions[count] = analytical_solution
print('Analytical EEG solutions computed')

# compare numerical and analytical solutions

# first define error measures

# relative error
# params : 
#			- numerical_solution 	: 1-dimensional numpy array
#			- analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
def relative_error(numerical_solution, analytical_solution):
  assert len(numerical_solution) == len(analytical_solution)
  return np.linalg.norm(np.array(numerical_solution) - np.array(analytical_solution)) / np.linalg.norm(analytical_solution)

relative_errors = [None] * number_of_dipoles
for i in range(number_of_dipoles):
  relative_errors[i] = relative_error(numerical_solutions[i], analytical_solutions[i])

print(f"The average relative error is {sum(relative_errors)/len(relative_errors)}")