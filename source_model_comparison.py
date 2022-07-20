#!/bin/python3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

duneuropy_path='/home/malte/duneuro/build-release/duneuro-py/src'

import sys
sys.path.append(duneuropy_path)
import duneuropy as dp

# optionally add simbiopy for analytical EEG solutions
simbiopy_path='/home/malte/duneuro/simbiosphere/build/src'
sys.path.append(simbiopy_path)
import simbiopy as sp

# list file paths
mesh_path='tet_mesh.msh'
tensors_path='conductivities.txt'
electrodes_path='electrodes.txt'
dipoles_path='dipoles_ecc_0.99_radial.txt'

# create driver
volume_conductor_cfg = {'grid.filename' : mesh_path, 'tensors.filename' : tensors_path}
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
electrodes = dp.read_field_vectors_3d(electrodes_path)
meeg_driver.setElectrodes(electrodes, electrode_cfg)
print('Electodes set')

# compute transfer matrix
print('Computing transfer matrix')
transfer_solver_config = {'reduction' : '1e-14'}
eeg_transfer_config = {'solver' : transfer_solver_config}
eeg_transfer_matrix, eeg_transfer_computation_information = meeg_driver.computeEEGTransferMatrix(eeg_transfer_config)
print('Transfer matrix computed')

# loading dipoles
print('Reading dipoles')
dipoles = dp.read_dipoles_3d(dipoles_path)
print('Dipoles read')

# define source model configs
partial_integration_cfg = {'type' : 'partial_integration'}
venant_cfg = {'type' : 'venant', 'numberOfMoments' : '3', 'referenceLength' : '20', 'weightingExponent' : '1', 'relaxationFactor' : '1e-6', 'mixedMoments' : 'true', 'restrict' : 'true', 'initialization' : 'closest_vertex'}
localized_subtraction_cfg = {'type' : 'localized_subtraction', 'restrict' : 'false', 'initialization' : 'single_element', 'intorderadd_eeg_patch' : '0', 'intorderadd_eeg_boundary' : '0', 'intorderadd_eeg_transition' : '0', 'extensions' : 'vertex vertex'}

print('Solving the EEG forward problem for different source models')
print('Partial integration...')
driver_cfg['source_model'] = partial_integration_cfg
start_time = time.time()
partial_integration_solutions, computation_information = meeg_driver.applyEEGTransfer(eeg_transfer_matrix, dipoles, driver_cfg)
print(f"Time : {time.time() - start_time}")

print('Venant...')
driver_cfg['source_model'] = venant_cfg
start_time = time.time()
venant_solutions, computation_information = meeg_driver.applyEEGTransfer(eeg_transfer_matrix, dipoles, driver_cfg)
print(f"Time : {time.time() - start_time}")

print('Localized subtraction...')
driver_cfg['source_model'] = localized_subtraction_cfg
start_time = time.time()
localized_subtraction_solutions, computation_information = meeg_driver.applyEEGTransfer(eeg_transfer_matrix, dipoles, driver_cfg)
print(f"Time : {time.time() - start_time}")
print('Numerical solutions computed')

print('Compute analytical solutions')
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

# relative error
# params : 
#			- numerical_solution 	: 1-dimensional numpy array
#			- analytical_solution : 1-dimensional numpy array, must be of the same size as numerical_solution
def relative_error(numerical_solution, analytical_solution):
  assert len(numerical_solution) == len(analytical_solution)
  return np.linalg.norm(np.array(numerical_solution) - np.array(analytical_solution)) / np.linalg.norm(analytical_solution)

re_pi = np.empty(number_of_dipoles)
re_venant = np.empty(number_of_dipoles)
re_ls = np.empty(number_of_dipoles)

print('Computing relative error')
for i in range(number_of_dipoles):
  re_pi[i] = relative_error(partial_integration_solutions[i], analytical_solutions[i])
  re_venant[i] = relative_error(venant_solutions[i], analytical_solutions[i])
  re_ls[i] = relative_error(localized_subtraction_solutions[i], analytical_solutions[i])

pi_df = pd.DataFrame({'source_model' : 'partial_integration', 'eccentricity' : 0.99, 'relative_error' : re_pi})
venant_df = pd.DataFrame({'source_model' : 'venant', 'eccentricity' : 0.99, 'relative_error' : re_venant})
ls_df = pd.DataFrame({'source_model' : 'localized_subtraction', 'eccentricity' : 0.99, 'relative_error' : re_ls})

total_df = pd.concat([pi_df, venant_df, ls_df])

sns.boxplot(x = 'eccentricity', y = 'relative_error', data=total_df, hue='source_model')
plt.show()







