'''prépare les données pour le réseau.
les place dans un dossier cible sous format de tenseurs pytorch'''

import os
import numpy as np
from get_reordered_unsplit_measures import get_reordered_unsplit_measures
from get_reconstructed_gt import get_reconstructed_gt

project_dir = os.path.dirname(os.path.abspath(__file__))
raw_data_dir = os.path.join(project_dir, "data")

#######################################
# entrees à ajuster
spectral_data_file = os.path.join(raw_data_dir, "tests_downloaded", "tomato_slice_2_zoomx2_spectraldata.npz")
meta_data_file = os.path.join(raw_data_dir, "tests_downloaded", "tomato_slice_2_zoomx2_metadata.json")
bin_number = 4
K = 1024
#######################################

#%%

#recuperer, reordonnancer et unsplitter les mesures
y_full, ind = get_reordered_unsplit_measures(spectral_data_file, meta_data_file)

#reconstruire les ground truths avec un pseudoinverse (hadamard) sur y_full 
x_gt_full, norm = get_reconstructed_gt(y_full, spectral_data_file, meta_data_file)
#%%
#bin des mesures et des gt (ON BIN SUR AXE 0 POUR x, puis on permute pour avoir (Nx,Ny,L))
print("binning x...")
#x_gt_full = x_gt_full.transpose(1,2,0)
x_gt_binned = x_gt_full.reshape(x_gt_full.shape[0], x_gt_full.shape[1], x_gt_full.shape[2]//bin_number, bin_number).sum(axis=-1)
#x_gt_binned = x_gt_binned.transpose(1, 2, 0)
print("done. \n")
print("binning y...")
y_binned = y_full.reshape(y_full.shape[0], y_full.shape[1]//bin_number, bin_number).sum(axis=-1)
print("done. \n")

#downsample des mesures
print("subsampling y...")
y_subsampled = y_binned[0:K, :]
print("done. \n")

print("_____________")
print("shapes of pre-processing outputs")
print("")
print(f"y : {y_subsampled.shape}")
print(f"x : {x_gt_binned.shape}")
