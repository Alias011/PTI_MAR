"""
  module pre-processing qui sert à : prépare les données pour le réseau.
    et les placer dans un dossier cible sous format de tableau numpy avec un document csv contenant les paramètres
    
    TODO faire la sauvegarde des paramètres , question : en amont ?
"""

import os
import numpy as np
from get_reordered_unsplit_measures import get_reordered_unsplit_measures
from get_reconstructed_gt import get_reconstructed_gt
from pathlib import Path

def preprocessing(name_spectral_data_file : str, name_meta_data_file : str, bin_number : int, K : int, name_destination_dir : str) -> None:
    """Fonction de prép-processing permet de sauvegarder dans un repertoire de destination 
    contenant trois repertoires avec le x le y et le h contenant leur valeur en .npy


    Args:
        name_spectral_data_file (str): nom du fichier du spectre des données
        name_meta_data_file (str): nom du fichier des méta données
        bin_number (int): entier représentant la compression sur les longueurs d'ondes
        K (int): entier représantant le nombre du subsampled
        name_destination_dir (str): nom du fichier de destination
    """

    project_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir = os.path.join(project_dir, "data", "tests_downloaded")
    destination_dir = os.path.join(project_dir, "data", name_destination_dir)

    spectral_data_file = os.path.join(raw_data_dir, name_spectral_data_file)
    meta_data_file = os.path.join(raw_data_dir, name_meta_data_file)

     #recuperer, reordonnancer et unsplitter les mesures
    y_full, ind_hadamard = get_reordered_unsplit_measures(spectral_data_file, meta_data_file)

    #reconstruire les ground truths avec un pseudoinverse (hadamard) sur y_full 
    x_gt_full, norm = get_reconstructed_gt(y_full, spectral_data_file, meta_data_file)

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
    print("H : ", ind_hadamard.shape)
    

    destination_path = Path(destination_dir)

    n = 0
    while True:
        path_y_sub = destination_path / "y_subsampled" / f"y_subsampled{n}.npy"
        path_x_gt_binned = destination_path / "x_gt_binned" / f"x_gt_binned{n}.npy"
        path_ind_h = destination_path / "ind_hadamard" / f"ind_hadamard{n}.npy"
        
        if not path_y_sub.exists() and not path_x_gt_binned.exists() and not path_ind_h.exists():
            break
        n += 1

    path_y_sub.parent.mkdir(parents=True, exist_ok=True)
    path_x_gt_binned.parent.mkdir(parents=True, exist_ok=True)
    path_ind_h.parent.mkdir(parents=True, exist_ok=True)

    np.save(path_y_sub, y_subsampled)
    np.save(path_x_gt_binned, x_gt_binned)
    np.save(path_ind_h, ind_hadamard)

    print(f"y_subsampled{n} a été enregistré")
    print(f"x_gt_binned{n} a été enregistré")
    print(f"ind_hadamard{n} a été enregistré")

if __name__ == '__main__':
    
    print("test pre-processing")
    
    name_spectral_data_file = "tomato_slice_2_zoomx2_spectraldata.npz"
    name_meta_data_file = "tomato_slice_2_zoomx2_metadata.json"
    bin_number = 4
    K = 1024
    name_destination_dir = "data_test"
    
    preprocessing(name_spectral_data_file, name_meta_data_file, bin_number, K, name_destination_dir)