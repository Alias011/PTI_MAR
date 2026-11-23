import numpy as np
from metadata_SPC2D import read_metadata
import torch
import deepinv as dinv
from spyrit.core.meas import HadamSplit2d

def get_reconstructed_gt(y_full, spectral_data_file, meta_data_file):
    
    #reconstruire les gt par pseudo-inverse
    print("reconstructing gt cube...")
    
    y_raw = y_full
    P, L = y_raw.shape

    #Get pattern order
    _, acquisition_parameters, _, _ = read_metadata(meta_data_file)
    wavelengths = acquisition_parameters.wavelengths 
    N = acquisition_parameters.pattern_dimension_x  # probably np.sqrt(P//2) for uncompressed sensing (? right)

    patterns = acquisition_parameters.patterns # order of acquisition patterns
    ind = np.array(patterns[0::2])//2 # corresponding hadamard rows


    #N = 64
    # Mesures complètes, réordonnées et déjà unsplit
    
    L = y_raw.shape[1]
    
    device = dinv.utils.get_freer_gpu() if torch.cuda.is_available() else "cpu"
    
    # 1. Définition de l'opérateur Hadamard Spyrit
    meas_spyrit = HadamSplit2d(N, device=device, reshape_output=True)
    
    # 2. Normalisation
    norm = torch.linalg.norm(meas_spyrit.H, ord=2)
    print(f"norm = {norm}")
    # 3. Création de l'opérateur DeepInv
    meas_deepinv = dinv.physics.LinearPhysics(
        A=lambda x: meas_spyrit.measure_H(x) / norm,
        A_adjoint=lambda y: meas_spyrit.unvectorize(
            meas_spyrit.adjoint_H(y) / norm
        ),
    )
    
    # 4. Mise en forme des mesures
    # (L, 1, N*N) : une bande spectrale par élément du batch
    y = torch.from_numpy(y_raw.T).float().to(device)
    y = y.reshape(L, 1, N*N)
    
    # 5. Reconstruction par pseudo-inverse
    with torch.no_grad():
        x_pinv = meas_deepinv.A_dagger(y / norm)
    
    # Dimensions : (L, 1, N, N)
    x_pinv = x_pinv[:, 0, :, :]
    
    # 6. Retour au format cube hyperspectral FORMAT NUMPY ! PAS TENSEUR !
    cube_reconstructed = x_pinv.permute(1, 2, 0).cpu().numpy()
    
    print("done.")
    print(f"shape of reconstructed cube :{cube_reconstructed.shape}")
    # (64, 64, L)    
    
    return cube_reconstructed, norm