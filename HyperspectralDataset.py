from pathlib import Path
import re
import numpy as np
import torch
from torch.utils.data import Dataset


class HyperspectralDataset(Dataset):

    def __init__(self, gt_dir, meas_dir):

        self.gt_dir = Path(gt_dir)
        self.meas_dir = Path(meas_dir)

        # Récupération des fichiers
        gt_files = list(self.gt_dir.glob("cube*.npy"))
        meas_files = list(self.meas_dir.glob("y_1024_*.npy"))

        # Association des fichiers par indice
        gt_dict = {
            int(re.search(r"cube(\d+)", f.stem).group(1)): f
            for f in gt_files
        }

        meas_dict = {
            int(re.search(r"y_1024_(\d+)", f.stem).group(1)): f
            for f in meas_files
        }

        # Vérification des correspondances
        if set(gt_dict) != set(meas_dict):
            raise ValueError(
                f"Indices différents :\n"
                f"GT : {sorted(gt_dict)}\n"
                f"Measurements : {sorted(meas_dict)}"
            )

        self.indices = sorted(gt_dict)

        self.gt_files = gt_dict
        self.meas_files = meas_dict

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        i = self.indices[idx]

        # Chargement des données
        x = np.load(self.gt_files[i])
        y = np.load(self.meas_files[i])

        # Conversion en tenseurs PyTorch
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # Conversion de l'hypercube :
        # (H, W, B) -> (B, H, W)
        if x.ndim == 3:
            x = x.permute(2, 0, 1)

        # Ajout d'une dimension de canal aux mesures 2D
        if y.ndim == 2:
            y = y.unsqueeze(0)
        
        
        return {"x": x, "y": y}