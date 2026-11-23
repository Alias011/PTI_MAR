#%%
import os
import deepinv as dinv
import torch
from spyrit.core.meas import HadamSplit2d
from pathlib import Path


device = dinv.utils.get_device()
rng = torch.Generator(device=device).manual_seed(0)

#%% bring dataset


import re
from torchvision.transforms import ToTensor
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import deepinv as dinv


# --------------------------------------------------
# Chargement et préparation des données
# --------------------------------------------------

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


#%% création du dataset et vérification compatiblité DeepInv

project_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(project_dir, "data")

gt_dir = os.path.join(data_dir, "GT_formatted")
meas_dir = os.path.join(data_dir, "meas_formatted")

dataset = HyperspectralDataset(gt_dir, meas_dir)

# Vérification de compatibilité DeepInverse
dinv.datasets.check_dataset(dataset)
#%% création du DataLoader et petits affichages de test
device = dinv.utils.get_device()

dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True
)

# Chargement d'un échantillon
sample = next(iter(dataloader))

x = sample["x"].to(device)
y = sample["y"].to(device)

print("Ground truth :", x.shape)
print("Measurement  :", y.shape)


print("GT original :", np.load(os.path.join(gt_dir, "cube0.npy")).shape)
print("Measurement :", np.load(os.path.join(meas_dir, "y_1024_0.npy")).shape)

print("GT tensor :", x.shape)
print("Measurement tensor :", y.shape)

#%% modele
N = 64

# define a spyrit meas_op
meas_spyrit = HadamSplit2d(N, device=device, reshape_output=True)

# la physique de notre problème inverse (linéaire)

meas_deepinv = dinv.physics.LinearPhysics(
    lambda y: meas_spyrit.measure_H(y) / norm,
    A_adjoint=lambda y: meas_spyrit.unvectorize(meas_spyrit.adjoint_H(y) / norm),
)


# calculer parametre de normalisation
norm = torch.linalg.norm(meas_spyrit.H, ord=2)


#%% modèle

model = dinv.models.UNet(
    in_channels=1,
    out_channels=1,
    channels_per_scale=[32, 64, 128],
    residual=False,
).to(device)

print(model)

#%% Entraînement

from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

# Paramètres
epochs = 50
learning_rate = 1e-3
band_batch_size = 32

# Séparation train / validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(0)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)

# Fonction de coût
criterion = nn.MSELoss()

# Optimiseur
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# Historique
train_losses = []
val_losses = []

# --------------------------------------------------
# Boucle d'entraînement
# --------------------------------------------------

for epoch in range(epochs):

    # ========== TRAIN ==========
    model.train()
    train_loss = 0.0

    for sample in train_loader:

        x = sample["x"].to(device)
        y = sample["y"].to(device)

        # x : (1, 512, 64, 64)
        # y : (1, 1, 128, 512)

        # Mise en forme des mesures
        y = y[0, 0].T   # (512, 128)

        # Rétroprojection
        x0 = y @ A.T    # (512, 4096)
        x0 = x0.reshape(-1, 1, *x.shape[-2:])

        # GT : (512, 64, 64)
        x = x[0].unsqueeze(1)

        # Traitement par petits groupes de bandes
        for start in range(0, x.shape[0], band_batch_size):

            end = start + band_batch_size

            x_batch = x[start:end]
            x0_batch = x0[start:end]

            optimizer.zero_grad()

            x_hat = model(x0_batch)

            loss = criterion(x_hat, x_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

    train_loss /= len(train_loader) * (
        (x.shape[0] + band_batch_size - 1) // band_batch_size
    )

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0

    with torch.no_grad():

        for sample in val_loader:

            x = sample["x"].to(device)
            y = sample["y"].to(device)

            y = y[0, 0].T

            x0 = y @ A.T
            x0 = x0.reshape(-1, 1, *x.shape[-2:])

            x = x[0].unsqueeze(1)

            for start in range(0, x.shape[0], band_batch_size):

                end = start + band_batch_size

                x_batch = x[start:end]
                x0_batch = x0[start:end]

                x_hat = model(x0_batch)

                loss = criterion(x_hat, x_batch)

                val_loss += loss.item()

    val_loss /= len(val_loader) * (
        (x.shape[0] + band_batch_size - 1) // band_batch_size
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Train Loss: {train_loss:.6e} | "
        f"Val Loss: {val_loss:.6e}"
    )









