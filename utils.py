import os
import logging
from typing import Tuple, List, Dict, Any
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




# ================== Utilities ==================

def load_config(config_path: str = 'config.yaml') -> OmegaConf:
    if not os.path.exists(config_path):
        logging.error(f"Configuration file not found at: {config_path}")
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    try:
        config = OmegaConf.load(config_path)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Error loading or parsing YAML file {config_path} with OmegaConf: {e}")
        raise


class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input
    
def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches

# ================== Dataset ==================
class FunctionDataset(torch.utils.data.Dataset):
    def __init__(self, N, dim, sigma, f):
        self.X = torch.rand((N, dim)) * 2 - 1
        self.Y = f(self.X) + torch.randn_like(self.X) * sigma

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class MultivariateNormalDataset(torch.utils.data.Dataset):

    def __init__(self, N, dim, rho):
        self.N = N
        self.rho = rho
        self.dim = dim

        self.cov_matrix = self._сov_matrix()
        self.dist = self.build_dist
        self.x = self.dist.sample((N, ))
        self.dim = dim

        


    def __getitem__(self, ix):
        a, b = self.x[ix, 0:self.dim], self.x[ix, self.dim:2 * self.dim]
        return a, b

    def __len__(self):
        return self.N

    @property
    def build_dist(self):
        mu = torch.zeros(2 * self.dim)
        dist = MultivariateNormal(mu, self.cov_matrix)
        return dist

    def _сov_matrix(self):
        cov = torch.zeros((2 * self.dim, 2 * self.dim))
        cov[torch.arange(self.dim), torch.arange(
            start=self.dim, end=2 * self.dim)] = self.rho
        cov[torch.arange(start=self.dim, end=2 * self.dim),
            torch.arange(self.dim)] = self.rho
        cov[torch.arange(2 * self.dim), torch.arange(2 * self.dim)] = 1.0

        return cov

    @property
    def true_mi(self):
        return -0.5 * np.log(np.linalg.det(self.cov_matrix.data.numpy()))

def plot_mi_results(results_dict: dict, output_dir: str = 'mine_mi_estimator/plots'):
    logging.info("Generating plot of True vs Estimated Mutual Information...")
    fig, axs = plt.subplots(1, len(results_dict), sharex=True, figsize=(15, 6))

    if len(results_dict) == 1:
        axs = [axs]  

    for ix, loss in enumerate(results_dict.keys()):
        
        results = results_dict[loss]
        
        rhos = results[:, 0]
        estimated_mis = results[:, 1]
        true_mis = results[:, 2]
        
        ax = axs[ix]
        
        line1 = ax.plot(rhos, estimated_mis, 'o-', label=f'MINE ({loss})')
        line2 = ax.plot(rhos, true_mis, 'r--', label='True MI')
        
        ax.set_xlabel('Correlation coefficient (ρ)')
        ax.set_ylabel('Mutual Information')
        ax.set_title(f"{loss} estimated vs True MI")
        ax.grid(True)
        
        if ix == 0:
            plots = [line1[0], line2[0]]  

    #fig.legend(plots, ['MINE', 'True MI'], loc='upper right')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'mi_estimation_results.png')
    fig.savefig(plot_path)
    logging.info(f"Plot saved to {plot_path}")
    plt.show()
    
    
