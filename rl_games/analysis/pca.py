import torch

import matplotlib.pyplot as plt

from loguru import logger

from crl.analysis._io import save_json, load_json

Tensor = torch.Tensor

class PCA:
    """Class for principal component analysis.

    Attributes:
        trajectory: hidden state trajectory
        fixed_points: hidden state fixed point
        pc_trajectory: trajectory in PCA space
        eigen_values: Evalues from PCA
        eigen_vectors: Evectors from PCA
    """
    
    def __init__(self,
                 trajectory: Tensor,
                 plot: bool=False,
    ):
        """Inits PCA class.
        
        Args:
            trajectory: hidden state trajectory of RNN
            fixed_points: fixed points of RNN
            plot: flag for plotting
        """
        # Center trajectory data around its mean
        self.trajectory = trajectory - torch.mean(trajectory,dim=0)
        self.eigen_values = None
        self.eigen_vectors = None

        # Run PCA analysis
        self._fit()

    def _fit(self):
        """Perform principal component analysis."""
        # Find covariance matrix
        covariance_matrix = torch.cov(torch.t(self.trajectory))

        # Returns eigenvalue matrix and eigenvector matrix (eigenvectors are column vectors)
        self.eigen_values, self.eigen_vectors = torch.linalg.eig(covariance_matrix)

        # Take the real component of the eigenvalues and eigenvectors
        self.eigen_values = torch.real(self.eigen_values)
        self.eigen_vectors = torch.real(self.eigen_vectors)

        logger.debug(f"\nEvalues (real): {self.eigen_values.size()} {self.eigen_values}\n")
        logger.debug(f"\nEvectors (real): {self.eigen_vectors.size()} {self.eigen_vectors}]\n")

    def save_pca_transform(self, filepath: str):
        """Save PCA transform to JSON file.
            
        Args:
            filepath: file path to save
        """
        save_json(filepath, self.eigen_vectors.tolist())

def transform_data(data: Tensor, projection_matrix: Tensor) -> Tensor:
    """Project tensor into principal component space.

    Args:
        data: input tensor data
    """
    pc_data = torch.matmul(data, projection_matrix)
    
    return pc_data

def transform_fixed_points(fixed_points: Tensor, projection_matrix: Tensor) -> Tensor:
    """Project FixedPoints into principal component space.

    Args:
        fixed_points: instance of FixedPoints
    """
    for n, fp in enumerate(fixed_points):
        fixed_points[n].h = transform_data(fp.h, projection_matrix)
    return fixed_points

def load_pca_transform(filepath: str):
    """Load PCA transform from JSON file.

    Args:
        filepath: file path to load
    """
    data = torch.tensor(load_json(filepath))

    return data
