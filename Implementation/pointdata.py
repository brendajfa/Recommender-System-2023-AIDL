import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple

class PointData(Dataset):
    def __init__(self,
                 data: np.ndarray,
                 dims: list) -> None:
        """
        Dataset formatter adapted point-wise algorithms
        Parameters
        """
        super(PointData, self).__init__()
        self.interactions = data
        self.dims = dims

    def __len__(self) -> int:
        return len(self.interactions)
        
    def __getitem__(self, 
                    index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the pairs user-item and the target.
        """
        return self.interactions[index][:-1], self.interactions[index][-1]