import torch
import numpy as np
import random

class RandomModel(torch.nn.Module):
    def __init__(self, 
                 dims: list) -> None:
        super(RandomModel, self).__init__()
        """
        Simple random based recommender system
        """
        self.all_items = list(range(dims[0], dims[1]))

    def forward(self) -> None:
        pass

    def predict(self,
                interactions: np.ndarray,
                device=None) -> torch.Tensor:
        return torch.FloatTensor(random.sample(self.all_items, len(interactions)))