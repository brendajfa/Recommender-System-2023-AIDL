import torch
import numpy as np


class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, 
                 field_dims: list,
                 embed_dim: float) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(len(field_dims), 1)
        self.embedding = torch.nn.Embedding(field_dims[-1], embed_dim)
        self.fm = FM_operation(reduce_sum=True)

        torch.nn.init.xavier_uniform_(self.embedding.weight.data) #Initializing weights

    def forward(self, interaction_pairs: torch.Tensor) -> torch.Tensor:
        """
        :param interaction_pairs: Long tensor of size ``(batch_size, num_fields)``
        """
        out = self.linear(interaction_pairs.float()) + self.fm(self.embedding(interaction_pairs))
        return out.squeeze(1)
        
    def predict(self, 
                interactions: np.ndarray,
                device: torch.device) -> torch.Tensor:
        # return the score, inputs are numpy arrays, outputs are tensors
        test_interactions = torch.from_numpy(interactions).to(dtype=torch.long, device=device)
        output_scores = self.forward(test_interactions)
        return output_scores

class FM_operation(torch.nn.Module):

    def __init__(self, 
                 reduce_sum: bool=True) -> None:
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self,
                x: torch.Tensor) -> float:
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        
        square_of_sum = torch.pow(torch.sum(x, dim=1),2)
        sum_of_square = torch.sum(torch.pow(x,2), dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    

