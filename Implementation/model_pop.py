import torch
import numpy as np

class PopularityBasedModel(torch.nn.Module):
  def __init__(self, 
                 popularity_recommendations) -> None:
        super(PopularityBasedModel, self).__init__()
        self.popularity_recommendations = popularity_recommendations
  
  def predict(self) -> torch.Tensor:
      return torch.IntTensor(self.popularity_recommendations.astype(np.int32))