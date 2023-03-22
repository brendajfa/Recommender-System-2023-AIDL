import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from typing import Tuple
from datetime import datetime

class Sample():
    def __init__(self):
        None
    
    def build_adj_mx(self,n_feat:int, data:np.ndarray) -> sp.dok_matrix :
        train_mat = sp.dok_matrix((n_feat, n_feat), dtype=np.float32)
        for x in tqdm(data, desc=f"BUILDING ADJACENCY MATRIX..."):
            train_mat[x[0], x[1]] = 1.0
            train_mat[x[1], x[0]] = 1.0
            # IDEA: We treat features that are not user or item differently because we do not consider
            #  interactions between contexts
            if data.shape[1] > 2:
                for idx in range(len(x[2:])):
                    train_mat[x[0], x[2 + idx]] = 1.0
                    train_mat[x[1], x[2 + idx]] = 1.0
                    train_mat[x[2 + idx], x[0]] = 1.0
                    train_mat[x[2 + idx], x[1]] = 1.0
        return train_mat
    
    def ng_sample(self, data: np.ndarray, dims: list, num_ng:int=4) -> Tuple[np.ndarray, sp.dok_matrix]:
        rating_mat = self.build_adj_mx(dims[-1], data)
        interactions = []
        min_item, max_item = dims[0], dims[1]
        for num, x in tqdm(enumerate(data), desc='Perform negative sampling...'):
            interactions.append(np.append(x, 1))
            for t in range(num_ng):
                j = np.random.randint(min_item, max_item) 
                # IDEA: Loop to exclude true interactions (set to 1 in adj_train) user - item
                while (x[0], j) in rating_mat or j == int(x[1]):
                    j = np.random.randint(min_item, max_item) 
                interactions.append(np.concatenate([[x[0], j], x[2:], [0]]))
        return np.vstack(interactions), rating_mat

    def zero_positions(self, rating_mat, showtime=False):
        if showtime:
            ini_time   = datetime.now()

        zero_true_matrix = np.where(rating_mat.A==0)
        zero_pos = np.asarray([zero_true_matrix[0],zero_true_matrix[1]]).T

        if showtime:
            end_time = datetime.now()
            time_dif = end_time - self.ini_time
            seconds = time_dif.seconds
            if seconds > 60: 
                seconds = seconds / 60
                print(f"zero_positions - Executed in {seconds} minutos")
            elif seconds <= 60: 
                print(f"zero_positions - Executed in {seconds} seconds")
        
        return zero_pos
