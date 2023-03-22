from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple
from tqdm import tqdm, trange
from statistics import mean
import torch
import random
import results

class Execution():
    def __init__(self):
        self.res = results.Results()
    
    def split_train_test(self, data: np.ndarray, n_users: int, strategy) -> Tuple[np.ndarray, np.ndarray]:
        # Split and remove timestamp
        train_x, test_x = [], []
        for u in trange(n_users, desc='spliting train/test and removing timestamp...'):
            user_data = data[data[:, 0] == u]
            sorted_data = user_data[user_data[:, -1].argsort()]
            if len(sorted_data) == 1:
                train_x.append(sorted_data[0][:-1])
            else:
                if (strategy == "TLOO"):
                    train_x.append(sorted_data[:-1][:, :-1])
                    test_x.append(sorted_data[-1][:-1])
                else:
                    # RLOO Random Leave One Out
                    idx = np.random.choice(np.arange(sorted_data.shape[0]), size=1)
                    test_x.append(sorted_data[idx,:-1]) #choose the random value for test
                    test_x= list(np.vstack(test_x))
                    sorted_data = np.delete(sorted_data, (idx), axis=0) # delete the random value from the train
                    train_x.append(sorted_data[:,:-1])
        return np.vstack(train_x), np.stack(test_x)
        
    def items_to_compute(self, zero_positions, dims):

        zp_df = pd.DataFrame(zero_positions[zero_positions[:, 1] >= dims[0]], columns=['u','i'])
        zp_df_index_u = zp_df.set_index(['u'])
        users = np.unique(zero_positions[:,0]) 

        items2compute = list(list() for _ in users)
        for i in trange(len(users)):
            items2compute[i] = np.hstack(zp_df_index_u.loc[i].to_numpy())

        return items2compute
    
    def build_test_set(self, itemsnoninteracted:list, gt_test_interactions: np.ndarray) -> list:
        test_set = []
        for pair, negatives in tqdm(zip(gt_test_interactions, itemsnoninteracted), desc="Building test set..."):
            # APPEND TEST SETS FOR SINGLE USER
            negatives = np.delete(negatives, np.where(negatives == pair[1]))
            single_user_test_set = np.vstack([pair, ] * (len(negatives)+1))
            single_user_test_set[:, 1][1:] = negatives
            test_set.append(single_user_test_set.copy())
        return test_set 
    
    def train_one_epoch(self, 
                        model: torch.nn.Module,
                        optimizer: torch.optim,
                        data_loader: torch.utils.data.DataLoader,
                        criterion: torch.nn.functional,
                        device: torch.device) -> float:
        model.train()
        total_loss = []

        for i, (interactions, targets) in enumerate(data_loader):
            interactions = interactions.to(device)
            targets = targets.to(device)
            predictions = model(interactions[:,:2])
            loss = criterion(predictions, targets.float())
            model.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())

        return mean(total_loss)

    def test(self, 
             model: torch.nn.Module,
             test_x: np.ndarray,
             total_items,
             device: torch.device,
             topk: int=10) -> Tuple[float, float]:
        # Test the HR and NDCG for the model @topK
        model.eval()
        
        user_reco_list = np.zeros(len(test_x)).tolist()
        idx = 0

        HR, NDCG = [], []
        for user_test in test_x:
            gt_item = user_test[0][1]
            predictions = model.predict(user_test, device)
            _, indices = torch.topk(predictions, min(topk, predictions.size()[0]))
            recommend_list = user_test[indices.cpu().detach().numpy()][:, 1]
            user_reco_list[idx] = recommend_list.tolist().copy()
            idx +=1
 
            HR.append(self.res.getHitRatio(recommend_list, gt_item))
            NDCG.append(self.res.getNDCG(recommend_list, gt_item))
        
        coverage = self.res.coverage(user_reco_list, total_items)
        return mean(HR), mean(NDCG), user_reco_list, coverage


    def test_pop(self, 
                 model: torch.nn.Module,
                 test_x: np.ndarray,
                 total_items,
                 device: torch.device,
                 topk: int=10) -> Tuple[float, float]:
        # Test the HR and NDCG for the model @topK
        model.eval()

        user_reco_list = np.zeros(len(test_x)).tolist()
        idx = 0

        HR, NDCG = [], []
        for user_test in test_x:
            gt_item = user_test[0][1]
            predictions = model.predict()
            reco_list = predictions[:topk]
            user_reco_list[idx] = np.hstack(reco_list.tolist().copy())
            idx +=1

            HR.append(self.res.getHitRatio(reco_list, gt_item))
            NDCG.append(self.res.getNDCG(reco_list, gt_item))
            
        coverage = self.res.coverage(user_reco_list, total_items)
        return mean(HR), mean(NDCG), user_reco_list, coverage

    def get_pop_recons(self, train_x):
        items_sorted = pd.DataFrame(train_x[:,:2], columns=[ "reviewerID","asin"]).groupby("asin").count().sort_values(by="reviewerID",ascending=False).reset_index()
        items_sorted.asin = items_sorted.asin.astype(str)
        return items_sorted.asin.to_numpy()

    def efe(self, startime):
        end_time = datetime.now()
        time_dif = end_time - startime
        seconds = time_dif.seconds
        secmin = ""
        if seconds > 60: 
            seconds = seconds / 60
            secmin = "minutes"
            if seconds > 60:
                seconds = seconds / 60
                secmin = "hours"
        elif seconds <= 60: 
            secmin = "seconds"
        efe = f'Total execution in {str(format(seconds,".4f"))} {secmin}'
        return efe
    
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
