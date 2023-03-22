import os
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import numpy as np
import logs
import dataset
import pointdata
import model_fm
import model_random
import model_pop
import model_nfc
import sampling
import exec
import plots

class Main():
    def __init__(self):
        
        # Select the dataset you want to try
        self.dataset = "Amazon"         #"movie lens"/"Amazon"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # > Variables ------------------------------------------------
        self.test_mode = False          # Simpler version with ess data + debug
        self.show_tb = False             #Tensorboard

        self.tuning_mode = False         # For experiments use the csv 8x8
        self.tuning_params = {
            "num_neg": 5,               # {4, 5, 6}         original: 4
            "leave_one_out": "TLOO",    # {TLOO, RLOO}      original: TLOO
            "topk": 10,                 # {10, 50, 100}     original: 10
            
            "hidden_size": 64,          # {32, 64, 128}     original: 32    
            "lr":1e-3,                  # {1e-4, 1e-3}      original: 1e-4

            "hidden_size_ncf": 64,      # {32, 64, 128}     original: 32  
            "lr_ncf":1e-4               # {1e-4, 1e-3}      original: 1e-4
        }

        self.ini_time  = datetime.now()
        self.exec_path = os.getcwd()
        self.strategy  = self.tuning_params["leave_one_out"]

        self.hparams = {
            'batch_size':64,
            'num_epochs': 12,
            'hidden_size':self.tuning_params["hidden_size"], 
            'learning_rate':self.tuning_params["lr"],
            'hidden_size_ncf':self.tuning_params["hidden_size_ncf"], 
            'learning_rate_ncf':self.tuning_params["lr_ncf"]
        }

        # Seed for Random Reproducibility
        self.seed=0
        self.random = exec.Execution.seed_everything(self.seed)

        self.pop_reco = []
        # < Variables ------------------------------------------------

        # > Classes --------------------------------------------------
        self.log = logs.Logs(exec_path=self.exec_path, ml=False)
        self.spl = sampling.Sample()
        self.exec = exec.Execution()
        self.ds = dataset.DataSet()
        # < Classes --------------------------------------------------
        
        # > Dataset --------------------------------------------------
        self.min_reviews, self.min_usuarios = [8,8] if self.tuning_mode else [6,6]
        
        # Dataset columns depending on chosen data
        if self.dataset == "movie lens":

            self.col_names =   {"col_id_reviewer": "user",
                                "col_id_product": "item",
                                "col_rating": "rating",
                                "col_timestamp": "timestamp"}
            
        else:

            self.col_names =   {"col_id_reviewer": "reviewerID",
                                "col_id_product": "asin",
                                "col_rating": "overall",
                                "col_unix_time": "unixReviewTime",
                                "col_timestamp": "timestamp",
                                "col_year": "year"}
            
        self.train_x, self.test_x = [], []
        # < Dataset --------------------------------------------------
        self.log.save_data_configuration("-"*20+f"\nDataset: {self.dataset}")

    def start(self):

        if self.test_mode:
            print(self.device)

        # Define Logs
        tb_fm, tb_rnd, tb_pop, tb_ncf = self.log.def_log()
        
        # > Dataset ---------------------------------------------------------------------------------
        if self.test_mode:
            NrRows = 5000
        else:
            NrRows = None

        # Message for tuning mode 
        if self.tuning_mode:
            self.log.save_data_configuration("-"*20+"\nTuning mode:")
            [self.log.save_data_configuration(f"{tuning_param}: {value_param}") for tuning_param, value_param in self.tuning_params.items()]
            self.log.save_data_configuration("-"*20+"\n")
        else:
            self.log.save_data_configuration("-"*20+"\nFinal results with 6x6 dataset:")
            [self.log.save_data_configuration(f"{tuning_param}: {value_param}") for tuning_param, value_param in self.tuning_params.items()]
            self.log.save_data_configuration("-"*20+"\n")

        df = self.ds.readDataSet(self.exec_path, self.min_reviews, self.min_usuarios, self.dataset,  nrows=NrRows)
        self.log.save_data_configuration(str(df.nunique()))
        data, dims = self.ds.getDims(df, self.col_names, self.dataset, self.col_names)
        if self.test_mode:
            print(f'df head :')
            print(f'{str(df.head())}')
            print(f'getDims data:')
            print(f'{str(data)}')
            print(f'getDims dims: {str(dims)}')
       
     
        if self.test_mode == True:
            print("Dim of users: {}\nDim of items: {}\nDims of unixtime: {}".format(dims[0], dims[1], dims[2]))
        # < Dataset ---------------------------------------------------------------------------------
        
        # > Split data Training and Test-------------------------------------------------------------
        self.train_x, self.test_x = self.exec.split_train_test(data, dims[0], self.strategy)
        self.train_x = self.train_x[:, :2]
        dims = dims[:2]
        print(dims)
        if self.test_mode:
            print(f'Train shape: {str(self.train_x.shape)}')
            print(f'Test shape: {str(self.test_x.shape)}')
            print(f'split_train_test dims: {str(dims)}')
        # < Split data Training and Test-------------------------------------------------------------
        
        # > Sampling Strategy -----------------------------------------------------------------------
        self.pop_reco = self.exec.get_pop_recons(self.train_x)
        
        # Perform negative sampling
        self.train_x, rating_mat = self.spl.ng_sample(self.train_x, dims, num_ng=self.tuning_params["num_neg"])
        if self.test_mode:
            print("Dimensions matrix:\n",dims)
        
        #Plot train dataset distribution
        plots.plot_Train_dataset(self.train_x, self.tuning_params, self.hparams, self.dataset)
        print(f"Start: train_dataset and dataloader")
        train_dataset = pointdata.PointData(self.train_x, dims)
        if self.test_mode:
            print(train_dataset[0])

        data_loader = DataLoader(train_dataset, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=0)
        print(f"End: train_dataset and dataloader")

        print(f"Start: zero_positions")
        zero_positions = self.spl.zero_positions(rating_mat, showtime=False)
        if self.test_mode:
            print(f"zero_positions size: {str(len(zero_positions))}") 
        print(f"End: zero_positions")

        print("Start: items2compute")
        items2compute = self.exec.items_to_compute(zero_positions, dims)
        if self.test_mode:
            print(f"items2compute size: {str(len(items2compute))}") 
        print("End: items2compute")
        # < Sampling Strategy -----------------------------------------------------------------------

        # > Build Test Set --------------------------------------------------------------------------
        self.test_x = self.exec.build_test_set(items2compute, self.test_x[:,:2])
        # > Build Test Set --------------------------------------------------------------------------
        
        # > Create Models --------------------------------------------------------------------------
        dims = train_dataset.dims
        if self.test_mode:
            print(f"train_dataset.dims: {str(dims)}")

        fm_model  = model_fm.FactorizationMachineModel(dims, self.hparams['hidden_size']).to(self.device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = torch.optim.Adam(params=fm_model.parameters(), lr=self.hparams['learning_rate'])
        rnd_model = model_random.RandomModel(dims)
        pop_model = model_pop.PopularityBasedModel(self.pop_reco)

        ncf_model     = model_nfc.NeuNCF(dims, self.hparams['hidden_size_ncf']).to(self.device)
        ncf_optimizer = torch.optim.Adam(params=ncf_model.parameters(), lr=self.hparams['learning_rate_ncf'])
        ncf_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        # < Create Models --------------------------------------------------------------------------
        
        # > Training and Test ----------------------------------------------------------------------
        # Hit Ratio: Measures whether the test item is in the top@K positions of the recommendation list
        # NDCG (Normalized Discounted Cumulative Gain). Measures the ranking quality which gives information about where in the raking is our test item. 
        # Coverage: Coverage is the percent of items in the training data the model is able to recommend on a test set.

        print("Start: Epochs")
        total_items = dims[1]-dims[0] 
        training_time_start = datetime.now()
        ln_sep_sz = 58
        ln_sep_c = "-"
        self.log.save_data_configuration("\n\n")
        self.log.save_data_configuration(ln_sep_c*ln_sep_sz)
        self.log.save_data_configuration("Training and Test")
        self.log.save_data_configuration(ln_sep_c*ln_sep_sz)
        topk = self.tuning_params["topk"]
        topks = str(topk)
        col1 = 5
        col2 = 5
        col3 = 11
        col4 = 10
        col5 = 10
        col6 = 12
        dp = ".4f" #Decimal places

        print(self.log.save_data_configuration(f'| {str("Epoch").ljust(col1)} | {str("Model").ljust(col2)} | {str("HR@"+topks).ljust(col4)} | {str("NDCG@"+topks).ljust(col5)} | {str("%Coverage@"+topks).ljust(col6)} |'))
        print(self.log.save_data_configuration(ln_sep_c*ln_sep_sz))
        for epoch_i in range(self.hparams['num_epochs']):
            train_loss_fm  = self.exec.train_one_epoch(fm_model, optimizer, data_loader, criterion, self.device)
            train_loss_ncf = self.exec.train_one_epoch(ncf_model, ncf_optimizer, data_loader, ncf_criterion, self.device)

            hr, ndcg, reco_list_fm, cov_fm = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_fm, cov_fm = self.exec.test(fm_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'| {str(epoch_i).rjust(col1)} | {str("FM").ljust(col2)} | {str(format(hr,dp)).rjust(col4)} | {str(format(ndcg,dp)).rjust(col5)} | {str(format(cov_fm,dp)).rjust(col6)} |'))
            tb_fm.add_scalar('train/loss', train_loss_fm, epoch_i)
            tb_fm.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_fm.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_fm.add_scalar(f'eval/Coverage@{topk}', cov_fm, epoch_i)

            hr, ndcg, reco_list_rnd, cov_rnd = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_rnd, cov_rnd = self.exec.test(rnd_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'| {str(epoch_i).rjust(col1)} | {str("RND").ljust(col2)} | {str(format(hr,dp)).rjust(col4)} | {str(format(ndcg,dp)).rjust(col5)} | {str(format(cov_rnd,dp)).rjust(col6)} |'))
            tb_rnd.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_rnd.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_rnd.add_scalar(f'eval/Coverage@{topk}', cov_rnd, epoch_i)

            hr, ndcg, reco_list_pop, cov_pop = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_pop, cov_pop = self.exec.test_pop(pop_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'| {str(epoch_i).rjust(col1)} | {str("POP").ljust(col2)} | {str(format(hr,dp)).rjust(col4)} | {str(format(ndcg,dp)).rjust(col5)} | {str(format(cov_pop,dp)).rjust(col6)} |'))
            tb_pop.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_pop.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_pop.add_scalar(f'eval/Coverage@{topk}', cov_pop, epoch_i)

            hr, ndcg, reco_list_ncf, cov_ncf = 0.0, 0.0, [], 0.0
            hr, ndcg, reco_list_ncf, cov_ncf = self.exec.test(ncf_model, self.test_x, total_items, self.device, topk=topk)
            print(self.log.save_data_configuration(f'| {str(epoch_i).rjust(col1)} | {str("NCF").ljust(col2)} | {str(format(hr,dp)).rjust(col4)} | {str(format(ndcg,dp)).rjust(col5)} | {str(format(cov_ncf,dp)).rjust(col6)} |'))
            tb_ncf.add_scalar('train/loss', train_loss_ncf, epoch_i)
            tb_ncf.add_scalar(f'eval/HR@{topk}', hr, epoch_i)
            tb_ncf.add_scalar(f'eval/NDCG@{topk}', ndcg, epoch_i)
            tb_ncf.add_scalar(f'eval/Coverage@{topk}', cov_ncf, epoch_i)

            # plot recommended vs Popular
            plots.plot_Reco_vs_POP(reco_list_fm, reco_list_pop, "reco_ep_"+ str(epoch_i)+ "" + datetime.now().strftime("%Y%m%d%H%M%S")+".png", str(epoch_i), self.hparams['num_epochs'], "Factorization Machine")
            plots.plot_Reco_vs_POP(reco_list_pop, reco_list_pop, "reco_ep_"+ str(epoch_i)+ "" + datetime.now().strftime("%Y%m%d%H%M%S")+".png", str(epoch_i), self.hparams['num_epochs'], "Popularity")
            plots.plot_Reco_vs_POP(reco_list_rnd, reco_list_pop, "reco_ep_"+ str(epoch_i)+ "" + datetime.now().strftime("%Y%m%d%H%M%S")+".png", str(epoch_i), self.hparams['num_epochs'], "Random")
            plots.plot_Reco_vs_POP(reco_list_ncf, reco_list_pop, "reco_ep_"+ str(epoch_i)+ "" + datetime.now().strftime("%Y%m%d%H%M%S")+".png", str(epoch_i), self.hparams['num_epochs'], "NeuMF")

            print(self.log.save_data_configuration(ln_sep_c*ln_sep_sz))
        
        plots.show_generated_plots()
        training_time_end = datetime.now()-training_time_start
        seconds = training_time_end.seconds
        if seconds > 60: 
            seconds = seconds / 60
            secmin = "minutes"
            if seconds > 60:
                seconds = seconds / 60
                secmin = "hours"
        else:
            secmin = "seconds"
        print(self.log.save_data_configuration(f'Training duration: {str(format(seconds,dp))} {secmin}'))
        # < Training and Test ----------------------------------------------------------------------
      
        #Calc Total Time of execution
        txt = self.exec.efe(self.ini_time)
        print(self.log.save_data_configuration(txt))

        if self.show_tb:
            self.log.show_tensorboard()

    # > End of Method Start-------------------------------------------------------------------------
    
# < End of Class Main

if __name__ == '__main__':
    main = Main()
    main.start()
    exit()
