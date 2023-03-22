import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import shutil

class Logs():
    def __init__(self, exec_path, ml=True):
        self.exec_path = exec_path
        self.multi_logs = ml
        self.log_dir = "logs"
        self.exectime = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.path_log_folder = f'{self.exec_path}/{self.log_dir}'
        self.save_data_dir = self.exectime + "_data_config.txt"
        self.path_save_data_dir = f'{self.exec_path}/{self.log_dir}/{self.save_data_dir}'

        if self.multi_logs == False:
            if os.path.exists(self.path_log_folder):
                shutil.rmtree(self.path_log_folder, ignore_errors=True)
            os.makedirs(self.path_log_folder, exist_ok=True)
            if os.path.isfile(self.path_save_data_dir):
                open(self.path_save_data_dir, "w").close()
        
        text = self.exectime + " - RS-Execution Result"
        with open(self.path_save_data_dir, "a") as data_file:
            data_file.write(text+"\n\n")

    def def_log(self):
        logs_base_dir = "runs"
        dir_path = f'{self.exec_path}/{self.log_dir}/{logs_base_dir}'

        if self.multi_logs == False:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)

        os.makedirs(dir_path, exist_ok=True)
        tb_fm  = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_FM/')
        tb_rnd = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_RANDOM/')
        tb_pop = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_POP/')
        tb_ncf = SummaryWriter(log_dir=f'{dir_path}/{logs_base_dir}_NCF/')
        return tb_fm, tb_rnd, tb_pop, tb_ncf

    def save_data_configuration(self, text):
        with open(self.path_save_data_dir, "a") as data_file:
            data_file.write(text+"\n")
        return text
    
    def show_tensorboard(self):
        os.system('tensorboard --logdir=./' + self.log_dir + ' --host localhost --port 8088')
        