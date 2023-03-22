import os
import pickle
import datetime
from pathlib import Path

class SaveData():
    def __init__(self):
        self.mod_path = ""
        self.train_file = ""
        self.test_file = ""
    
    def init(self, exec_path):
        today = datetime.today()
        self.mod_path = exec_path / Path("4_Modelling/mod_baseline")
        os.makedirs(self.mod_path, exist_ok=True)
        timestamp = today.strftime("%dday%mmon%Yyear")
        self.train_file = self.mod_path / f"MOD_baseline_train_{timestamp}.pkl"
        self.test_file = self.mod_path / f"MOD_baseline_test_{timestamp}.pkl"
        self.pop_file = self.mod_path / f"MOD_baseline_popRec_{timestamp}.pkl"
    
    def write(self, data, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save_train(self, data):
        self.write(data, self.train_file)
    
    def save_test(self, data):
        self.write(data, self.test_file)

    def save_pop(self, data):
        self.write(data, self.pop_file)