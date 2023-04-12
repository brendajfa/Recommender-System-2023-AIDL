import os
from datetime import datetime
import torch
# from torch.utils.data import DataLoader
import numpy as np
import logs
# import pointdata
# import model_fm
# import model_random
# import model_pop
# import model_nfc
import sampling
import exec
# import plots
from main import Main

def test_main():
    dataset = "movie lens"
    main = Main(dataset)

