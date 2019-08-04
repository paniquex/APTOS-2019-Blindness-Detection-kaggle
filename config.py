from model import MainModel
from torch import optim
import torch.nn as nn
import os
from logger import Logger
from libs.earlystopping import EarlyStopping

class Config:
    def __init__(self):
        ## INFO ABOUT EXPERIMENT
        self.logsFileName = 'LOGS'
        self.seed = 13
        if os.path.exists('./Logs/' + self.logsFileName + '.csv'):
            self.df_logger = Logger(self.logsFileName + '.csv', 'df')
            self.experiment_name = 'exp{}'.format(len(self.df_logger.logsFile)) + '_end_epoch'
        else:
            self.experiment_name = 'exp{}'.format(0) + '_end_epoch'



        ## MODEL PARAMETERS
        self.model_type = 'ResNet'

        self.model = MainModel(model_type=self.model_type).model
        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.lr = 0.0015
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = None
        self.criterion = nn.MSELoss()
        self.model_param_list = [self.model, self.optimizer, self.scheduler]

        ## EARLY STOPPING
        self.early_stopping_patience = 8
        self.early_stopping = EarlyStopping(self.early_stopping_patience)

        ## TRAINING & VALIDATION SETUP
        self.n_epochs = 100
        self.batch_size = 32
        self.valid_type = 'HoldOut' #CV
        self.valid_size = 0.2
        self.n_folds = 5 ## for CV!


        ## TRANSFORMER AND DATASET
        self.p_horizontalflip = 0.4

        ## PRINT FREQUENCY
        self.print_frequency = 25
