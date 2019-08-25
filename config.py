from model import MainModel
from torch import optim
import torch.nn as nn
import os
from logger import Logger
from libs.earlystopping import EarlyStopping

import torch, random
import numpy as np

def seed_torch(seed=13):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Config:
    def __init__(self, batch_size=32, lr=0.00015, p_horizontalflip=0.4, model_type='ResNet101', training_mode='only_new'):
        ## INFO ABOUT EXPERIMENT
        self.logsFileName = 'LOGS'
        self.logsFileName_finetuning = 'LOGS_finetuning'
        self.seed = 13

        seed_torch(self.seed)

        if os.path.exists('./Logs/' + self.logsFileName + '.csv'):
            if training_mode == 'only_new':
                self.df_logger = Logger(self.logsFileName + '.csv', 'df')
                self.experiment_name = 'exp{}'.format(len(self.df_logger.logsFile)) + '_end_epoch'
                self.df_logger.save()
            elif training_mode == 'finetuning':
                self.df_logger = Logger(self.logsFileName_finetuning + '.csv', 'df')
                self.experiment_name = 'exp{}'.format(len(self.df_logger.logsFile)) + '_end_epoch'
                self.df_logger.save()
        else:
            self.experiment_name = 'exp{}'.format(0) + '_end_epoch'
        self.exper_type = 'mixed_data_imgsize_300'
        self.img_size = 480
        # self.img_size_crop = 300

        ## MODEL PARAMETERS
        self.weights_dir = './Model_weights/'
        self.weights_dir_finetuning = './Model_weights_finetuning/'
        self.model_type = model_type

        self.model = MainModel(model_type=self.model_type).model

        self.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=2, verbose=True)
        self.criterion = nn.MSELoss()
        # self.num_classes = 5
        self.model_param_list = [self.model, self.optimizer, self.scheduler]

        ## EARLY STOPPING
        self.early_stopping_patience = 10
        self.early_stopping = EarlyStopping(self.early_stopping_patience)
        self.early_stopping_loss = 'pytorch' #kappa

        ## TRAINING & VALIDATION SETUP

        self.num_workers = 16
        self.n_epochs = 150
        self.batch_size = batch_size
        self.valid_type = 'holdout' #CV
        self.valid_size = 0.2
        self.n_folds = 5 ## for CV!



        ## TRANSFORMER AND DATASET
        self.p_horizontalflip = p_horizontalflip
        self.data_type = 'new'

        ## PRINT FREQUENCY
        self.print_frequency = 50