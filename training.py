import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # Plotting
import seaborn as sns # Plotting

# Import Image Libraries - Pillow and OpenCV
from PIL import Image
import cv2

# Import PyTorch and useful fuctions
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
import torch.optim as optim
import torchvision.models as models # Pre-Trained models

# Import useful sklearn functions
import sklearn
from sklearn.metrics import cohen_kappa_score, accuracy_score

import time
from datetime import datetime
from tqdm import tqdm_notebook

import os
import random

# User-defined modules
from train_dataset import transforms, CreateDataset
from model import MainModel
from logger import Logger
from config import Config

# Open source libs


def add_data_to_loggers(loggers_list, column_name, data):
    loggers_list[0].add_data(column_name, data)
    loggers_list[1].add_data(column_name, data)

# FOR DETERMINISTIC RESLTS
def seed_torch(seed=13):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    ## CONFIG!
    cfg = Config()

    ## REPRODUCIBILITY
    seed_torch(cfg.seed)

    print(os.listdir("./input"))
    base_dir = "./input"

    # Loading Data + EDA

    train_csv = pd.read_csv('./input/train.csv')
    test_csv = pd.read_csv('./input/test.csv')
    print('Train Size = {}'.format(len(train_csv)))
    print('Public Test Size = {}'.format(len(test_csv)))

    train_csv.head()

    counts = train_csv['diagnosis'].value_counts()
    class_list = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    for i, x in enumerate(class_list):
        counts[x] = counts.pop(i)
        print("{:^12} - class examples: {:^6}".format(x, counts[x]))

    # Data Processing

    train_path = "./input/train_images/"
    train_data = CreateDataset(df_data=train_csv, data_dir=train_path, transform=transforms)

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(cfg.valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=valid_sampler)

    # Model
    model = cfg.model

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        model.cuda()

    # Trainable Parameters
    print("Number of trainable parameters: \n{}".format(cfg.pytorch_total_params))

    #Training(Fine-Tuning) and Validation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    val_kappa = []
    test_accuracies = []
    valid_accuracies = []
    kappa_epoch = []

    # Loggers

    logger_df = Logger(logsFileName=cfg.logsFileName + '.csv', mode = 'df')
    logger_txt = Logger(logsFileName=cfg.logsFileName + '.txt', mode = 'txt')
    loggers_list = [logger_df, logger_txt]

    loggers_list[0].add_empty_row()
    loggers_list[1].add_empty_row()

    add_data_to_loggers(loggers_list, 'date', datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S"))
    add_data_to_loggers(loggers_list, 'data-type', '')
    loggers_list[0].add_data('net-architecture', open('model.py', 'r+').read())
    add_data_to_loggers(loggers_list, 'loss-func', str(cfg.criterion))
    add_data_to_loggers(loggers_list, 'optim', str(cfg.optimizer))

    if cfg.scheduler is not None:
        add_data_to_loggers(loggers_list, 'scheduler', str(cfg.scheduler))

    if cfg.early_stopping is not None:
        add_data_to_loggers(loggers_list, 'early-stopping-patience', cfg.early_stopping_patience)
    else:
        add_data_to_loggers(loggers_list, 'early-stopping-patience', cfg.early_stopping)

    add_data_to_loggers(loggers_list, 'parameters-amount', cfg.pytorch_total_params)
    add_data_to_loggers(loggers_list, 'n-epochs', cfg.n_epochs)
    add_data_to_loggers(loggers_list, 'batch-size', cfg.batch_size)

    train_loss_best = np.inf
    valid_loss_best = np.inf
    kappa_best = 0

    add_data_to_loggers(loggers_list, 'best-train-loss', train_loss_best)
    add_data_to_loggers(loggers_list, 'best-valid-loss', valid_loss_best)
    add_data_to_loggers(loggers_list, 'best-kappa', kappa_best)

    loggers_list[0].add_data('cfg', open('config.py', 'r+').read())



    ## PRINT OUTPUT FREQUENCY
    print_frequency = cfg.print_frequency
    start_full_time = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        # For timing
        loggers_list[1].open()
        start_epoch_time = time.time()

        # keep track of training and validation loss
        train_loss_batch = []
        train_loss_epoch = []
        valid_loss_epoch = []
        ###################
        # train the cfg.model #
        ###################
        cfg.model.train()
        batch_n = 0
        for data, target in train_loader:
            batch_n += 1
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cpu().float()
            target = target.view(-1, 1)
            # clear the gradients of all optimized variables
            cfg.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward pass: compute predicted outputs by passing inputs to the cfg.model
                output = cfg.model(data).cpu()
                # calculate the batch loss
                loss = cfg.criterion(output, target).cpu()
                # backward pass: compute gradient of the loss with respect to cfg.model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                cfg.optimizer.step()
                data = data.cpu()
                train_loss_batch.append(loss.item())
                if batch_n % print_frequency == (print_frequency-1):
                    print('Train loss on {} batch: {:.6f}'.format(batch_n+1, np.mean(train_loss_batch)))
                    loggers_list[1].add_data(None, 'Train loss on {} batch: {:.6f}'.format(batch_n+1, np.mean(train_loss_batch)))
                    train_loss_epoch.append(np.mean(train_loss_batch))
                    train_loss_batch = []

        ######################
        # validate the cfg.model #
        ######################
        cfg.model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cpu().float()
            # forward pass: compute predicted outputs by passing inputs to the cfg.model
            target = target.view(-1, 1)
            with torch.set_grad_enabled(True):
                output = cfg.model(data).cpu()
                # calculate the batch loss
                loss = cfg.criterion(output, target)
            # update average validation loss
            valid_loss_epoch.append(loss.item())
            # output = output.cohen_kappa_score_kappa_score)
            y_actual = target.data.cpu().numpy()
            y_pred = output[:, -1].detach().cpu().numpy()
            val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))

            # calculate average losses
        train_loss_epoch = np.mean(train_loss_epoch)
        valid_loss_epoch = np.mean(valid_loss_epoch)
        valid_kappa = np.mean(val_kappa)
        kappa_epoch.append(np.mean(val_kappa))
        train_losses.append(train_loss_epoch)
        valid_losses.append(valid_loss_epoch)

        ## LOGGINS LOSSES
        if valid_loss_best > valid_loss_epoch:
            valid_loss_best  = valid_loss_epoch
            train_loss_best = train_loss_epoch
            kappa_best = valid_kappa
            add_data_to_loggers(loggers_list, 'best-train-loss', '{:.6f}'.format(train_loss_best))
            add_data_to_loggers(loggers_list, 'best-valid-loss', '{:.6f}'.format(valid_loss_best))
            add_data_to_loggers(loggers_list, 'best-kappa', '{:.4f}'.format(kappa_best))

        # print training/validation statistics
        print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f} | Estimated time: {:.2f}'.format(
            epoch, train_loss_epoch, valid_loss_epoch, valid_kappa, time.time() - start_epoch_time))
        loggers_list[1].add_data('', 'Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f} | Estimated time: {:.2f}'.format(
            epoch, train_loss_epoch, valid_loss_epoch, valid_kappa, time.time() - start_epoch_time))

        ##################
        # Early Stopping #
        ##################
        cfg.early_stopping(valid_loss_epoch, model_params_list=cfg.model_param_list, experiment_name=cfg.experiment_name, epoch=epoch)
        loggers_list[0].save()
        loggers_list[1].save()

    add_data_to_loggers(loggers_list, 'time_estimated', start_full_time - time.time())

if __name__ == '__main__':
    main()
