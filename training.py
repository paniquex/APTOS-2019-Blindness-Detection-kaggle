
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import time
from datetime import datetime
from tqdm import tqdm_notebook

import os
import random

# User-defined modules
from train_dataset import transforms_train, transforms_valid, CreateDataset
from config import Config

from logger import Logger

# Open source libs


## GLOBAL CONSTANTS:



def add_data_to_loggers(loggers_list, column_name, data):
    loggers_list[0].add_data(column_name, data)
    loggers_list[1].add_data(column_name, data)

# FOR DETERMINISTIC RESLTS
from config import seed_torch

def __init_fn(worker_id):
    np.random.seed(13 + worker_id)

def main(batch_size, lr, p_horizontalflip, model_type, info):
    ## CONFIG!
    cfg = Config(batch_size=batch_size, lr=lr, p_horizontalflip=p_horizontalflip, model_type=model_type)

    ## REPRODUCIBILITY
    seed_torch(cfg.seed)


    print(os.listdir("./input"))
    base_dir = "./input"

    # Loading Data + EDA

    train_new_csv = pd.read_csv('./input/train_new.csv')
    train_old_csv = pd.read_csv('./input/train_old.csv')
    train_csv = None
    train_path = None
    if cfg.data_type == 'new_old_mixed':
        train_csv = pd.concat([train_new_csv, train_old_csv], axis=0)
        train_path = './input/train_mixed_images/'
    elif cfg.data_type == 'new':
        train_csv = train_new_csv
        train_path = './input/train_new_images/'           #### NEED TO CHANGED IT!!!!!!!!!!!!!!!!!!!
    elif cfg.data_type == 'old':
        train_csv = train_old_csv
        train_path = './input/train_old_images/'
    elif cfg.data_type == 'new_old_mixed_ben_preprocessing':
        train_csv = pd.concat([train_new_csv, train_old_csv], axis=0)
        train_path = './input/train_mixed_BEN_preprocessing/'
    elif cfg.data_type == 'new_old_balanced':
        train_csv = pd.read_csv('./input/train_balanced.csv')
        train_path = './input/train_mixed_images/'
    test_csv = pd.read_csv('./input/test.csv')
    print('Train Size = {}'.format(len(train_csv)))
    print('Public Test Size = {}'.format(len(test_csv)))

    counts = train_csv['diagnosis'].value_counts()
    class_list = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferate']
    for i, x in enumerate(class_list):
        counts[x] = counts.pop(i)
        print("{:^12} - class examples: {:^6}".format(x, counts[x]))
    print("Training path: {}".format(train_path))
    # Data Processing


    ## SHUFFLE DATA
    skf = None
    # if cfg.valid_type == 'holdout':
    test_805 = pd.read_csv('./input/test_805.csv')
    test_805 = pd.concat([test_805, test_805])
    train_csv, valid_csv = train_test_split(train_csv, test_size=cfg.valid_size,  shuffle=True,
                                                                   random_state=cfg.seed, stratify=train_csv['diagnosis'])
    # train_csv, valid_csv = train_test_split(train_csv, test_size=cfg.valid_size,  shuffle=True, random_state=cfg.seed, stratify=train_csv['diagnosis'])
    train_data = CreateDataset(df_data=train_csv, data_dir=train_path, transform=transforms_train)
    valid_data = CreateDataset(df_data=valid_csv, data_dir=train_path, transform=transforms_valid) # need to change It!!!!!
    # elif cfg.valid_type == 'cv':
    #     skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)



    # obtain training indices that will be used for validation


    # Create Samplers


    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_data, batch_size=cfg.batch_size,
                              pin_memory=True,
                              num_workers=cfg.num_workers,
                              shuffle=True,
                              worker_init_fn=__init_fn)

    valid_loader = DataLoader(valid_data, batch_size=cfg.batch_size,
                              pin_memory=True,
                              num_workers=cfg.num_workers,
                              shuffle=True,
                              worker_init_fn=__init_fn)

    # Model
    cfg.model.load_state_dict(
        torch.load('./Model_weights_finetuning/finetune18_like17_but_with_diffcoefs_colorjitter.pth')['model'])

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        cfg.model = cfg.model.cuda()

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

    loggers_list[1].add_data('Experiment N: {}'.format(len(loggers_list[0].logsFile)-1), '')
    loggers_list[1].add_data(info, '')
    add_data_to_loggers(loggers_list, 'date', datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S"))

    add_data_to_loggers(loggers_list, 'data-type', cfg.data_type)
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
    add_data_to_loggers(loggers_list, 'lb-kappa-score', np.nan)

    loggers_list[0].add_data('cfg', open('config.py', 'r+').read())
    loggers_list[0].add_data('dataset', open('train_dataset.py', 'r+').read())
    loggers_list[0].add_data('trainloop', open('training.py', 'r+').read())



    ## PRINT OUTPUT FREQUENCY
    print_frequency = cfg.print_frequency
    start_full_time = time.time()

    for epoch in range(1, cfg.n_epochs + 1):
        # For timing
        # loggers_list[0].open()
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
                pass
                data, target = data.cuda(), target.cuda().float()
            target = target.view(-1, 1)
            # clear the gradients of all optimized variables
            cfg.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward pass: compute predicted outputs by passing inputs to the cfg.model
                output = cfg.model(data)
                # calculate the batch loss
                loss = cfg.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to cfg.model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                cfg.optimizer.step()
                # data = data.cpu()
                train_loss_batch.append(loss.item())
                if batch_n % print_frequency == (print_frequency-1):
                    print('Train loss on {} batch: {:.6f}'.format(batch_n+1, np.mean(train_loss_batch)))
                    loggers_list[1].add_data(None, 'Train loss on {} batch: {:.6f}'.format(batch_n+1, np.mean(train_loss_batch)))
                    train_loss_epoch.append(np.mean(train_loss_batch))
                    train_loss_batch = []
        train_loss_epoch.append(np.mean(train_loss_batch))
        torch.cuda.empty_cache()
        ######################
        # validate the cfg.model #
        ######################
        cfg.model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                pass
                data, target = data.cuda(), target.cuda().float()
            # forward pass: compute predicted outputs by passing inputs to the cfg.model
            target = target.view(-1, 1)
            with torch.no_grad():
                output = cfg.model(data)
                # calculate the batch loss
                loss = cfg.criterion(output, target)
            # loss = loss.cpu()
            # update average validation loss
            valid_loss_epoch.append(loss.item())
            y_actual = target.data.cpu().numpy()
            y_pred = output[:, -1].detach().cpu().numpy()
            val_kappa.append(cohen_kappa_score(y_actual, y_pred.round(), weights='quadratic'))

            # calculate average losses
        train_loss_epoch = np.mean(train_loss_epoch)
        valid_loss_epoch = np.mean(valid_loss_epoch)
        valid_kappa = np.nanmean(val_kappa)
        kappa_epoch.append(valid_kappa)
        train_losses.append(train_loss_epoch)
        valid_losses.append(valid_loss_epoch)

        ## SCHEDULER STEP
        if cfg.scheduler is not None:
            if cfg.early_stopping_loss == 'pytorch':
                cfg.scheduler.step(valid_loss_epoch)
            elif cfg.early_stopping_loss == 'kappa':
                cfg.scheduler.step(1-valid_kappa)

        ## LOGGINS LOSSES
        if cfg.early_stopping_loss == 'pytorch':
            if valid_loss_best > valid_loss_epoch:
                valid_loss_best  = valid_loss_epoch
                train_loss_best = train_loss_epoch
                kappa_best = valid_kappa
                add_data_to_loggers(loggers_list, 'best-train-loss', '{:.6f}'.format(train_loss_best))
                add_data_to_loggers(loggers_list, 'best-valid-loss', '{:.6f}'.format(valid_loss_best))
                add_data_to_loggers(loggers_list, 'best-kappa', '{:.4f}'.format(kappa_best))
        elif cfg.early_stopping_loss == 'kappa':
            if kappa_best < valid_kappa:
                valid_loss_best  = valid_loss_epoch
                train_loss_best = train_loss_epoch
                kappa_best = valid_kappa
                add_data_to_loggers(loggers_list, 'best-train-loss', '{:.6f}'.format(train_loss_best))
                add_data_to_loggers(loggers_list, 'best-valid-loss', '{:.6f}'.format(valid_loss_best))
                add_data_to_loggers(loggers_list, 'best-kappa', '{:.4f}'.format(kappa_best))

        # print training/validation statistics
        print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f} | LR: {:.6f} | Estimated time: {:.2f}'.format(
            epoch, train_loss_epoch, valid_loss_epoch, valid_kappa, cfg.optimizer.param_groups[0]['lr'], time.time() - start_epoch_time))
        loggers_list[1].add_data('', 'Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f} | LR: {:.6f} | Estimated time: {:.2f}'.format(
            epoch, train_loss_epoch, valid_loss_epoch, valid_kappa, cfg.optimizer.param_groups[0]['lr'], time.time() - start_epoch_time))

        ##################
        # Early Stopping #
        ##################
        if cfg.early_stopping_loss == 'pytorch':
            cfg.early_stopping(valid_loss_epoch, model_params_list=cfg.model_param_list, experiment_name=cfg.weights_dir + cfg.experiment_name, epoch=epoch)
        elif cfg.early_stopping_loss == 'kappa':
            cfg.early_stopping(1 - valid_kappa, model_params_list=cfg.model_param_list,
                               experiment_name=cfg.weights_dir + cfg.experiment_name, epoch=epoch)
        if cfg.early_stopping.early_stop:
            add_data_to_loggers(loggers_list, 'time_estimated', '{:.2f}'.format(time.time() - start_full_time))
            add_data_to_loggers(loggers_list, 'n-epochs', epoch)
            loggers_list[0].save()
            loggers_list[1].save()
            break
        loggers_list[0].save()
        loggers_list[1].save()

    loggers_list[1].open()
    add_data_to_loggers(loggers_list, 'time_estimated', '{:.2f}'.format(time.time() - start_full_time))
    loggers_list[0].save()
    loggers_list[1].save()
    del cfg.model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    batch_size_list = [16]
    lr_list = [1e-3]
    p_horizontalflip_list = [0.4]
    model_type_list = ['efficientnet-b5']
    for batch_size in batch_size_list:
        for lr in lr_list:
            for p_horizontalflip in p_horizontalflip_list:
                for model_type in model_type_list:
                    info = "\n\n\nEXPERIMENT WITH BATCH_SIZE: {}, LR: {}, p_horizontalflip: {}, model_type: {}\n\n\n".format(batch_size, lr, p_horizontalflip, model_type)
                    print(info)
                    main(batch_size, lr, p_horizontalflip, model_type, info)
