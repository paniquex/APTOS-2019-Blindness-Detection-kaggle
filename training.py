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

# User-defined modules
from train_dataset import transforms, CreateDataset
from model import MainModel
from logger import Logger


def add_data_to_loggers(loggers_list, column_name, data):
    loggers_list[0].add_data(column_name, data)
    loggers_list[1].add_data(column_name, data)


def main():
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

    # Set Batch Size
    batch_size = 4

    # Percentage of training set to use as validation
    valid_size = 0.2

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)

    # Model
    model = MainModel('Resnet').model

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')
        model.cuda()

    # Trainable Parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: \n{}".format(pytorch_total_params))

    #Training(Fine-Tuning) and Validation

    # specify loss function (categorical cross-entropy loss)
    criterion = nn.MSELoss()

    # specify optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0015)

    #specify scheduler
    scheduler = None

    #specify early stopping
    early_stopping = False
    early_stopping_patience = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # number of epochs to train the model
    N_EPOCHS = 35


    valid_loss_min = np.Inf

    # keeping track of losses as it happen
    train_losses = []
    valid_losses = []
    val_kappa = []
    test_accuracies = []
    valid_accuracies = []
    kappa_epoch = []

    # Loggers

    logger_df = Logger("LOGS.csv", mode = 'df')
    logger_txt = Logger("LOGS.txt", mode = 'txt')
    loggers_list = [logger_df, logger_txt]
    add_data_to_loggers(loggers_list, 'date', datetime.strftime(datetime.now(), "%Y.%m.%d %H:%M:%S"))
    add_data_to_loggers(loggers_list, 'data-type', '')
    add_data_to_loggers(loggers_list, 'net-architecture', "!!!Need to parse model file!!!")
    add_data_to_loggers(loggers_list, 'loss-func', str(criterion))
    add_data_to_loggers(loggers_list, 'optim', str(optimizer))
    if scheduler is not None:
        add_data_to_loggers(loggers_list, 'scheduler', str(scheduler))
    if early_stopping:
        add_data_to_loggers(loggers_list, 'early-stopping-patience', early_stopping_patience)
    else:
        add_data_to_loggers(loggers_list, 'early-stopping-patience', early_stopping)

    add_data_to_loggers(loggers_list, 'parameters-amount', pytorch_total_params)
    add_data_to_loggers(loggers_list, 'n-epochs', N_EPOCHS)
    add_data_to_loggers(loggers_list, 'batch-size', batch_size)

    train_loss_best = np.inf
    valid_loss_best = np.inf
    kappa_best = 0

    add_data_to_loggers(loggers_list, 'best-train-loss', train_loss_best)
    add_data_to_loggers(loggers_list, 'best-valid-loss', valid_loss_best)
    add_data_to_loggers(loggers_list, 'best-kappa', kappa_best)


    for epoch in range(1, N_EPOCHS + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        ###################
        # train the model #
        ###################
        model.train()
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cpu().float()
            target = target.view(-1, 1)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data).cpu()
                # calculate the batch loss
                loss = criterion(output, target).cpu()
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                data = data.cpu()
            # Update Train loss and accuracies
            train_loss += loss.item() * data.size(0)

        ######################
        # validate the model #
        ######################
        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda().float()
            # forward pass: compute predicted outputs by passing inputs to the model
            target = target.view(-1, 1)
            with torch.set_grad_enabled(True):
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
            # update average validation loss
            valid_loss += loss.item() * data.size(0)
            # output = output.cohen_kappa_score_kappa_score)
            y_actual = target.data.cpu().numpy()
            y_pred = output[:, -1].detach().cpu().numpy()
            val_kappa.append(cohen_kappa_score(y_actual, y_pred.round()))

            # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        valid_loss = valid_loss / len(valid_loader.sampler)
        valid_kappa = np.mean(val_kappa)
        kappa_epoch.append(np.mean(val_kappa))
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        # print training/validation statistics
        print('Epoch: {} | Training Loss: {:.6f} | Val. Loss: {:.6f} | Val. Kappa Score: {:.4f}'.format(
            epoch, train_loss, valid_loss, valid_kappa))

        ##################
        # Early Stopping #
        ##################
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'best_model.pt')
            valid_loss_min = valid_loss


if __name__ == '__main__':
    main()
