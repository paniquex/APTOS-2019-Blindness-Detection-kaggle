import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model_params_list, experiment_name, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_params_list, experiment_name, epoch)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_params_list, experiment_name, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_param_list, experiment_name, epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')


            model_full_state = {'model': model_param_list[0].state_dict()}
                                #'optim': model_param_list[1].state_dict()}

            # if model_param_list[2] is not None:
            #     model_full_state.update({'Scheduler' : model_param_list[2].state_dict()})
            # else:
            #     model_full_state.update({'Scheduler': None})

            torch.save(model_full_state, experiment_name + str(epoch))
            self.val_loss_min = val_loss
