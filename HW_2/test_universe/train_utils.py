# %load train_utils.py
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import sys
from IPython.display import clear_output

def _epoch(network, loss, loader,
           backward=True,
           optimizer=None,
           device='cpu',
           ravel_init=False,
           need_accuracies=False):
    losses = []
    accuracies = []
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        if ravel_init:
            X = X.view(X.size(0), -1)
        network.zero_grad()
        prediction = network(X)
        loss_batch = loss(prediction, y)
        losses.append(loss_batch.cpu().item())
        if need_accuracies:
            accuracies.append(np.mean((prediction.argmax(1) == y).numpy()))
        if backward:
            loss_batch.backward()
            optimizer.step()
    if not need_accuracies:
        return losses
    else:
        return losses, accuracies

# Trains neural network and shows plots
# network - object of torch.nn.Sequential
# train(test)_loader - objects of torch.utils.data.DataLoader with train and test set
# epochs - count of epochs
# loss - loss function from torch.nn
# learning_rate - actual only for default optimizer - Adam
# optimizer - custom optimizer if need, default - Adam with learning_rate
# ravel_init - X will be view as (X.size(0), -1)
# device - train on cpu or gpu
# tolerate_keyboard_interrupt - do not throw exception on "stop", only end train
def train(network, train_loader, test_loader,
          epochs, loss=nn.NLLLoss(), learning_rate=0.1,
          optimizer=None,
          ravel_init=False, device='cpu', tolerate_keyboard_interrupt=True,
          need_accuracies=False):
    if optimizer is None:
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    
    train_loss_epochs = []
    test_loss_epochs = []
    train_acc_epochs = []
    test_acc_epochs = []
    network = network.to(device)
    try:
        for epoch in range(epochs):
            network.train()
            pack = _epoch(network,
                                        loss,
                                        train_loader,
                                        True,
                                        optimizer,
                                        device,
                                        ravel_init,
                                        need_accuracies)
            if need_accuracies:
                losses, accuracies = pack
                train_acc_epochs.append(np.mean(accuracies))
            else:
                losses = pack
            train_loss_epochs.append(np.mean(losses))
            
            network.eval()
            pack = _epoch(network,
                                        loss,
                                        test_loader,
                                        False,
                                        optimizer,
                                        device,
                                        ravel_init,
                                        need_accuracies)
            if need_accuracies:
                losses, accuracies = pack
                test_acc_epochs.append(np.mean(accuracies))
            else:
                losses = pack

            test_loss_epochs.append(np.mean(losses))
            clear_output(True)
            print('Epoch {0}... (Train/Test) Loss: {1:.3f}/{2:.3f}\t'.format(
                        epoch, train_loss_epochs[-1], test_loss_epochs[-1]))
            plt.figure(figsize=(12, 5))
            plt.plot(train_loss_epochs, label='Train')
            plt.plot(test_loss_epochs, label='Test')
            plt.xlabel('Epochs', fontsize=16)
            plt.ylabel('Loss', fontsize=16)
            plt.legend(loc=0, fontsize=16)
            plt.grid()
            
            if need_accuracies:
                print('Epoch {0}... (Train/Test) Accuracies: {1:.3f}/{2:.3f}\t'.format(
                epoch, test_acc_epochs[-1], test_acc_epochs[-1]))
                plt.figure(figsize=(12, 5))
                plt.plot(train_acc_epochs, label='Train')
                plt.plot(test_acc_epochs, label='Test')
                plt.xlabel('Epochs', fontsize=16)
                plt.ylabel('Accuracy', fontsize=16)
                plt.legend(loc=0, fontsize=16)
                plt.grid()
            plt.show()
    except KeyboardInterrupt:
        if tolerate_keyboard_interrupt:
            pass
        else:
            raise KeyboardInterrupt
    return train_loss_epochs, \
           test_loss_epochs
