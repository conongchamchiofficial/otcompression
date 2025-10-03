import parameters
from data import get_dataloader
import routines
import baseline
import wasserstein_ensemble
import os
import utils
import numpy as np
import sys
import torch


PATH_TO_CIFAR = "./cifar/"
sys.path.append(PATH_TO_CIFAR)
import train as cifar_train
from tensorboardX import SummaryWriter

# ------- Setting up parameters -------
# Setup parameters: def parameters.get_parameters()
    # parameters = {
    #     'n_epochs': 1,
    #     'enable_dropout': False,
    #     'batch_size_train': 128,
    #     'batch_size_test': 1000,
    #     'learning_rate': 0.01,
    #     'momentum': 0.5,
    #     'log_interval': 100,

    #     'to_download':True, # set to True if MNIST/dataset hasn't been downloaded,
    #     'disable_bias': True, # no bias at all in fc or conv layers,
    #     'dataset': 'Cifar10',
    #     # dataset: mnist,
    #     'num_models': 2,
    #     'model_name': 'vgg11_nobias',
    #     # model_name: net,
    #     # model_name: mlpnet,
    #     'num_hidden_nodes': 100,
    #     'num_hidden_nodes1': 400,
    #     'num_hidden_nodes2': 200,
    #     'num_hidden_nodes3': 100,
    # }
# Setup a timestamp
# Set rootdir and dump directories
# Load configuration: utils._get_config(args)

# ------- Loading pre-trained models -------
# ? import train as cifar_train ?
# Get dataset: cifar_train.get_dataset(config)
# Get pre-trained models: routines.get_pretrained_model(///)
# Model structure: in _make_layers [Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), AvgPool2d(kernel_size=1, stride=1, padding=0)]
# Model parameters: [torch.Size([64, 3, 3, 3]), torch.Size([128, 64, 3, 3]), torch.Size([256, 128, 3, 3]), torch.Size([256, 256, 3, 3]), torch.Size([512, 256, 3, 3]), torch.Size([512, 512, 3, 3]), torch.Size([512, 512, 3, 3]), torch.Size([512, 512, 3, 3]), torch.Size([10, 512])]
# Load model path, accuracy, epoch




