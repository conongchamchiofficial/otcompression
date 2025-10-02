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
