# https://github.com/hsgser/clafusion/tree/905016742729b6846905e1f8c780bb43562b1e9b

import random
import sys
from test_basic_config import PATH_TO_CIFAR
sys.path.append(PATH_TO_CIFAR)
import time

import numpy as np
import ot
import routines
import torch
import torch.nn.functional as F
import train as cifar_train
import utils
from data import get_dataloader
from ground_metric import GroundMetric
from test_layer_similarity import cca, cka, gram_linear
from test_log import logger
from torch.autograd import Variable
from wasserstein_ensemble import get_network_from_param_list


vgg_cfg = {
    "vgg8": [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"],
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg11_quad": [64, "M", 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "vgg11_doub": [64, "M", 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "vgg11_half": [64, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13_quad": [64, 256, "M", 512, 512, "M", 1024, 1024, "M", 2048, 2048, "M", 2048, 512, "M"],
    "vgg13_doub": [64, 128, "M", 256, 256, "M", 512, 512, "M", 1024, 1024, "M", 1024, 512, "M"],
    "vgg13_half": [64, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

resnet_cfg = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}


def get_wasserstein_distance(a, b, args):
    mu = np.ones(len(a)) / len(a)
    nu = np.ones(len(b)) / len(b)
    ground_metric_object = GroundMetric(args)
    logger.info(f"{a.size()}, {b.size()}")
    M = ground_metric_object.process(a, b)
    M_cpu = M.data.cpu().numpy()

    return ot.emd2(mu, nu, M_cpu)
    
def get_cost(a, b, args):
    # check the form of eucli/cosin output in matrix, check 1- of sim or dissim of cca cka wd
    if args.similarity_type == "euclidean":
        return (a - b) ** 2
    elif args.similarity_type == "cca":
        return 1 - cca(a, b)
    elif args.similarity_type == "cka":
        return 1 - cka(gram_linear(a), gram_linear(b))
    elif args.similarity_type == "wd":
        return get_wasserstein_distance(a, b, args)
    elif args.similarity_type == "cosine":
        return F.normalize(a, dim=0) @ F.normalize(b, dim=0).T
    else:
        raise NotImplementedError

def get_cost_matrix(x, y, args):
    """
    Compute the cost matrix between two measures.

    :param x: list of measures, size m
    :param y: list of measures, size n
    :param args: config parameters
    :return: cost matrix, size m x n
    """

    cost = args.layer_metric
    m, n = len(x), len(y)
    if m * n == 0:
        return []
    C = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            C[i][j] = get_cost(x[i], y[j], args)

    return C

def get_top_k_layers(args, networks, num_layers):
    """
    Choose layers to be fused.

    :param args: config parameters
    :param network: the model
    :return: 
    """
    
    return 






