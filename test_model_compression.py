import random
import sys
from basic_config import PATH_TO_CIFAR
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
from layer_similarity import cca, cka, gram_linear
# from log import logger
from torch.autograd import Variable
# from wasserstein_ensemble import get_network_from_param_list

def get_wasserstein_distance(a, b, args):
    mu = np.ones(len(a)) / len(a)
    nu = np.ones(len(b)) / len(b)
    ground_metric_object = GroundMetric(args)
    print(f"{a.size()}, {b.size()}")
    M = ground_metric_object.process(a, b)
    M_cpu = M.data.cpu().numpy()

    return ot.emd2(mu, nu, M_cpu)


def get_cost(x, y, args, layer_metric):
    if layer_metric == "euclidean":
        return (a - b) ** 2
    elif layer_metric == "cca":
        return 1 - cca(a, b)
    elif layer_metric == "cka":
        return 1 - cka(gram_linear(a), gram_linear(b))
    elif layer_metric == "wd":
        return get_wasserstein_distance(a, b, args)
    elif layer_metric == "cosine":
        return F.normalize(a, dim=0) @ F.normalize(b, dim=0).T
    else:
        raise NotImplementedError


def get_cost_matrix(x, args):
    """
    Compute the cost matrix for elements among one measure
    
    :param x: list of measures, size m
    :param args: config parameters
    :return: cost matrix, size m x m
    """
    
    layer_metric = args.layer_metric
    m = len(x)
    if m * m == 0:
        return []
  
    cost_matrix = np.full((m, m), np.inf)
    for i in range(m):
        for j in range(i, m):
            cost_matrix[i][j] = get_cost(x[i], y[j], args, layer_metric)

    return cost_matrix


def get_cost_matrix_conv_layer(x, y, model_name, args, dissimilarity_matrix):
    """
    Compute the cost matrix between two measures of convolutional layers
    
    :param x: list of measures, size m
    :param y: list of measures, size n
    :param args: config parameters
    :return: cost matrix, size m x n
    """
    assert "vgg" in model_name
    layer_idx = []
    cost_matrices = []
    
    model_config = vgg_cfg[name.split("_")[0]]
    idx = 0
    for layer_size in model_config:
        if layer_size == "M":
            layer_idx.append(idx)
        else:
            idx += 1
    layer_idx = [0] + layer_idx

    for idx in range(len(layer_idx) - 1):
        if layer_idx[idx + 1] - layer_idx[idx] > 1:
            cost_matrix = get_cost_matrix(x[layer_idx[idx] : layer_idx[idx + 1]], x[layer_idx[idx] : layer_idx[idx + 1]], args)
            cost_matrices.append(cost_matrix)

    for i in range 
    
  return dissimilarity_matrix
  


def get_dissimilarity_matrix(args, networks, num_layers, model_names):
    """
    Calculate dissimilarity index among layers in the large model
    
    :param args: config parameters
    :param networks: list of models
    :param num_layers: list of num_layers
    :param model_names: list of model_names
    :return: a dissimilarity matrix
    """
    # measure time
    # act_time = 0
    # align_time = 0
    # align_st_time = time.perf_counter()

    # initialize dissmilarity matrix
    dissimilarity_matrix = = np.full((num_layers[0], num_layers[0]), np.inf)
    
    # get layer representation of models (check x, y for layers within model)
    layer_representations = []
    if args.layer_measure == "index":
        layer_representation = np.arange(1, num_layers[0] - 1)
        layer_representations.append(layer_representation)
        assert args.layer_metric == "euclidean"
    elif args.layer_measure == "neuron":
        layer_representation = get_number_of_neurons(networks[0])
        layer_representations.append(layer_representation)
        assert args.layer_metric == "euclidean"
    elif args.layer_measure == "activation":
        # act_st_time = time.perf_counter()
        is_wd = args.layer_metric == "wd"
        x, y = get_activation_matrices(args, networks, personal_dataset=personal_dataset, config=args.config, is_wd=is_wd)
        # act_end_time = time.perf_counter()
        # act_time = act_end_time - act_st_time
        assert args.layer_metric in ["cka", "cca", "wd"]

    # separate where is the FC layer start
    classifier_idx = [None, None]
    for i in range(2):
        for idx, (_, layer_weight) in enumerate(networks[i].named_parameters()):
            if len(layer_weight.shape) == 2:
                break
    classifier_idx[i] = idx
    print(f"FC layers of model {i} start from {idx}")
    
    # get dissimilarity matrix among layers of model 0
    if classifier_idx[0] > 0:
        if "vgg" in model_names[0]:
            mat1 = align_conv_layers()
        elif "resnet" in model_names[0]:
            mat1 = align_resnet_block()
        else:
            raise NotImplementedError

    if classifier_idx[0] < len(x):
        mat2 = get_cost_matrix()
        print("Cost matrix between layers {}-{} of model 0 is \n{}".format(classifier_idx[0] + 1, len(x), classifier_idx[1] + 1, len(y), mat2")
        map2 = 
          
      

def compress_model(args, networks, accuracies, num_layers, model_names=None):
    """
    Compress deeper model to be the same size of smaller one
    
    :param args: config parameters
    :param networks: list of models
    :param accuracies: list of accuracies
    :param num_layers: list of num_layers
    :param model_names: list of model_names
    :return: updated large model, accuracy and config parameters
    """
    
    if num_layers[0] < num_layers[1]:
    print("Shuffle two models so that model 0 has more layers than model 1")
    networks = networks[::-1]
    accuracies = accuracies[::-1]
    num_layers = num_layers[::-1]
    model_names = model_names[::-1]

    print("------ Before compression ------")
    for i, network in enumerate(networks):
        print("Model {} has accuracy of {} with {} layers and parameters".format(i, accuracies[i],num_layers[i]))
        print(networks)

    dissimilarity_matrix = get_dissimilarity_matrix(args, networks, num_layers, model_names)
  
    return args, networks, accuracies, num_layers, model_names



  
  

  
