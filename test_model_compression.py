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
from test_layer_similarity import cca, cka, gram_linear
# from log import logger
from torch.autograd import Variable
# from wasserstein_ensemble import get_network_from_param_list

def get_number_of_neurons(network):
    """
    Get number of neurons of each hidden layer in MLPNet

    :param netwokrs: a network
    """
    n_neurons = []

    for _, layer_weight in network.named_parameters():
        n_neurons.append(layer_weight.size(0))

    return np.array(n_neurons)[:-1]

    return np.array(n_neurons)[:-1]


def get_weight_matrices(network):
    """
    Get weights of each hidden layer in MLPNet

    :param netwokrs: a network
    """
    model_weights = []

    for _, layer_weight in network.named_parameters():
        model_weights.append(layer_weight)

    return model_weights


def get_activation_matrices(args, networks, personal_dataset=None, config=None, is_wd=False):
    """
    Get activation matrix for each layer of each network

    :param args: config parameters
    :param networks: list of networks
    :param personal_dataset: personalized dataset
    :param config: hyperparameters for CNNs, default = None
    :param is_wd: whether the cost is Wassersten distance
    :return: list of activation matrices for each model
    """
    activations = utils.get_model_activations(args, networks, personal_dataset=personal_dataset, config=config)
    list_act = []

    for _, model_dict in activations.items():
        model_act = []

        for _, layer_act in model_dict.items():
            if is_wd:
                reorder_dim = [l for l in range(2, len(layer_act.shape))]
                reorder_dim.extend([0, 1])
                layer_act = layer_act.permute(*reorder_dim).contiguous()
            layer_act = layer_act.view(layer_act.size(0), -1)
            model_act.append(layer_act)

        # exclude the activation of output layer
        list_act.append(model_act[:-1])

    return list_act


def get_wasserstein_distance(a, b, args):
    mu = np.ones(len(a)) / len(a)
    nu = np.ones(len(b)) / len(b)
    ground_metric_object = GroundMetric(args)
    print(f"{a.size()}, {b.size()}")
    M = ground_metric_object.process(a, b)
    M_cpu = M.data.cpu().numpy()

    return ot.emd2(mu, nu, M_cpu)


def get_cost(a, b, args, layer_metric):
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
            print(f"Shape weights of layer {i}: ", x[i].shape)
            cost_matrix[i][j] = get_cost(x[i], x[j], args, layer_metric)

    return cost_matrix


def get_cost_matrix_conv_layer(x, model_name, args, dissimilarity_matrix):
    """
    Compute the cost matrix between two measures of convolutional layers
    
    :param x: list of measures, size m
    :param args: config parameters
    :return: cost matrix, size m x m
    """
    assert "vgg" in model_name
    layer_idx = []
    
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
            dissimilarity_matrix[layer_idx[idx] : layer_idx[idx + 1], layer_idx[idx] : layer_idx[idx + 1]] = cost_matrix
            print("Updating dissimilarity_matrix: ", dissimilarity_matrix)

    return dissimilarity_matrix
  

def get_dissimilarity_matrix(args, networks, num_layers, model_names, personal_dataset=None):
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
    dissimilarity_matrix = np.full((num_layers[0], num_layers[0]), np.inf)
    print("dissimilarity_matrix: ", dissimilarity_matrix)
    # get layer representation of models (check x, y for layers within model)
    layer_representations = []
    if args.layer_measure == "index":
        x = np.arange(0, num_layers[0])
        y = np.arange(0, num_layers[1])
        # layer_representations.append(layer_representation)
        assert args.layer_metric == "euclidean"
    elif args.layer_measure == "neuron":
        x = get_number_of_neurons(networks[0])
        y = get_number_of_neurons(networks[1])
        assert args.layer_metric == "euclidean"
    elif args.layer_measure == "weight":
        x = get_weight_matrices(networks[0])
        y = get_weight_matrices(networks[1])
        assert args.layer_metric in ["cka", "cca", "wd"]
    elif args.layer_measure == "activation":
        # act_st_time = time.perf_counter()
        is_wd = args.layer_metric == "wd"
        x, y = get_activation_matrices(args, networks, personal_dataset=personal_dataset, config=args.config, is_wd=is_wd)
        # act_end_time = time.perf_counter()
        # act_time = act_end_time - act_st_time
        assert args.layer_metric in ["cka", "cca", "wd"]

    # separate where is the FC layer start
    classifier_idx = [0, 0]
    for i in range(2):
        for idx, (_, layer_weight) in enumerate(networks[i].named_parameters()):
            if len(layer_weight.shape) == 2:
                break
    classifier_idx[i] = idx
    print(f"FC layers of model {i} start from {idx}")
    
    # get dissimilarity matrix among layers of model 0
    if classifier_idx[0] > 0:
        if "vgg" in model_names[0]:
            dissimilarity_matrix = get_cost_matrix_conv_layer(x[: classifier_idx[0]], model_names[0], args, dissimilarity_matrix)
        # elif "resnet" in model_names[0]:
        #     dissimilarity_matrix = get_cost_matrix_resnet_block(x[: classifier_idx[0]], model_names[0], args, dissimilarity_matrix)
        else:
            raise NotImplementedError

        if classifier_idx[0] < len(x):
            dissimilarity_matrix = get_cost_matrix(x[classifier_idx[0] :], args)
            print("Cost matrix between layers {}-{} of model 0 is \n{}".format(classifier_idx[0], len(x), dissimilarity_matrix))
    else:
        print(x[classifier_idx[0] :])
        dissimilarity_matrix = get_cost_matrix(x[classifier_idx[0] :], args)
        print("Cost matrix between layers {}-{} of model 0 is \n{}".format(classifier_idx[0], len(x), dissimilarity_matrix))

    print("Optimal map from model 1 to model 0 is {}".format(dissimilarity_matrix))

    return dissimilarity_matrix


def get_dissimilarity_matrix1(args, networks, num_layers, model_names):
    dissimilarity_matrix = np.full((num_layers[0], num_layers[0]), np.inf)
    
    ground_metric_object = GroundMetric(args)
    for idx0, (layer_name0, layer_weight0) in networks[0].named_parameters():
        for idx1, (layer_name1, layer_weight1) in networks[0].named_parameters():
            if idx1 <= idx0:
                break
            layer_weight_data0 = layer_weight0.data
            layer_weight_data1 = layer_weight1.data
            
            M = ground_metric_object.process(layer_weight_data0.view(layer_weight_data0.shape[0], -1),
                                             layer_weight_data1.view(layer_weight_data1.shape[0], -1))
            cpuM = M.data.cpu().numpy()
            
            cost = ot.emd2(mu, nu, cpuM)
            print(f"Cost between layer {idx0} and layer {idx1} is: ", cost)
            
            dissimilarity_matrix[idx0, idx1] = cost
            print("Dissimilarity matrix among layers: ", dissimilarity_matrix)
    return dissimilarity_matrix


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

    dissimilarity_matrix = get_dissimilarity_matrix1(args, networks, num_layers, model_names)
  
    return args, networks, accuracies, num_layers, model_names



  
  

  
