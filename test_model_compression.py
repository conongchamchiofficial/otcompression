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
from wasserstein_ensemble import get_network_from_param_list

def get_number_of_neurons(network):
    """
    Get number of neurons of each hidden layer in MLPNet

    :param netwokrs: a network
    """
    n_neurons = []

    for _, layer_weight in network.named_parameters():
        n_neurons.append(layer_weight.size(0))

    return np.array(n_neurons)[:-1]


def get_weight_matrices(network):
    """
    Get weights of each hidden layer in MLPNet

    :param netwokrs: a network
    """
    model_weights = []

    for _, layer_weight in network.named_parameters():
        model_weights.append(layer_weight)
    
    return  model_weights[:-1]


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
            print("before layer_act.shape: ", layer_act.shape)
            if is_wd:
                reorder_dim = [l for l in range(2, len(layer_act.shape))]
                reorder_dim.extend([0, 1])
                layer_act = layer_act.permute(*reorder_dim).contiguous()
            layer_act = layer_act.view(layer_act.size(0), -1)
            print("layer_act.shape: ", layer_act.shape)
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
        for j in range(i + 1, m):
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

    return dissimilarity_matrix


def get_histogram(args, idx, cardinality, layer_name, activations=None, return_numpy = True, float64=False):
    if activations is None:
        # returns a uniform measure
        if not args.unbalanced:
            print("returns a uniform measure of cardinality: ", cardinality)
            return np.ones(cardinality)/cardinality
        else:
            return np.ones(cardinality)
    else:
        # return softmax over the activations raised to a temperature
        # layer_name is like 'fc1.weight', while activations only contains 'fc1'
        print(activations[idx].keys())
        unnormalized_weights = activations[idx][layer_name.split('.')[0]]
        print("For layer {}, shape  of unnormalized weights is ".format(layer_name), unnormalized_weights.shape)
        unnormalized_weights = unnormalized_weights.squeeze()
        assert unnormalized_weights.shape[0] == cardinality

        if return_numpy:
            if float64:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy().astype(
                    np.float64)
            else:
                return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0).data.cpu().numpy()
        else:
            return torch.softmax(unnormalized_weights / args.softmax_temperature, dim=0)


def get_layer_representation(args, networks, num_layers, model_names, personal_dataset=None):
    # get layer representation of models (check x, y for layers within model)
    layer_representations = []
    if args.layer_measure == "index":
        x = np.arange(0, num_layers[0] - 1)
        y = np.arange(0, num_layers[1] - 1)
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
    return x, y


def get_dissimilarity_matrix(args, networks, num_layers, model_names, x):
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

    # initialize dissmilarity matrix among hidden layers
    dissimilarity_matrix = np.full((num_layers[0] - 1, num_layers[0] - 1), np.inf)
    
    # separate where is the FC layer start
    classifier_idx = [0, 0]
    for i in range(2):
        for idx, (_, layer_weight) in enumerate(networks[i].named_parameters()):
            if len(layer_weight.shape) == 2:
                break
    classifier_idx[i] = idx
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
            #print("Cost matrix between layers {}-{} of model 0 is \n{}".format(classifier_idx[0], len(x), dissimilarity_matrix))
    else:
        #print(x[classifier_idx[0] :])
        dissimilarity_matrix = get_cost_matrix(x[classifier_idx[0] :], args)
        #print("Cost matrix between layers {}-{} of model 0 is \n{}".format(classifier_idx[0], len(x), dissimilarity_matrix))
    print("Dissimilarity matrix among layers of model 0 is {}".format(dissimilarity_matrix))
    
    return dissimilarity_matrix


def find_min_position(matrix):
    """
    Finds the position (row and column indices) of the minimum value in a list of lists.
    """
    min_value = float('inf')  # Initialize with a very large value
    min_row_index = -1
    min_col_index = -1

    for row_index, inner_list in enumerate(matrix):
        for col_index, value in enumerate(inner_list):
            if value < min_value:
                min_value = value
                min_row_index = row_index
                min_col_index = col_index   

    return min_row_index, min_col_index, min_value


def choose_layers_to_merge(args, network0, num_layer0, dissimilarity_matrix, desired_number_of_hidden_layers):
    """
    Choose top-k layers to merge in large model

    :param 
    """
    # Choose groups of layers that needed to be compressed
    I = []
    inner_group = []
    num_hidden_layer = num_layer0 - 1
    num_merged_grp = 0
    if args.compression_only and desired_number_of_hidden_layers <= num_hidden_layer:
        while num_merged_grp < (num_hidden_layer - desired_number_of_hidden_layers):
            min_row_index, min_col_index, min_value = find_min_position(dissimilarity_matrix)
            print("find_min_position: ",  min_row_index, min_col_index, min_value)
            for idx, grp in enumerate(I):
                if min_row_index == grp[-1]:
                    if min_col_index - min_row_index == 1:
                        I[idx].append(min_col_index)
                        for i in range(min_row_index + 1):
                            for j in range(min_col_index, num_hidden_layer):
                                dissimilarity_matrix[i][j] = float('inf')
                        num_merged_grp += min_col_index - min_row_index
                        inner_group = []
                    elif min_col_index - min_row_index > 1:
                        I[idx].append(layer for layer in range(min_row_index + 1, min_col_index + 1))
                        for i in range(min_col_index):
                            for j in range(min_row_index, num_hidden_layer):
                                dissimilarity_matrix[i][j] = float('inf')
                        num_merged_grp += min_col_index - min_row_index
                        inner_group = []
                    min_value = float('inf')  # Initialize with a very large value
                    min_row_index = -1
                    min_col_index = -1
                    break
                elif min_col_index == grp[0]:
                    if min_col_index - min_row_index == 1:
                        I[idx].insert(0, min_row_index)
                        for i in range(min_row_index + 1):
                            for j in range(min_col_index, num_hidden_layer):
                                dissimilarity_matrix[i][j] = float('inf')
                        num_merged_grp += min_col_index - min_row_index
                        inner_group = []
                    elif min_col_index - min_row_index > 1:
                        I[idx].insert(0, (layer for layer in range(min_row_index, min_col_index)))
                        for i in range(min_col_index):
                            for j in range(min_row_index, num_hidden_layer):
                                dissimilarity_matrix[i][j] = float('inf')
                        num_merged_grp += min_col_index - min_row_index   
                        inner_group = []
                    min_value = float('inf')  # Initialize with a very large value
                    min_row_index = -1
                    min_col_index = -1
                    break
                    
            if min_col_index - min_row_index == 1:
                inner_group = [layer for layer in range(min_row_index, min_col_index + 1)]
                I.append(inner_group)
                for i in range(min_row_index + 1):
                    for j in range(min_col_index, num_hidden_layer):
                        dissimilarity_matrix[i][j] = float('inf')
                num_merged_grp += min_col_index - min_row_index
                inner_group = []
            elif min_col_index - min_row_index > 1:
                inner_group.append(range(min_row_index, min_col_index + 1))
                I.append(inner_group)
                for i in range(min_col_index):
                    for j in range(min_row_index, num_hidden_layer):
                        dissimilarity_matrix[i][j] = float('inf')
                num_merged_grp += min_col_index - min_row_index
                inner_group = []
            min_value = float('inf')  # Initialize with a very large value
            min_row_index = -1
            min_col_index = -1
            print("I: ", I)
    else:
        raise NotImplementedError

    # Add the rest individual layers
    flat_I = [layer for group in I for layer in group]
    for i in range(num_layer0):
        if i not in flat_I:
            I.append([i])
    I.sort()
    return I    


def approximate_relu(act_mat, num_columns, args, method):
    """
    Approximate ReLU activation function by a diagonal matrix

    :param act_mat: the pre-activation matrix, i.e. before applying ReLU
    :param num_columns: the number of nodes in the previous layer
    :param args: config parameters
    :param method: method to approximate the sign of activation ["sum", "majority", "avg"], default = "sum"
    :return: a matrix in which each row has the same value
    """
    if method == "sum":
        act_vec = act_mat.sum(axis=1) >= 0
    elif method == "majority":
        act_vec = (act_mat > 0).mean(axis=1) >= 0.5
    elif method == "avg":
        act_vec = ((act_mat > 0) * 1.0).mean(axis=1)
    else:
        raise NotImplementedError

    if isinstance(act_vec, torch.Tensor):
        return act_vec.unsqueeze(1).repeat(1, num_columns)
    else:
        return np.tile(act_vec, (1, num_columns))


def merge_layers(args, networks, num_layers, model_names, acts, I, method):
    """
    Merge consecutive layers in the larger model.

    :param args: config parameters
    :param networks: list of models
    :param num_layers:  list of number of layers
    :param model_names: list of model names
    :param acts: list of activation matrices for hidden layers
    :param I: groups of layers merged
    :param method: method to approximate the sign of activation ["sum", "majority"], default = "sum"
    
    :return: list of weight matrix of the new model and the updated args
    """
    args.fused_model_name = model_names[1]
    new_weights = []
    network_params = list(networks[0].named_parameters())
    
    if args.dataset == "mnist":
        input_dim = 784
    elif args.dataset == "cifar10":
        input_dim = 3072
    else:
        raise ValueError
    pre_weight = torch.eye(input_dim).cuda(args.gpu_id)
    
    for idx_grp, grp in enumerate(I):
        for idx_layer, layer in enumerate(grp):
            (_, layer_weight) = network_params[layer]
            if layer != grp[-1]:
                print(f"Merge layer {layer} with {grp[idx_layer + 1]}")
                print("Approximate ReLU at hidden layer {} with activation of shape {}".format(layer + 1, acts[layer].shape))
                act_vec = approximate_relu(acts[layer], layer_weight.shape[1], args, method)
                print("act_vec.shape == layer_weight.shape", act_vec.shape, layer_weight.shape)
                assert act_vec.shape == layer_weight.shape
                if not isinstance(act_vec, torch.Tensor):
                    act_vec = torch.from_numpy(act_vec).cuda(args.gpu_id)
                print("layer_weight = layer_weight * act_vec", layer_weight.shape, act_vec.shape)
                layer_weight = layer_weight * act_vec
                print("pre_weight = layer_weight @ pre_weight", layer_weight.shape, pre_weight.shape)
                pre_weight = layer_weight @ pre_weight
                print("pre_weight", pre_weight.shape)
            else:
                print(f"Main layer {layer}")
                print("pre_weight = layer_weight @ pre_weight", layer_weight.shape, pre_weight.shape)
                pre_weight = layer_weight @ pre_weight
                print("pre_weight", pre_weight.shape)
                setattr(args, "num_hidden_nodes" + str(len(new_weights) + 1), layer_weight.shape[0])
                new_weights.append(pre_weight)
                pre_weight = torch.eye(layer_weight.shape[0]).cuda(args.gpu_id)
    setattr(args, "num_hidden_layers", len(new_weights))

    if args.dataset == "mnist":
        _, test_loader = get_dataloader(args)
    else:
        _, test_loader = cifar_train.get_dataset(args.config, to_download=args.to_download)
        
    if args.parse_config:
        print("Change configuration from list of hidden_layer_sizes to num_hidden_layers/num_hidden_nodes style.")
        setattr(args, "parse_config", False)
        
    new_acc, new_network = get_network_from_param_list(args, new_weights, test_loader, model_name=args.fused_model_name)
    print(new_acc)

    return new_acc, new_network, args

def fuse_layer(args, networks, num_layers, model_names, acts, I, method):
    args.fused_model_name = model_names[1]
    new_weights = []
    network_params = list(networks[0].named_parameters())
    
    if args.dataset == "mnist":
        input_dim = 784
    elif args.dataset == "cifar10":
        input_dim = 3072
    else:
        raise ValueError

    for idx_grp, grp in enumerate(I):
        if len(grp) < 2:
            continue
        else:
            (_, layer_weight0) = network_params[grp[0]]
            (_, layer_weight1) = network_params[grp[1]]
    
    new_acc, new_network = get_network_from_param_list(args, new_weights, test_loader, model_name=args.fused_model_name)
    return new_acc, new_network, args


def compress_model(args, networks, accuracies, num_layers, model_names=None):
    """
    Compress deeper model to be the same size of swallower one
    
    :param args: config parameters
    :param networks: list of models
    :param accuracies: list of accuracies
    :param num_layers: list of num_layers
    :param model_names: list of model_names
    :return: updated large model, accuracy and config parameters
    """
    print("------ Construct dissimilarity matrix among layers in model 0 ------")
    config_param0, config_param1 = get_layer_representation(args, networks, num_layers, model_names)
    dissimilarity_matrix = get_dissimilarity_matrix(args, networks, num_layers, model_names, config_param0)

    print("------ Choose top-k layers to merge ------")
    I = choose_layers_to_merge(args, networks[0], num_layers[0], dissimilarity_matrix, num_layers[1] - 1)
    #I = [[0, 1], [2], [3], [4]]
    print(I)
    print("------ Model compression by merging layers via OT ------")
    new_acc, new_network, args = merge_layers(args, networks, num_layers, model_names, config_param0, I, method=args.relu_approx_method)
    networks[0] = new_network
    accuracies[0] = new_acc
    
    for idx, weight in enumerate(networks[1].parameters()):
        if (idx != 0) and (len(weight.shape) > 1):
            setattr(args, "num_hidden_nodes" + str(idx), weight.size(1))
    
    return args, networks, accuracies, num_layers, model_names



  
  

  
