import ot
import torch
import numpy as np
import routines
from model import get_model_from_name
import utils
from ground_metric import GroundMetric
import math
import sys
import compute_activations

def geometric_ensembling_modularized(args, networks, train_loader, test_loader, activations=None):
    
    if args.geom_ensemble_type == 'wts':
        avg_aligned_layers = get_wassersteinized_layers_modularized(args, networks, activations, test_loader=test_loader)
    elif args.geom_ensemble_type == 'acts':
        #avg_aligned_layers = get_acts_wassersteinized_layers_modularized(args, networks, activations, train_loader=train_loader, test_loader=test_loader)
        pass
        
    return get_network_from_param_list(args, avg_aligned_layers, test_loader)

def get_network_from_param_list(args, param_list, test_loader):

    print("using independent method")
    new_network = get_model_from_name(args, idx=1)
    if args.gpu_id != -1:
        new_network = new_network.cuda(args.gpu_id)

    # check the test performance of the network before
    log_dict = {}
    log_dict['test_losses'] = []
    routines.test(args, new_network, test_loader, log_dict)

    # set the weights of the new network
    # print("before", new_network.state_dict())
    print("len of model parameters and avg aligned layers is ", len(list(new_network.parameters())),
          len(param_list))
    assert len(list(new_network.parameters())) == len(param_list)

    layer_idx = 0
    model_state_dict = new_network.state_dict()

    print("len of model_state_dict is ", len(model_state_dict.items()))
    print("len of param_list is ", len(param_list))

    for key, value in model_state_dict.items():
        model_state_dict[key] = param_list[layer_idx]
        layer_idx += 1

    new_network.load_state_dict(model_state_dict)

    # check the test performance of the network after
    log_dict = {}
    log_dict['test_losses'] = []
    acc = routines.test(args, new_network, test_loader, log_dict)

    return acc, new_network

def get_wassersteinized_layers_modularized(args, networks, activations=None, eps=1e-7, test_loader=None):
    '''
    Two neural networks that have to be averaged in geometric manner (i.e. layerwise).
    The 1st network is aligned with respect to the other via wasserstein distance.
    Also this assumes that all the layers are either fully connected or convolutional *(with no bias)*

    :param networks: list of networks
    :param activations: If not None, use it to build the activation histograms.
    Otherwise assumes uniform distribution over neurons in a layer.
    :return: list of layer weights 'wassersteinized'
    '''

    # simple_model_0, simple_model_1 = networks[0], networks[1]
    # simple_model_0 = get_trained_model(0, model='simplenet')
    # simple_model_1 = get_trained_model(1, model='simplenet')
    print(network)
    print(network[0])
    print(network[1])
  








