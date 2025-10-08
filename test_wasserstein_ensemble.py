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

    print(networks)
    
    avg_aligned_layers = []
    T_var = None
    previous_layer_shape = None
    ground_metric_object = GroundMetric(args)

    if args.eval_aligned:
        model0_aligned_layers = []

    if args.gpu_id==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu_id))

    print("Networks: ", networks[0].parameters())

    num_layers = len(list(zip(networks[0].parameters(), networks[1].parameters())))
    print("Num layers: ", num_layers)

    for idx0, (layer0_name, fc_layer0_weight) in enumerate(networks[0].named_parameters()):
        print("layer0_name: ", layer0_name)
        
        layer0_shape = fc_layer0_weight.shape
        if len(layer0_shape) > 2:
            is_layer0_conv = True
            # For convolutional layers, it is (#out_channels, #in_channels, height, width)
            fc_layer0_weight_data = fc_layer0_weight.data.view(fc_layer0_weight.shape[0], fc_layer0_weight.shape[1], -1)
        else:
            is_layer0_conv = False
            fc_layer0_weight_data = fc_layer0_weight.data
            
        for idx1, (layer1_name, fc_layer1_weight) in enumerate(networks[0].named_parameters()):
            print("layer1_name: ", layer1_name)
            
            layer1_shape = fc_layer1_weight.shape
            # print(layer1_shape)
            if len(layer1_shape) > 2:
                is_layer1_conv = True
                # For convolutional layers, it is (#out_channels, #in_channels, height, width)
                fc_layer1_weight_data = fc_layer1_weight.data.view(fc_layer1_weight.shape[0], fc_layer1_weight.shape[1], -1)
            else:
                is_layer1_conv = False
                fc_layer1_weight_data = fc_layer1_weight.data      

            if idx0 <= idx1:
                break
            if is_layer0_conv =! is_layer1_conv:
                break

            if is_layer0_conv:
                M = ground_metric_object.process(fc_layer0_weight_data.view(fc_layer0_weight_data.shape[0], -1),
                                fc_layer1_weight_data.view(fc_layer1_weight_data.shape[0], -1))
            else:
                M = ground_metric_object.process(fc_layer0_weight_data, fc_layer1_weight_data)
            print("M: ", M)
                
    return avg_aligned_layers

def get_pairwise_similarity_score(args, networks, activations=None, eps=1e-7, test_loader=None):
    pass







