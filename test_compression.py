import parameters
from data import get_dataloader
import routines
import baseline
import test_wasserstein_ensemble
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
# Get dataset: def cifar_train.get_dataset(config)
# Get pre-trained models: def routines.get_pretrained_model(///)
# Model structure: in _make_layers [Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), ReLU(), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), AvgPool2d(kernel_size=1, stride=1, padding=0)]
# Model parameters: [torch.Size([64, 3, 3, 3]), torch.Size([128, 64, 3, 3]), torch.Size([256, 128, 3, 3]), torch.Size([256, 256, 3, 3]), torch.Size([512, 256, 3, 3]), torch.Size([512, 512, 3, 3]), torch.Size([512, 512, 3, 3]), torch.Size([512, 512, 3, 3]), torch.Size([10, 512])]
# Load model path, accuracy, epoch
# Recheck models' accuracy: def routines.test(///)
# Print layer-param: print(f'layer {name} has #params ', param.numel())
# Get model activation: utils.get_model_activations(///)

# ------- Geometric Ensembling -------
# def wasserstein_ensemble.geometric_ensembling_modularized(///)
# def get_wassersteinized_layers_modularized(///)
# Define ground metric: class GroundMetric(args)
# Check number layers of models: len(list(zip(networks[0].parameters(), networks[1].parameters())))
# for l=0..L layers, do:
    # Distinguish layer type: FC and Conv
    # if l==0:
        # Form ground metric: M = ground_metric_object.process(///)
            # Normalize weight: coordinates = self._normed_vecs(///)
            # Compute ground metric matrix (euclidean/cosine/angular) (pairwise distance): ground_metric_matrix = self.get_metric(///)
            # Normalize ground metric: ground_metric_matrix = self._normalize(///)
    # else:
        # Align incoming edge weight: aligned_wt = torch.bmm(///).permute(///)
        # Form ground metric: M = ground_metric_object.process(///)
    # Define histogram muy and nuy: mu = get_histogram(///)
    # Solve OT and compute transport map (emd/sinkhorn): T = ot.bregman.sinkhorn(///)
    # Fix transport map for marginals: if args.correction
    # print("Ratio of trace to the matrix sum: ")
    # Align model0's weight to model1's: t_fc0_model = torch.matmul(///)
    # Average aligned weights: geometric_fc = (///))/2        avg_aligned_layers.append(geometric_fc)
    # Evaluate performace of the update model (optinal): if args.eval_aligned
    
# ------- Prediction based ensembling -------
# ------- Naive ensembling of weights -------
# ----- Saved results at sample.csv ------
    # Result: {'exp_name': 'exp_2025-10-02_06-52-23_760615', 'model0_acc': 90.30999821424489, 'model1_acc': 90.4999980330467, 'geometric_acc': 85.98, 'prediction_acc': 91.34, 'naive_acc': 17.02, 'geometric_gain': -4.51999803304669, 'geometric_gain_%': -4.994473073243805, 'prediction_gain': 0.8400019669533094, 'prediction_gain_%': 0.9281789891824936, 'relative_loss_wrt_prediction': 5.922652062426298, 'geometric_time': 5.022159557000009}
    # Parameters: Namespace(n_epochs=300, batch_size_train=64, batch_size_test=1000, learning_rate=0.01, momentum=0.5, log_interval=100, to_download=False, disable_bias=True, dataset='Cifar10', num_models=2, model_name='vgg11_nobias', config_file=None, config_dir='/content/otfusion/otfusion/exp_sample/configurations', num_hidden_nodes=400, num_hidden_nodes1=400, num_hidden_nodes2=200, num_hidden_nodes3=100, num_hidden_nodes4=50, sweep_id=90, gpu_id=0, skip_last_layer=False, skip_last_layer_type='average', debug=False, cifar_style_data=False, activation_histograms=False, act_num_samples=100, softmax_temperature=1, activation_mode=None, options_type='generic', deprecated=None, save_result_file='sample.csv', sweep_name='exp_sample', reg=0.01, reg_m=0.001, ground_metric='euclidean', ground_metric_normalize='none', not_squared=True, clip_gm=False, clip_min=0, clip_max=5, tmap_stats=False, ensemble_step=0.5, ground_metric_eff=True, retrain=0, retrain_lr_decay=-1, retrain_lr_decay_factor=None, retrain_lr_decay_epochs=None, retrain_avg_only=False, retrain_geometric_only=False, load_models='./cifar_models/', ckpt_type='best', recheck_cifar=True, recheck_acc=False, eval_aligned=False, enable_dropout=False, dump_model=False, dump_final_models=False, correction=True, activation_seed=21, weight_stats=True, sinkhorn_type='normal', geom_ensemble_type='wts', act_bug=False, standardize_acts=False, transform_acts=False, center_acts=False, prelu_acts=True, pool_acts=False, pool_relu=False, normalize_acts=False, normalize_wts=True, gromov=False, gromov_loss='square_loss', tensorboard_root='./tensorboard', tensorboard=False, same_model=-1, dist_normalize=False, update_acts=False, past_correction=True, partial_reshape=False, choice='0 2 4 6 8', diff_init=False, partition_type='labels', personal_class_idx=9, partition_dataloader=-1, personal_split_frac=0.1, exact=True, skip_personal_idx=False, prediction_wts=False, width_ratio=1, proper_marginals=False, retrain_seed=-1, no_random_trainloaders=False, reinit_trainloaders=False, second_model_name=None, print_distances=False, deterministic=False, skip_retrain=-1, importance=None, unbalanced=False, temperature=20, alpha=0.7, dist_epochs=60, handle_skips=False, timestamp='2025-10-02_06-52-23_760615', rootdir='/content/otfusion/otfusion/exp_sample', baseroot='/content/otfusion/otfusion', result_dir='/content/otfusion/otfusion/exp_sample/results', exp_name='exp_2025-10-02_06-52-23_760615', csv_dir='/content/otfusion/otfusion/exp_sample/csv', config={'dataset': 'Cifar10', 'model': 'vgg11_nobias', 'optimizer': 'SGD', 'optimizer_decay_at_epochs': [30, 60, 90, 120, 150, 180, 210, 240, 270], 'optimizer_decay_with_factor': 2.0, 'optimizer_learning_rate': 0.05, 'optimizer_momentum': 0.9, 'optimizer_weight_decay': 0.0005, 'batch_size': 128, 'num_epochs': 300, 'seed': 42}, second_config={'dataset': 'Cifar10', 'model': 'vgg11_nobias', 'optimizer': 'SGD', 'optimizer_decay_at_epochs': [30, 60, 90, 120, 150, 180, 210, 240, 270], 'optimizer_decay_with_factor': 2.0, 'optimizer_learning_rate': 0.05, 'optimizer_momentum': 0.9, 'optimizer_weight_decay': 0.0005, 'batch_size': 128, 'num_epochs': 300, 'seed': 42}, cifar_init_lr=0.05, activation_time=2.1127999957570864e-05, params_model_0=9222848, params_model_1=9222848, geometric_time=5.022159557000009, params_geometric=9222848, **{'trace_sum_ratio_features.0.weight': 0.046875, 'trace_sum_ratio_features.3.weight': 0.0, 'trace_sum_ratio_features.6.weight': 0.0, 'trace_sum_ratio_features.8.weight': 0.0078125, 'trace_sum_ratio_features.11.weight': 0.001953125, 'trace_sum_ratio_features.13.weight': 0.001953125, 'trace_sum_ratio_features.16.weight': 0.001953125, 'trace_sum_ratio_features.18.weight': 0.005859374534338713, 'trace_sum_ratio_classifier.weight': 1.0})

if __name__ == '__main__':

    print("------- Setting up parameters -------")
    args = parameters.get_parameters()
    print("The parameters are: \n", args)

    if args.deterministic:
        # torch.backends.cudnn.enabled = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # loading configuration
    config, second_config = utils._get_config(args)
    args.config = config
    args.second_config = second_config

    # obtain trained models
    if args.load_models != '':
        print("------- Loading pre-trained models -------")

        # currently mnist is not supported!
        # assert args.dataset != 'mnist'

        # ensemble_experiment = "exp_2019-04-23_18-08-48/"
        # ensemble_experiment = "exp_2019-04-24_02-20-26"
        
        ensemble_experiment = args.load_models.split('/')
        if len(ensemble_experiment) > 1:
            # both the path and name of the experiment have been specified
            ensemble_dir = args.load_models
        elif len(ensemble_experiment) == 1:
            # otherwise append the directory before!
            ensemble_root_dir = "{}/{}_models/".format(args.baseroot, (args.dataset).lower())
            ensemble_dir = ensemble_root_dir + args.load_models

        # checkpoint_type = 'final'  # which checkpoint to use for ensembling (either of 'best' or 'final)

        if args.dataset=='mnist':
            train_loader, test_loader = get_dataloader(args)
            retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)
        elif args.dataset.lower()[0:7] == 'cifar10':
            args.cifar_init_lr = config['optimizer_learning_rate']
            if args.second_model_name is not None:
                assert second_config is not None
                assert args.cifar_init_lr == second_config['optimizer_learning_rate']
                # also the below things should be fine as it is just dataloader loading!
            print('loading {} dataloaders'.format(args.dataset.lower()))
            train_loader, test_loader = cifar_train.get_dataset(config)
            retrain_loader, _ = cifar_train.get_dataset(config, no_randomness=args.no_random_trainloaders)


        models = []
        accuracies = []

        for idx in range(args.num_models):
            print("loading model with idx {} and checkpoint_type is {}".format(idx, args.ckpt_type))

            if args.dataset.lower()[0:7] == 'cifar10' and (args.model_name.lower()[0:5] == 'vgg11' or args.model_name.lower()[0:6] == 'resnet'):
                if idx == 0:
                    config_used = config
                elif idx == 1:
                    config_used = second_config
                    
                model, accuracy = cifar_train.get_pretrained_model(
                        config_used, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)),
                        args.gpu_id, relu_inplace=not args.prelu_acts # if you want pre-relu acts, set relu_inplace to False
                )
            else:
                model, accuracy = routines.get_pretrained_model(
                        args, os.path.join(ensemble_dir, 'model_{}/{}.checkpoint'.format(idx, args.ckpt_type)), idx = idx
                )

            models.append(model)
            accuracies.append(accuracy)
        print("Done loading all the models")

        # Additional flag of recheck_acc to supplement the legacy flag recheck_cifar
        if args.recheck_cifar or args.recheck_acc:
            recheck_accuracies = []
            for model in models:
                log_dict = {}
                log_dict['test_losses'] = []
                recheck_accuracies.append(routines.test(args, model, test_loader, log_dict))
            print("Rechecked accuracies are ", recheck_accuracies)

        # print('checking named modules of model0 for use in compute_activations!', list(models[0].named_modules()))

        # print('what about named parameters of model0 for use in compute_activations!', [tupl[0] for tupl in list(models[0].named_parameters())])

    else:
        # get dataloaders
        print("------- Obtain dataloaders -------")
        train_loader, test_loader = get_dataloader(args)
        retrain_loader, _ = get_dataloader(args, no_randomness=args.no_random_trainloaders)

        print("------- Training independent models -------")
        models, accuracies = routines.train_models(args, train_loader, test_loader)

    # if args.debug:
    #     print(list(models[0].parameters()))

    if args.same_model!=-1:
        print("Debugging with same model")
        model, acc = models[args.same_model], accuracies[args.same_model]
        models = [model, model]
        accuracies = [acc, acc]

    for name, param in models[0].named_parameters():
        print(f'layer {name} has #params ', param.numel())

    import time
    # second_config is not needed here as well, since it's just used for the dataloader!
    print("Activation Timer start")
    st_time = time.perf_counter()
    activations = utils.get_model_activations(args, models, config=config)
    end_time = time.perf_counter()
    setattr(args, 'activation_time', end_time - st_time)
    print("Activation Timer ends")

    for idx, model in enumerate(models):
        setattr(args, f'params_model_{idx}', utils.get_model_size(model))

    # if args.ensemble_iter == 1:
    #
    # else:
    #     # else just recompute activations inside the method iteratively
    #     activations = None


    # set seed for numpy based calculations
    NUMPY_SEED = 100
    np.random.seed(NUMPY_SEED)

    # run geometric aka wasserstein ensembling
    print("------- Geometric Ensembling -------")
    # Deprecated: wasserstein_ensemble.geometric_ensembling(models, train_loader, test_loader)


    print("Timer start")
    st_time = time.perf_counter()

    geometric_acc, geometric_model = test_wasserstein_ensemble.geometric_ensembling_modularized(args, models, train_loader, test_loader, activations)
    
    end_time = time.perf_counter()
    print("Timer ends")
    setattr(args, 'geometric_time', end_time - st_time)
    args.params_geometric = utils.get_model_size(geometric_model)

    print("Time taken for geometric ensembling is {} seconds".format(str(end_time - st_time)))
