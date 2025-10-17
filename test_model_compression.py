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
  
  # get layer representation of models (check x, y for layers within model)
  layer_representations = []
  if args.layer_measure == "index":
    layer_representation = np.arange(1, num_layers[0])
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



  
  

  
