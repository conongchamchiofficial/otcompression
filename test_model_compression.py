#def 


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

  return args, networks, accuracies, num_layers, model_names



  
  

  
