from typing import Union
from collections import OrderedDict

import torch
from torch import nn

# TODO: Union und _str.xxx verstehen!
Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            MLP (nn.Module)
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    # class MLP(nn.Module):
    #     def __init__(self):
    #         super().__init__()
    # TOBI: Erzeuge OrderedDict mit hidden Layers
    layers = OrderedDict([])
    for i in range(n_layers):
        if i == 0: # Erster Layer hat andere input Dimension
            layers['hl_{0}'.format(i)] = nn.Linear(input_size,size)
        else:
            layers['hl_{0}'.format(i)] = nn.Linear(size,size)
        layers['act_{0}'.format(i)] = activation
    # Füge output layer hinzu
    layers['output'] = nn.Linear(size,output_size)
    layers['output_activation'] = output_activation
    
    model = nn.Sequential(layers)
    
    # Done: return a MLP. This should be an instance of nn.Module
    # Note: nn.Sequential is an instance of nn.Module.
    return model
            

device = None


def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()
