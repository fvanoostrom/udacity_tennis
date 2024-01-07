import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

layer_type = {"linear":nn.Linear, "batchnorm": nn.BatchNorm1d,"relu":nn.ReLU, "tanh":nn.Tanh}

def create_layer(layer_config:dict):

    return layer

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, configuration):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): cd p1_naviRandom seed
            layers: configuration of layers
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.configuration = configuration
        self.layers = nn.ModuleList()

        for config in configuration:
            self.layers.append(layer_type[config["type"]](*config["arguments"]))

        self.reset_parameters()
        
    def reset_parameters(self):
        for layer, configuration in zip(self.layers, self.configuration):
            if 'initial_weight' in configuration.keys():
                if configuration['initial_weight'] is not None:
                    layer.weight.data.uniform_(*configuration['initial_weight'])
                else:
                    layer.weight.data.uniform_(*hidden_init(layer))


    def forward(self, state, action = None):
        """Build a network that maps state -> action values."""
        
        # if state has 1 dim, unsqueeze it
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        data = state
        for layer in self.layers:
            data = layer(data)
        return data

class Actor(Network):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layers: configuration of layers
        """
        super().__init__(state_size, action_size, seed, layers)
        
        

class Critic(Network):
    """Critic (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layers: configuration of layers
        """
        super().__init__(state_size, action_size, seed, layers)

    def forward(self, state, action = None):
        """Build a network that maps state -> action values."""
        # if state has 1 dim, unsqueeze it
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        data = state

        for layer in self.layers:
            #if the sizes between the input and required input are different we will add the actions to it:
            if hasattr(layer,'in_features') and layer.in_features != data.shape[1]:
                data = torch.cat((data, action), dim=1)
            data = layer(data)
        return data
