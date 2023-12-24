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

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layers):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): cd p1_naviRandom seed
            layers: configuration of layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.layers = nn.ModuleList()

        for layer in layers:
            self.layers.append(layer_type[layer["type"]](*layer["arguments"]))

        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                layer.weight.data.normal_(*hidden_init(layer))


    def forward(self, state, action = None):
        """Build a network that maps state -> action values."""
        data = state
        for idx, layer in enumerate(self.layers):
            # for the critic the actions are added
            if idx == 3 and action is not None:
                data = torch.cat((data, action), dim=1)
            data = layer(data)
        # return output between -1 and 1
        # data = (data * 2) - 1
        return data