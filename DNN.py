#!/usr/bin/env python
# coding: utf-8

import torch
from collections import OrderedDict



class SinActivation(torch.nn.Module):
    def forward(self, x):
        return torch.sin(x)


# class DNN_PI(torch.nn.Module):
#     def __init__(self, input_dim, hidden1, layers, i_ac):
#         """
#         input_dim: Input dimension
#         hidden1: Number of neurons in the first layer (with sin activation)
#         layers: Network structure starting from the second layer, e.g., [64, 64, 1]
#         i_ac: Activation function selection (1-8)
#         """
#         super(DNN_PI, self).__init__()

#         # Activation function dictionary (used from the second layer onward)
#         af_all = {
#             '1': torch.nn.Tanh, '2': torch.nn.Sigmoid, '3': torch.nn.SELU,
#             '4': torch.nn.Softmax, '5': torch.nn.ReLU, '6': torch.nn.ELU,
#             '7': torch.nn.Softplus, '8': torch.nn.LeakyReLU, '9': SinActivation
#         }

#         # First layer: input -> hidden1 (activation is sin)
#         self.first_linear = torch.nn.Linear(input_dim, hidden1)

#         # Build subsequent layers
#         layer_list = []
#         in_dim = hidden1
#         for i, out_dim in enumerate(layers[:-1]):
#             layer_list.append((f'layer_{i}', torch.nn.Linear(in_dim, out_dim)))
#             af_code = str(i_ac[i])
#             layer_list.append((f'activation_{i}', af_all[af_code]()))
#             in_dim = out_dim
        
#         # Output layer (no activation)
#         layer_list.append((f'layer_out', torch.nn.Linear(in_dim, layers[-1])))

#         self.rest_layers = torch.nn.Sequential(OrderedDict(layer_list))

#     def forward(self, x):
#         x = self.first_linear(x)
#         x = torch.sin(2*torch.pi*x)  # First layer activation is sin
#         out = self.rest_layers(x)
#         return out


class DNN_PI(torch.nn.Module):
    def __init__(self, input_dim, hidden1, layers, i_ac):
        """
        input_dim: Input dimension
        hidden1: Number of neurons in the first layer (with sin activation)
        layers: Network structure starting from the second layer, e.g., [64, 64, 1]
        i_ac: Activation function selection (1-8)
        """
        super(DNN_PI, self).__init__()

        # Activation function dictionary (used from the second layer onward)
        af_all = {
            '1': torch.nn.Tanh, '2': torch.nn.Sigmoid, '3': torch.nn.SELU,
            '4': torch.nn.Softmax, '5': torch.nn.ReLU, '6': torch.nn.ELU,
            '7': torch.nn.Softplus, '8': torch.nn.LeakyReLU, '9': SinActivation
        }

        # First layer: input -> hidden1 (activation is sin)
        self.first_linear = torch.nn.Linear(input_dim, hidden1)

        # without output layer
        layer_list = []
        in_dim = hidden1
        for i, out_dim in enumerate(layers[:-1]):
            layer_list.append((f'layer_{i}', torch.nn.Linear(in_dim, out_dim)))
            af_code = str(i_ac[i])
            layer_list.append((f'activation_{i}', af_all[af_code]()))
            in_dim = out_dim
        
        self.hidden_layers = torch.nn.Sequential(OrderedDict(layer_list))

        # output layer
        self.output_layer = torch.nn.Linear(in_dim, layers[-1])

    def forward(self, x, flag=0):
        x = self.first_linear(x)
        x = torch.sin(2 * torch.pi * x)
        x = self.hidden_layers(x)
        if flag == 0:
            x = self.output_layer(x)
        return x