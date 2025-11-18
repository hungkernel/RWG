"""
Utility functions for neural network initialization.
"""
import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize a layer with orthogonal initialization for weights and constant for bias.

    Args:
        layer: PyTorch layer (nn.Linear, nn.Conv2d, etc.)
        std: Standard deviation for weight initialization
        bias_const: Constant value for bias initialization

    Returns:
        Initialized layer
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


