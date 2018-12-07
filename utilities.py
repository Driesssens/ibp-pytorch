import torch
import numpy as np
from enum import Enum

def tensor_from(*args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    tensor_parts = []

    for part in args:
        if isinstance(part, (int, float, complex)):
            part = torch.tensor([part]).float()
        if isinstance(part, np.ndarray):
            part = torch.from_numpy(part).float()

        if not isinstance(part, torch.Tensor):
            raise TypeError("This value has wrong type {}: {}".format(type(part), part))

        tensor_parts.append(part)

    return torch.cat(tensor_parts)


def make_mlp_with_relu(input_size, hidden_layer_sizes, output_size, final_relu):
    if isinstance(hidden_layer_sizes, tuple):
        hidden_layer_sizes = list(hidden_layer_sizes)

    layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

    layers = []

    for i_input_layer in range(0, len(layer_sizes) - 1):
        i_output_layer = i_input_layer + 1

        layers += [
            torch.nn.Linear(layer_sizes[i_input_layer], layer_sizes[i_output_layer]),
            torch.nn.ReLU()
        ]

    if not final_relu:
        layers = layers[:-1]  # Remove last ReLU - that's the output already

    sequence = torch.nn.Sequential(*layers)
    return sequence
