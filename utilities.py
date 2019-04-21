import torch
import numpy as np
from enum import Enum
import random


def tensor_from(*args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    tensor_parts = []

    for part in args:
        if part is None:
            continue
        if isinstance(part, (int, float, complex)):
            part = torch.tensor([part]).float()
        if isinstance(part, np.number):
            part = torch.from_numpy(np.array([part])).float()
        if isinstance(part, np.ndarray):
            part = torch.from_numpy(part).float()

        if not isinstance(part, torch.Tensor):
            raise TypeError("This value has wrong type {}: {}".format(type(part), part))

        tensor_parts.append(part)

    return torch.cat(tensor_parts)


def make_mlp_with_relu(input_size, hidden_layer_sizes, output_size, final_relu, selu=False, leaky=False, prelu=False):
    if isinstance(hidden_layer_sizes, tuple):
        hidden_layer_sizes = list(hidden_layer_sizes)

    layer_sizes = [input_size] + hidden_layer_sizes

    if output_size > 0:
        layer_sizes += [output_size]

    layers = []

    activation = torch.nn.ReLU

    if selu:
        activation = torch.nn.SELU
    elif leaky is not False:
        activation = torch.nn.LeakyReLU
    elif prelu:
        print("yes")
        activation = torch.nn.PReLU

    for i_input_layer in range(0, len(layer_sizes) - 1):
        i_output_layer = i_input_layer + 1

        layers += [
            torch.nn.Linear(layer_sizes[i_input_layer], layer_sizes[i_output_layer]),
            activation() if leaky is False else activation(leaky),
        ]

    if not final_relu:
        layers = layers[:-1]  # Remove last ReLU - that's the output already

    sequence = torch.nn.Sequential(*layers)
    return sequence


def has_nan_gradient(parameters):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    for parameter in parameters:
        if torch.isnan(parameter.grad.mean()):
            return True
    return False


def gradient_norm(parameters):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0

    for parameter in parameters:
        parameter_norm = parameter.grad.data.norm(2)
        total_norm += parameter_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)
    return total_norm


class Accumulator:
    def __init__(self):
        self.cumulative_value = None
        self.counter = 0

    def add(self, value):
        if self.cumulative_value is None:
            self.cumulative_value = value
        else:
            self.cumulative_value += value
            # the_sum = self.cumulative_value + value
            # self.cumulative_value = the_sum

        self.counter += 1

    def average(self):
        if self.cumulative_value is None:
            return None
        average = self.cumulative_value / self.counter
        return average


class ConfigurationSeries:
    def __init__(self):
        self.names = []
        self.value_tuples = []

    def add(self, name, value):
        self.names.append(name)
        self.value_tuples.append(value)

    def index_of(self, name):
        return self.names.index(name)

    def values_of(self, name):
        return self.value_tuples[self.index_of(name)]

    def factor_of(self, index):
        if isinstance(index, str):
            index = self.index_of(index)

        factor = 1

        for values in self.value_tuples[index + 1:]:
            factor *= len(values)

        return factor

    def rank(self, name, value):
        return self.values_of(name).index(value)

    def contribution(self, name, value):
        return self.factor_of(name) * self.rank(name, value)

    def get_id(self, **kwargs):
        return sum([self.contribution(name, value) for (name, value) in kwargs.items()])

    def get_settings(self, the_id):
        settings = {}

        for name in self.names:
            settings[name] = self.values_of(name)[the_id // self.factor_of(name)]
            the_id = the_id % self.factor_of(name)

        return settings

    def get_title(self, the_id):
        settings = self.get_settings(the_id)
        title = ""

        for name in reversed(self.names):
            title += name + '_' + str(settings[name]) + '-'

        title += 'id_'
        title += str(the_id)

        return title

    def n_configurations(self):
        return len(self.values_of(self.names[0])) * self.factor_of(self.names[0])

    @classmethod
    def test(cls):
        series = cls()
        series.add('ponder', (2, 3, 4))
        series.add('grad_clip', (False, True))
        series.add('sum_loss', (False, True))

        print(series.get_id(ponder=2, grad_clip=False, sum_loss=False))
        print(series.get_settings(series.get_id(ponder=2, grad_clip=False, sum_loss=False)))

        print(series.get_id(ponder=2, grad_clip=False, sum_loss=True))
        print(series.get_settings(series.get_id(ponder=2, grad_clip=False, sum_loss=True)))

        print(series.get_id(ponder=3, grad_clip=False, sum_loss=False))
        print(series.get_settings(series.get_id(ponder=3, grad_clip=False, sum_loss=False)))

        print(series.get_id(ponder=3, grad_clip=False, sum_loss=True))
        print(series.get_settings(series.get_id(ponder=3, grad_clip=False, sum_loss=True)))

        print(series.get_id(ponder=4, grad_clip=False, sum_loss=False))
        print(series.get_settings(series.get_id(ponder=4, grad_clip=False, sum_loss=False)))

        print(series.get_id(ponder=4, grad_clip=False, sum_loss=True))
        print(series.get_settings(series.get_id(ponder=4, grad_clip=False, sum_loss=True)))

        print(series.factor_of('ponder'))
        print(series.factor_of('grad_clip'))
        print(series.factor_of('sum_loss'))

        print(series.n_configurations())


class ClippedNormal(torch.distributions.Normal):

    def __init__(self, loc, scale, lower=None, upper=None):
        super().__init__(loc, scale)

        self.lower = lower
        self.upper = upper
        self.active = self.lower is not None or self.upper is not None

    def sample(self):
        result = super().sample()

        clipped_result = torch.clamp(result, self.lower, self.upper) if self.active else result

        return result, clipped_result

    def log_prob(self, value):
        results = super().log_prob(value)

        if self.lower is not None:
            results = torch.where(value > self.lower, results, super().cdf(self.lower).log())
        if self.upper is not None:
            results = torch.where(value < self.upper, results, (1 - super().cdf(self.upper)).log())

        return results


colors = [
    (255, 193, 7),
    (0, 188, 212),
    (158, 158, 158),
    (233, 30, 99),
    (244, 67, 54),
    (156, 39, 176),
    (103, 58, 183),
    (63, 81, 181),
    (33, 150, 243),
    (3, 169, 244),
    (0, 150, 136),
    (76, 175, 80),
    (139, 195, 74),
    (205, 220, 57),
    (255, 235, 59),
    (255, 152, 0),
    (255, 87, 34),
] * 5


def get_color(i, shuffle=0):
    if shuffle == 0:
        the_colors = colors
    else:
        the_colors = random.Random(shuffle).sample(colors, len(colors))

    the_color = the_colors[i % len(the_colors)]
    return the_color


def color_string(the_color, alpha=1.0):
    return 'rgba({},{},{},{})'.format(the_color[0], the_color[1], the_color[2], alpha)
