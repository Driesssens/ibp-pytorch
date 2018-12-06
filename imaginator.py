import torch
import numpy as np


class Imaginator(torch.nn.Module):
    def __init__(self,
                 object_vector_length=5,
                 relation_module_layer_sizes=[150, 150, 150, 150],
                 effect_embedding_length=100,
                 action_vector_length=2,
                 object_module_layer_sizes=[100],
                 state_prediction_vector_length=2):
        super().__init__()

        self.relation_module = make_mlp_with_relu(
            input_size=2 * object_vector_length,
            hidden_layer_sizes=relation_module_layer_sizes,
            output_size=effect_embedding_length,
            final_relu=True
        )

        fuel_cost_prediction_vector_length = 1
        task_loss_prediction_vector_length = 1

        self.object_module = make_mlp_with_relu(
            input_size=object_vector_length + action_vector_length + effect_embedding_length,
            hidden_layer_sizes=object_module_layer_sizes,
            output_size=state_prediction_vector_length + fuel_cost_prediction_vector_length + task_loss_prediction_vector_length,
            final_relu=False
        )

    def forward(self, agent_ship, planets, action=None):
        if action is None:
            action = np.zeros(2)

        for planet in planets:
            print(planet.dtype)

        effect_embeddings = [self.relation_module(tensor_from(agent_ship, planet)) for planet in planets]

        aggregate_effect_embedding = torch.sum(torch.stack(effect_embeddings), dim=0)
        prediction = self.object_module(tensor_from(agent_ship, action, aggregate_effect_embedding))

        return prediction


def tensor_from(*args):
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]

    tensor_parts = [torch.from_numpy(item).float() if isinstance(item, np.ndarray) else item for item in args]

    return torch.cat(tensor_parts)


def make_mlp_with_relu(input_size, hidden_layer_sizes, output_size, final_relu):
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
