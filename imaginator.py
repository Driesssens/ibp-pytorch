import torch
import numpy as np
from utilities import *


class Imaginator(torch.nn.Module):
    def __init__(self,
                 object_vector_length=5,
                 relation_module_layer_sizes=(150, 150, 150, 150),
                 effect_embedding_length=100,
                 action_vector_length=2,
                 object_module_layer_sizes=(100,),
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

        effect_embeddings = [self.relation_module(tensor_from(agent_ship, planet)) for planet in planets]

        aggregate_effect_embedding = torch.sum(torch.stack(effect_embeddings), dim=0)
        prediction = self.object_module(tensor_from(agent_ship, action, aggregate_effect_embedding))

        return prediction
