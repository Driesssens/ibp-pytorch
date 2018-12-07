import torch
import numpy as np
from enum import Enum, IntEnum
from utilities import *
from torch.distributions import Categorical


class Manager(torch.nn.Module):
    class Strategies(Enum):
        ONE_STEP = 1
        N_STEP = 2
        TREE = 3

    class Moves(IntEnum):
        ACT = 1
        IMAGINE_FROM_REAL_STATE = 2
        IMAGINE_FROM_LAST_IMAGINATION = 3

    moves_of_strategy = {
        Strategies.ONE_STEP: [Moves.ACT, Moves.IMAGINE_FROM_REAL_STATE],
        Strategies.N_STEP: [Moves.ACT, Moves.IMAGINE_FROM_REAL_STATE],
        Strategies.TREE: [Moves.ACT, Moves.IMAGINE_FROM_REAL_STATE, Moves.IMAGINE_FROM_LAST_IMAGINATION]
    }

    @property
    def moves(self):
        return self.moves_of_strategy[self.strategy]

    def __init__(self, object_vector_length=5, n_planets=5, history_vector_length=100, hidden_layer_sizes=(100, 100), strategy=Strategies.TREE):
        super().__init__()

        self.strategy = strategy

        self.neural_net = make_mlp_with_relu(
            input_size=object_vector_length * (n_planets + 1) + history_vector_length,
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=len(self.moves),
            final_relu=False
        )

    def forward(self, agent_ship, planets, history):
        input_vectors = [agent_ship] + planets + [history]
        action_logits = self.neural_net(tensor_from(input_vectors))
        action_probabilities = torch.nn.functional.softmax(action_logits, dim=0)
        action_distribution = Categorical(action_probabilities)

        i_selected_action = action_distribution.sample().item()
        selected_action = self.moves[i_selected_action]

        return selected_action
