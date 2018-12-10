from utilities import *
from torch.distributions import Categorical


class Manager(torch.nn.Module):

    def __init__(self, state_vector_length, history_embedding_length, route_vector_length, hidden_layer_sizes=(100, 100)):
        super().__init__()

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=route_vector_length,
            final_relu=False
        )

    def forward(self, state_vector, history_embedding):
        route_logits = self.neural_net(tensor_from(state_vector, history_embedding))
        route_probabilities = torch.nn.functional.softmax(route_logits, dim=0)
        route_distribution = Categorical(route_probabilities)
        i_selected_route = route_distribution.sample().item()
        return i_selected_route
