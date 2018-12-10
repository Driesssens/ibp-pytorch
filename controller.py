from utilities import *


class Controller(torch.nn.Module):
    def __init__(self, state_vector_length, history_embedding_length, action_vector_length, hidden_layer_sizes=(100, 100)):
        super().__init__()

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=action_vector_length,
            final_relu=False
        )

    def forward(self, state_vector, history_embedding):
        action = self.neural_net(tensor_from(state_vector, history_embedding))
        return action
