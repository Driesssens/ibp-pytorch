from utilities import *


class Memory(torch.nn.Module):
    def __init__(self, route_vector_length, state_vector_length, action_vector_length, history_embedding_length=100):
        super().__init__()

        input_vector_length = sum([
            route_vector_length,  # Route, so whether to ACT, IMAGINE_FROM_REAL_STATE or IMAGINE_FROM_LAST_IMAGINATION
            state_vector_length,  # Current actual state
            state_vector_length,  # Last imagined state
            action_vector_length,  # Performed or imagined action
            state_vector_length,  # New imagined or actual state
            1,  # Resultant imagined or actual reward
            1,  # Index of real actions ('j' in the paper)
            1,  # Index of imagined actions since the last real action ('k' in the paper)
        ])

        self.lstm_cell = torch.nn.LSTMCell(input_vector_length, history_embedding_length)
        self.cell_state = torch.randn(1, history_embedding_length)
        self.history_embedding = torch.randn(1, history_embedding_length)

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination, reset_state=False):
        if reset_state:
            self.reset_state()

        input_tensor = tensor_from(route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination).unsqueeze(0)
        self.history_embedding, self.cell_state = self.lstm_cell(input_tensor, (self.history_embedding, self.cell_state))

        return self.history_embedding.squeeze()

    def reset_state(self):
        self.cell_state = torch.randn(self.cell_state.shape)
        self.history_embedding = torch.randn(self.history_embedding.shape)
