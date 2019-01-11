from utilities import *
from spaceship_environment import Ship


class ControllerAndMemory:
    def __init__(self, parent, learning_rate=0.003, max_gradient_norm=10):
        self.parent = parent
        self.controller = Controller(parent)
        self.memory = Memory(parent)

        self.batch_total_loss = Accumulator()
        self.batch_task_loss = Accumulator()
        self.batch_fuel_loss = Accumulator()

        self.optimizer = torch.optim.Adam(list(self.controller.parameters()) + list(self.memory.parameters()), learning_rate)
        self.max_gradient_norm = max_gradient_norm

    def accumulate_loss(self, critic_evaluation, fuel_cost):
        self.batch_fuel_loss.add(fuel_cost)
        self.batch_task_loss.add(critic_evaluation)
        self.batch_total_loss.add(critic_evaluation + fuel_cost)

    def finish_batch(self):
        mean_loss = self.batch_total_loss.average()
        self.parent.log("controller_and_memory_mean_loss", mean_loss.item())

        self.optimizer.zero_grad()
        mean_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(list(self.controller.parameters()) + list(self.memory.parameters()), self.max_gradient_norm)

        self.parent.log("controller_and_memory_batch_norm", norm)
        self.optimizer.step()

        self.batch_total_loss = Accumulator()

        self.parent.log("controller_and_memory_mean_task_loss", self.batch_task_loss.average().item())
        self.batch_task_loss = Accumulator()

        self.parent.log("controller_and_memory_mean_fuel_loss", self.batch_fuel_loss.average().item())
        self.batch_fuel_loss = Accumulator()

    def store(self):
        torch.save(self.controller.state_dict(), self.parent.experiment_folder + '\controller_state_dict')
        torch.save(self.memory.state_dict(), self.parent.experiment_folder + '\memory_state_dict')
        torch.save(self.optimizer.state_dict(), self.parent.experiment_folder + '\controller_and_memory_optimizer_state_dict')

    def load(self, experiment_name):
        self.controller.load_state_dict(torch.load("storage\{}\controller_state_dict".format(experiment_name)))
        self.memory.load_state_dict(torch.load("storage\{}\memory_state_dict".format(experiment_name)))
        self.optimizer.load_state_dict(torch.load("storage\{}\controller_and_memory_optimizer_state_dict".format(experiment_name)))


class Controller(torch.nn.Module):
    def __init__(self, parent, hidden_layer_sizes=(100, 100)):
        super().__init__()

        self.parent = parent

        state_vector_length = 5 * (1 + parent.environment.n_planets)  # (mass, xy position, xy velocity) * n_objects

        if not self.parent.use_ship_mass:
            state_vector_length -= 1

        history_embedding_length = len(parent.history_embedding)
        action_vector_length = 2  # ship xy force

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=action_vector_length,
            final_relu=False
        )

    def forward(self, ship_state):
        input_tensor = tensor_from(ship_state.encode_state(self.parent.use_ship_mass), self.parent.environment.planet_state_vector(), self.parent.history_embedding)
        action = self.neural_net(input_tensor)
        return action


class Memory(torch.nn.Module):
    def __init__(self, parent):
        super().__init__()

        self.parent = parent

        route_vector_length = len(parent.routes_of_strategy)
        state_vector_length = 5 * (1 + parent.environment.n_planets)

        if not self.parent.use_ship_mass:
            state_vector_length -= 1

        action_vector_length = 2  # ship xy force
        history_embedding_length = len(parent.history_embedding)

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

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination, reset_state=False):
        if reset_state:
            self.reset_state()

        input_tensor = tensor_from(route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination)
        history_embedding, self.cell_state = self.lstm_cell(input_tensor.unsqueeze(0), (self.parent.history_embedding.unsqueeze(0), self.cell_state))

        return history_embedding.squeeze()

    def reset_state(self):
        self.cell_state = torch.randn(self.cell_state.shape)
