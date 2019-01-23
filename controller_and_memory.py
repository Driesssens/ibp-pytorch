from utilities import *

if False:
    from experiment import Experiment


class ControllerAndMemory:
    def __init__(self, experiment: 'Experiment'):
        self.exp = experiment
        self.controller = Controller(self.exp)
        self.memory = Memory(self.exp)

        self.batch_total_loss = Accumulator()
        self.batch_task_loss = Accumulator()
        self.batch_fuel_loss = Accumulator()

        self.optimizer = torch.optim.Adam(list(self.controller.parameters()) + list(self.memory.parameters()), self.exp.conf.controller.learning_rate)
        self.max_gradient_norm = self.exp.conf.controller.max_gradient_norm

    def accumulate_loss(self, critic_evaluation, fuel_cost):
        if self.exp.conf.controller.immediate_mode:
            total_loss = critic_evaluation + fuel_cost
            total_loss.backward()

        self.batch_fuel_loss.add(fuel_cost)
        self.batch_task_loss.add(critic_evaluation)
        self.batch_total_loss.add(critic_evaluation + fuel_cost)

    def finish_batch(self):
        mean_loss = self.batch_total_loss.average()

        if self.exp.train_model:
            if not self.exp.conf.controller.immediate_mode:
                self.optimizer.zero_grad()
                mean_loss.backward()

            self.exp.log("controller_and_memory_batch_norm", gradient_norm(self.parameters()))

            if self.exp.conf.controller.max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.controller.max_gradient_norm)
                self.exp.log("controller_and_memory_batch_clipped_norm", gradient_norm(self.parameters()))

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.exp.log("controller_and_memory_mean_loss", mean_loss.item())
        self.batch_total_loss = Accumulator()

        self.exp.log("controller_and_memory_mean_task_loss", self.batch_task_loss.average().item())
        self.batch_task_loss = Accumulator()

        self.exp.log("controller_and_memory_mean_fuel_loss", self.batch_fuel_loss.average().item())
        self.batch_fuel_loss = Accumulator()

    def store_model(self):
        torch.save(self.controller.state_dict(), self.exp.file_path('controller_state_dict'))
        torch.save(self.memory.state_dict(), self.exp.file_path('memory_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('controller_and_memory_optimizer_state_dict'))

    def load_model(self):
        self.controller.load_state_dict(torch.load(self.exp.file_path("controller_state_dict")))
        self.memory.load_state_dict(torch.load(self.exp.file_path("memory_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("controller_and_memory_optimizer_state_dict")))

    def parameters(self):
        return list(self.controller.parameters()) + list(self.memory.parameters())


class Controller(torch.nn.Module):
    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        state_vector_length = 5 * (1 + self.exp.conf.n_planets)  # (mass, xy position, xy velocity) * n_objects

        if not self.exp.conf.use_ship_mass:
            state_vector_length -= 1

        history_embedding_length = self.exp.conf.history_embedding_length
        action_vector_length = 2  # ship xy force

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=self.exp.conf.controller.hidden_layer_sizes,
            output_size=action_vector_length,
            final_relu=False
        )

    def forward(self, ship_state):
        input_tensor = tensor_from(ship_state.encode_state(self.exp.conf.use_ship_mass), self.exp.env.planet_state_vector(), self.exp.agent.history_embedding)
        action = self.neural_net(input_tensor)
        return action


class Memory(torch.nn.Module):
    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        route_vector_length = len(self.exp.conf.routes_of_strategy)
        state_vector_length = 5 * (1 + self.exp.conf.n_planets)

        if not self.exp.conf.use_ship_mass:
            state_vector_length -= 1

        action_vector_length = 2  # ship xy force
        history_embedding_length = self.exp.conf.history_embedding_length

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

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination):
        input_tensor = tensor_from(route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination)
        history_embedding, self.cell_state = self.lstm_cell(input_tensor.unsqueeze(0), (self.exp.agent.history_embedding.unsqueeze(0), self.cell_state))

        return history_embedding.squeeze()

    def reset_state(self):
        self.cell_state = torch.randn(self.cell_state.shape)
