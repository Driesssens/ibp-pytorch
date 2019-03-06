from utilities import *
from abc import ABC, abstractmethod
from spaceship_environment import cartesian2polar, polar2cartesian, Planet
from copy import copy

if False:
    from experiment import Experiment


class AbstractControllerAndMemory(ABC):
    @abstractmethod
    def __init__(self, experiment: 'Experiment'):
        self.controller = None
        self.memory = None
        self.optimizer = None

        self.exp = experiment

        self.batch_total_loss = Accumulator()
        self.batch_task_loss = Accumulator()
        self.batch_fuel_loss = Accumulator()

    def accumulate_loss(self, critic_evaluation, fuel_cost):
        if self.exp.conf.controller.immediate_mode:
            total_loss = critic_evaluation + fuel_cost
            if self.exp.train_model:
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


class ControllerAndMemory(AbstractControllerAndMemory):
    def __init__(self, experiment: 'Experiment'):
        super().__init__(experiment)

        self.controller = Controller(self.exp)
        self.memory = Memory(self.exp)

        self.optimizer = torch.optim.Adam(list(self.controller.parameters()) + list(self.memory.parameters()), self.exp.conf.controller.learning_rate)


class SetControllerAndFlatMemory(AbstractControllerAndMemory):
    def __init__(self, experiment: 'Experiment'):
        super().__init__(experiment)

        self.controller = SetController(self.exp)
        self.memory = Memory(self.exp)

        self.optimizer = torch.optim.Adam(list(self.controller.parameters()) + list(self.memory.parameters()), self.exp.conf.controller.learning_rate)


class SetControllerAndSetMemory(AbstractControllerAndMemory):
    def __init__(self, experiment: 'Experiment'):
        super().__init__(experiment)

        self.controller = SetController(self.exp)
        self.memory = SetMemory(self.exp)

        self.optimizer = torch.optim.Adam(list(self.controller.parameters()) + list(self.memory.parameters()), self.exp.conf.controller.learning_rate)


class SetController(torch.nn.Module):
    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        history_embedding_length = self.exp.conf.history_embedding_length
        action_vector_length = 2  # ship xy force

        if self.exp.conf.controller.effect_embedding_length > 0:
            self.relation_module = make_mlp_with_relu(
                input_size=(1 if self.exp.conf.use_ship_mass else 0) + 1 + 2,  # ship mass + planet mass + difference vector between ship and planet xy position
                hidden_layer_sizes=self.exp.conf.controller.relation_module_layer_sizes,
                output_size=self.exp.conf.controller.effect_embedding_length,
                final_relu=False,
                selu=self.exp.conf.controller.selu
            )

        self.control_module = make_mlp_with_relu(
            input_size=(1 if self.exp.conf.use_ship_mass else 0) + (2 + 2 if not self.exp.conf.controller.hide_ship_state else 0) + self.exp.conf.controller.effect_embedding_length + history_embedding_length,  # ship mass + ship xy position + ship xy velocity + effect embedding + history embedding
            hidden_layer_sizes=self.exp.conf.controller.control_module_layer_sizes,
            output_size=action_vector_length,
            final_relu=False,
            selu=self.exp.conf.controller.selu
        )

    def forward(self, ship):
        if self.exp.conf.controller.effect_embedding_length > 0:
            effect_embeddings = [self.relation_module(tensor_from(
                ship.mass if self.exp.conf.use_ship_mass else None,
                planet.mass,
                tensor_from(ship.xy_position) - tensor_from(planet.xy_position))
            ) for planet in self.exp.env.planets]

            if hasattr(self, 'measure_controller_planet_embedding_introspection'):
                for i, embedding in enumerate(effect_embeddings):
                    planet = self.exp.env.planets[i]

                    actual_radius = np.linalg.norm(ship.xy_position - planet.xy_position)
                    pretended_radius = actual_radius

                    pretended_xy_distance = planet.xy_position - ship.xy_position
                    minimal_radius = planet.mass

                    if pretended_radius < minimal_radius:
                        pretended_radius = minimal_radius
                        actual_angle, actual_radius = cartesian2polar(pretended_xy_distance[0], pretended_xy_distance[1])
                        pretended_xy_distance = np.array(polar2cartesian(actual_angle, pretended_radius))

                    xy_gravitational_force = self.exp.env.gravitational_constant * planet.mass * ship.mass * pretended_xy_distance / pretended_radius ** 3
                    gravitational_force_magnitude = np.linalg.norm(xy_gravitational_force)

                    self.embeddings.append(embedding.detach().numpy())
                    self.metrics.append([planet.mass, actual_radius, gravitational_force_magnitude])

            aggregate_effect_embedding = torch.mean(torch.stack(effect_embeddings), dim=0)

        action = self.control_module(tensor_from(
            ship.mass if self.exp.conf.use_ship_mass else None,
            ship.xy_position if not self.exp.conf.controller.hide_ship_state else None,
            ship.xy_velocity / self.exp.conf.controller.velocity_normalization_factor if not self.exp.conf.controller.hide_ship_state else None,
            aggregate_effect_embedding if self.exp.conf.controller.effect_embedding_length > 0 else None,
            self.exp.agent.history_embedding
        ))

        return action


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
        state_vector_length = 5 if self.exp.conf.controller.only_ship_state else 5 * (1 + self.exp.conf.n_planets)

        if not self.exp.conf.use_ship_mass:
            state_vector_length -= 1

        action_vector_length = 2  # ship xy force
        history_embedding_length = self.exp.conf.history_embedding_length

        input_vector_length = sum([
            route_vector_length if self.exp.conf.controller.use_route else 0,  # Route, so whether to ACT, IMAGINE_FROM_REAL_STATE or IMAGINE_FROM_LAST_IMAGINATION
            state_vector_length if self.exp.conf.controller.use_actual_state else 0,  # Current actual state
            state_vector_length if self.exp.conf.controller.use_last_imagined_state else 0,  # Last imagined state
            action_vector_length if self.exp.conf.controller.use_action else 0,  # Performed or imagined action
            state_vector_length if self.exp.conf.controller.use_new_state else 0,  # New imagined or actual state
            1 if self.exp.conf.controller.use_reward else 0,  # Resultant imagined or actual reward
            1 if self.exp.conf.controller.use_i_action else 0,  # Index of real actions ('j' in the paper)
            1 if self.exp.conf.controller.use_i_imagination else 0,  # Index of imagined actions since the last real action ('k' in the paper)
        ])

        self.lstm_cell = torch.nn.LSTMCell(input_vector_length, history_embedding_length)
        self.cell_state = torch.zeros(1, history_embedding_length)

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination):
        input_parts = []

        if self.exp.conf.controller.use_route:
            input_parts.append(route)

        if self.exp.conf.controller.use_actual_state:
            actual_state_vector = actual_state.encode_state(self.exp.conf.use_ship_mass)
            if not self.exp.conf.controller.only_ship_state:
                actual_state_vector = np.concatenate([actual_state_vector, self.exp.env.planet_state_vector()])
            input_parts.append(actual_state_vector)

        if self.exp.conf.controller.use_last_imagined_state:
            last_imagined_state_vector = last_imagined_state.encode_state(self.exp.conf.use_ship_mass)
            if not self.exp.conf.controller.only_ship_state:
                last_imagined_state_vector = np.concatenate([last_imagined_state_vector, self.exp.env.planet_state_vector()])
            input_parts.append(last_imagined_state_vector)

        if self.exp.conf.controller.use_action:
            input_parts.append(action)

        if self.exp.conf.controller.use_new_state:
            new_state_vector = new_state.encode_state(self.exp.conf.use_ship_mass)
            if not self.exp.conf.controller.only_ship_state:
                if new_state.state_is_tensor():
                    new_state_vector = tensor_from(new_state_vector, self.exp.env.planet_state_vector())
                else:
                    new_state_vector = np.concatenate([new_state_vector, self.exp.env.planet_state_vector()])
            input_parts.append(new_state_vector)

        if self.exp.conf.controller.use_reward:
            input_parts.append(reward)

        if self.exp.conf.controller.use_i_action:
            input_parts.append(i_action)

        if self.exp.conf.controller.use_i_imagination:
            input_parts.append(i_imagination)

        input_tensor = tensor_from(input_parts)
        history_embedding, self.cell_state = self.lstm_cell(input_tensor.unsqueeze(0), (self.exp.agent.history_embedding.unsqueeze(0), self.cell_state))

        if hasattr(self, 'measure_history_embedding_introspection'):
            self.embeddings.append(history_embedding.squeeze().detach().numpy())
            self.n_imagination.append(i_imagination)

        return history_embedding.squeeze()

    def reset_state(self):
        self.cell_state = torch.zeros(self.cell_state.shape)


class SetMemory(torch.nn.Module):
    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        object_vector_length = 7  # type (2) + mass (1) + position (2) + velocity (2)

        self.object_function = make_mlp_with_relu(
            input_size=object_vector_length,
            hidden_layer_sizes=self.exp.conf.controller.object_function_layer_sizes,
            output_size=self.exp.conf.controller.object_embedding_length,
            final_relu=False,
            selu=self.exp.conf.controller.selu
        )

        if len(self.exp.conf.controller.aggregate_function_layer_sizes) > 0:
            self.aggregate_function = make_mlp_with_relu(
                input_size=self.exp.conf.controller.object_embedding_length,
                hidden_layer_sizes=self.exp.conf.controller.aggregate_function_layer_sizes,
                output_size=self.exp.conf.controller.aggregate_embedding_length,
                final_relu=self.exp.conf.controller.relu_after_aggregate_function,
                selu=self.exp.conf.controller.selu
            )

        action_vector_length = 2  # ship xy force

        if not self.exp.conf.controller.memoryless:
            lstm_input_vector_length = sum([
                self.exp.conf.controller.aggregate_embedding_length if len(self.exp.conf.controller.aggregate_function_layer_sizes) > 0 else self.exp.conf.controller.object_embedding_length,
                action_vector_length if self.exp.conf.controller.use_action else 0,
                1 if self.exp.conf.controller.use_i_imagination else 0,
            ])

            self.lstm_cell = torch.nn.LSTMCell(lstm_input_vector_length, self.exp.conf.history_embedding_length)

        self.cell_state = torch.zeros(1, self.exp.conf.history_embedding_length)

    def get_object_embeddings(self, objects):
        objects = copy(objects)

        for i in range(len(objects)):
            if objects[i] is not None:
                tensor = tensor_from(
                    0 if isinstance(objects[i], Planet) else 1,
                    1 if isinstance(objects[i], Planet) else 0,
                    objects[i].mass,
                    objects[i].encode_state(False)
                )

                embedding = self.object_function(tensor)
                objects[i] = embedding

        return objects

    def forward(self, route, actual_state, last_imagined_state, action, new_state, reward, i_action, i_imagination, filter_indices=None):
        objects = [new_state] + self.exp.env.planets

        if filter_indices is not None:
            objects = [objekt for (objekt, filter_value) in zip(objects, filter_indices) if filter_value]

        if hasattr(self, 'measure_performance_under_more_and_unobserved_planets'):
            if self.measure_performance_under_more_and_unobserved_planets == 'only_ship_observed':
                assert filter_indices is None
                objects = [new_state]
            elif self.measure_performance_under_more_and_unobserved_planets == 'extra_planets_unobserved':
                assert filter_indices is None
                objects = [new_state] + self.exp.env.planets[:-1]

        object_embeddings = self.get_object_embeddings(objects)

        if hasattr(self, 'measure_setmemory_object_embedding_introspection'):
            for i, embedding in enumerate(object_embeddings):
                obj = objects[i]
                radius = np.linalg.norm(obj.xy_position)
                is_planet = isinstance(obj, Planet)

                self.embeddings.append(embedding.detach().numpy())
                self.metrics.append([int(is_planet), obj.mass, radius, obj.x, obj.y])

        if len(object_embeddings) == 0:
            aggregate_embedding = torch.zeros(self.exp.conf.controller.object_embedding_length)
        else:
            aggregate_embedding = torch.mean(torch.stack(object_embeddings), dim=0)

        if len(self.exp.conf.controller.aggregate_function_layer_sizes) > 0:
            aggregate_embedding = self.aggregate_function(torch.mean(torch.stack(object_embeddings), dim=0))

        lstm_input = tensor_from(
            action if self.exp.conf.controller.use_action else None,
            aggregate_embedding,
            i_imagination if self.exp.conf.controller.use_i_imagination else None
        )

        if self.exp.conf.controller.memoryless:
            history_embedding = lstm_input
        else:
            history_embedding, self.cell_state = self.lstm_cell(
                lstm_input.unsqueeze(0),
                (self.exp.agent.history_embedding.unsqueeze(0), self.cell_state)
            )

            history_embedding = history_embedding.squeeze()

        if hasattr(self, 'measure_history_embedding_introspection'):
            self.embeddings.append(history_embedding.detach().numpy())
            self.n_imagination.append(i_imagination)

        return history_embedding

    def reset_state(self):
        self.cell_state = torch.zeros(self.cell_state.shape)
