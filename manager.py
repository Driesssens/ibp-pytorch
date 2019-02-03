from utilities import *
from torch.distributions import Categorical, Normal

if False:
    from experiment import Experiment


class Manager(torch.nn.Module):

    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        state_vector_length = 5 * (1 + self.exp.conf.n_planets + self.exp.conf.n_secondary_planets)  # (mass, xy position, xy velocity) * n_objects

        if not self.exp.conf.use_ship_mass:
            state_vector_length -= 1

        history_embedding_length = self.exp.conf.history_embedding_length
        route_vector_length = len(self.exp.conf.routes_of_strategy) if self.exp.conf.manager.manage_n_imaginations else 0
        gaussian_parameters_vector_length = 2 if self.exp.conf.manager.manage_planet_filtering else 0

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=self.exp.conf.manager.hidden_layer_sizes,
            output_size=route_vector_length + gaussian_parameters_vector_length,
            final_relu=False
        )

        self.episode_log_probabilities = []
        self.episode_entropies = []
        self.episode_costs = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.exp.conf.manager.learning_rate)

        self.batch_entropy = Accumulator()
        self.batch_entropy_loss = Accumulator()
        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_policy_loss = Accumulator()
        self.batch_total_loss = Accumulator()
        self.batch_threshold = Accumulator()
        self.batch_final_threshold = Accumulator()
        self.batch_gaussian_mean = Accumulator()
        self.batch_gaussian_variance = Accumulator()

    def forward(self):
        input_tensor = tensor_from(
            self.exp.env.agent_ship.encode_state(self.exp.conf.use_ship_mass),
            self.exp.env.planet_state_vector(),
            self.exp.agent.history_embedding.detach())

        route_logits = None
        gaussian_parameters = None

        neural_net_output = self.neural_net(input_tensor)

        if self.exp.conf.manager.manage_n_imaginations and self.exp.conf.manager.manage_planet_filtering:
            route_logits = neural_net_output[:-2]
            gaussian_parameters = neural_net_output[-2:]
        elif self.exp.conf.manager.manage_n_imaginations:
            route_logits = neural_net_output
        elif self.exp.conf.manager.manage_planet_filtering:
            gaussian_parameters = neural_net_output
        else:
            raise ValueError("A manager should manage at least one of both things.")

        if route_logits is not None:
            route_probabilities = torch.nn.functional.softmax(route_logits, dim=0)

            route_distribution = Categorical(route_probabilities)
            self.episode_entropies.append(route_distribution.entropy())

            i_selected_route = route_distribution.sample()
            self.episode_log_probabilities.append(route_distribution.log_prob(i_selected_route))

            route = i_selected_route.item()
        else:
            route = None

        if gaussian_parameters is not None:
            threshold_distribution = Normal(*gaussian_parameters)
            self.episode_entropies.append(threshold_distribution.entropy())

            self.batch_gaussian_mean.add(gaussian_parameters[0].item())
            print("mean: {}".format(gaussian_parameters[0].item()))
            print("vari: {}".format(gaussian_parameters[1].item()))
            self.batch_gaussian_variance.add(gaussian_parameters[1].item())

            selected_threshold = threshold_distribution.sample()
            self.episode_log_probabilities.append(threshold_distribution.log_prob(selected_threshold))

            threshold = selected_threshold.item()
        else:
            threshold = None

        return route, threshold

    def finish_episode(self):
        reversed_action_values = []

        for cost in reversed(self.episode_costs):
            last_action_value = 0 if not reversed_action_values else reversed_action_values[-1]
            reversed_action_values.append(last_action_value + cost)

        action_values = list(reversed(reversed_action_values))

        for i_action in range(len(action_values)):
            policy_loss = tensor_from(action_values[i_action]) * self.episode_log_probabilities[i_action]
            self.batch_policy_loss.add(policy_loss)

            entropy = self.episode_entropies[i_action]
            self.batch_entropy.add(entropy)

            entropy_loss = -entropy * self.episode_log_probabilities[i_action] * self.exp.conf.manager.entropy_factor
            self.batch_entropy_loss.add(entropy_loss)

            total_loss = policy_loss + entropy_loss
            self.batch_total_loss.add(total_loss)

            if self.exp.train_model:
                total_loss.backward()

        self.episode_log_probabilities = []
        self.episode_entropies = []
        self.episode_costs = []

    def finish_batch(self):
        if self.exp.train_model:
            self.exp.log("manager_batch_norm", gradient_norm(self.parameters()))

            if self.exp.conf.manager.max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.manager.max_gradient_norm)
                self.exp.log("manager_batch_clipped_norm", gradient_norm(self.parameters()))

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.exp.log("manager_mean_entropy", self.batch_entropy.average().item())
        self.exp.log("manager_mean_task_cost", self.batch_task_cost.average())
        self.exp.log("manager_mean_ponder_cost", self.batch_ponder_cost.average())
        self.exp.log("manager_mean_policy_loss", self.batch_policy_loss.average().item())
        self.exp.log("manager_mean_entropy_loss", self.batch_entropy_loss.average().item())
        self.exp.log("manager_mean_total_loss", self.batch_total_loss.average().item())

        if self.exp.conf.manager.manage_planet_filtering:
            self.exp.log("manager_mean_threshold", self.batch_threshold.average())
            self.exp.log("manager_mean_final_threshold", self.batch_final_threshold.average())
            self.exp.log("manager_mean_gaussian_mean", self.batch_gaussian_mean.average())
            self.exp.log("manager_mean_gaussian_variance", self.batch_gaussian_variance.average())

        self.batch_entropy = Accumulator()
        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_policy_loss = Accumulator()
        self.batch_entropy_loss = Accumulator()
        self.batch_total_loss = Accumulator()
        self.batch_threshold = Accumulator()
        self.batch_final_threshold = Accumulator()
        self.batch_gaussian_mean = Accumulator()
        self.batch_gaussian_variance = Accumulator()

    def store_model(self):
        torch.save(self.state_dict(), self.exp.file_path('manager_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('manager_optimizer_state_dict'))

    def load_model(self):
        self.load_state_dict(torch.load(self.exp.file_path("manager_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("manager_optimizer_state_dict")))


class Manager2(torch.nn.Module):

    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        state_vector_length = 5 * (1 + self.exp.conf.n_planets)  # (mass, xy position, xy velocity) * n_objects

        if not self.exp.conf.use_ship_mass:
            state_vector_length -= 1

        history_embedding_length = self.exp.conf.history_embedding_length
        route_vector_length = len(self.exp.conf.routes_of_strategy)

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=self.exp.conf.manager.hidden_layer_sizes,
            output_size=route_vector_length,
            final_relu=False
        )

        self.episode_log_probabilities = []
        self.episode_entropies = []
        self.episode_costs = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.exp.conf.manager.learning_rate)

        self.batch_entropy = Accumulator()
        self.batch_entropy_loss = Accumulator()
        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_policy_loss = Accumulator()
        self.batch_total_loss = Accumulator()

    def forward(self):
        input_tensor = tensor_from(
            self.exp.env.agent_ship.encode_state(self.exp.conf.use_ship_mass),
            self.exp.env.planet_state_vector(),
            self.exp.agent.history_embedding.detach())

        route_logits = self.neural_net(input_tensor)
        route_probabilities = torch.nn.functional.softmax(route_logits, dim=0)

        route_distribution = Categorical(route_probabilities)
        self.episode_entropies.append(route_distribution.entropy())

        i_selected_route = route_distribution.sample()
        self.episode_log_probabilities.append(route_distribution.log_prob(i_selected_route))

        return i_selected_route.item()

    def finish_episode(self):
        reversed_action_values = []

        for cost in reversed(self.episode_costs):
            last_action_value = 0 if not reversed_action_values else reversed_action_values[-1]
            reversed_action_values.append(last_action_value + cost)

        action_values = list(reversed(reversed_action_values))

        for i_action in range(len(action_values)):
            policy_loss = tensor_from(action_values[i_action]) * self.episode_log_probabilities[i_action]
            self.batch_policy_loss.add(policy_loss)

            entropy = self.episode_entropies[i_action]
            self.batch_entropy.add(entropy)

            entropy_loss = -entropy * self.episode_log_probabilities[i_action] * self.exp.conf.manager.entropy_factor
            self.batch_entropy_loss.add(entropy_loss)

            total_loss = policy_loss + entropy_loss
            self.batch_total_loss.add(total_loss)

            if self.exp.train_model:
                total_loss.backward()

        self.episode_log_probabilities = []
        self.episode_entropies = []
        self.episode_costs = []

    def finish_batch(self):
        if self.exp.train_model:
            self.exp.log("manager_batch_norm", gradient_norm(self.parameters()))

            if self.exp.conf.manager.max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.manager.max_gradient_norm)
                self.exp.log("manager_batch_clipped_norm", gradient_norm(self.parameters()))

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.exp.log("manager_mean_entropy", self.batch_entropy.average().item())
        self.exp.log("manager_mean_task_cost", self.batch_task_cost.average())
        self.exp.log("manager_mean_ponder_cost", self.batch_ponder_cost.average())
        self.exp.log("manager_mean_policy_loss", self.batch_policy_loss.average().item())
        self.exp.log("manager_mean_entropy_loss", self.batch_entropy_loss.average().item())
        self.exp.log("manager_mean_total_loss", self.batch_total_loss.average().item())

        self.batch_entropy = Accumulator()
        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_policy_loss = Accumulator()
        self.batch_entropy_loss = Accumulator()
        self.batch_total_loss = Accumulator()

    def store_model(self):
        torch.save(self.state_dict(), self.exp.file_path('manager_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('manager_optimizer_state_dict'))

    def load_model(self):
        self.load_state_dict(torch.load(self.exp.file_path("manager_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("manager_optimizer_state_dict")))
