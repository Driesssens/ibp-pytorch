from utilities import *
from torch.distributions import Categorical, Normal

if False:
    from experiment import Experiment


class PPOManager(torch.nn.Module):
    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        feature_input_size = sum([
            1,  # highest norm is always included
            1 if self.exp.conf.manager.feature_average_norm else 0,
            1 if self.exp.conf.manager.feature_cumulative_norm else 0,
            self.exp.conf.history_embedding_length if self.exp.conf.manager.feature_history_embedding else 0,
        ])

        self.shared_network = make_mlp_with_relu(
            input_size=feature_input_size,
            hidden_layer_sizes=self.exp.conf.manager.hidden_layer_sizes,
            output_size=0,  # no output layer; actor and critic heads will be the output
            final_relu=True
        )

        self.gaussian_mean = torch.nn.Linear(self.exp.conf.manager.hidden_layer_sizes[-1], 1)
        self.critic = torch.nn.Linear(self.exp.conf.manager.hidden_layer_sizes[-1], 1)

        self.gaussian_log_stddev = torch.nn.Parameter(torch.tensor(self.exp.conf.manager.initial_gaussian_stddev).float())

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.exp.conf.manager.learning_rate)

        self.episode_costs = []
        self.episode_estimated_values = []

        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_threshold = Accumulator()
        self.batch_final_threshold = Accumulator()
        self.batch_gaussian_mean = Accumulator()
        self.batch_gaussian_stddev = Accumulator()

        self.batch_features = torch.FloatTensor(0, feature_input_size)
        self.batch_actions = torch.FloatTensor(0, 1)
        self.batch_old_logprobs = torch.FloatTensor(0, 1)
        self.batch_advantages = torch.FloatTensor(0, 1)
        self.batch_target_values = torch.FloatTensor(0, 1)

    def act(self, planet_embeddings):
        planet_norms = [planet_embedding.norm().item() for planet_embedding in planet_embeddings]

        features = tensor_from(
            max(planet_norms) if len(planet_norms) > 0 else 0,  # highest norm is always included
            (sum(planet_norms) / len(planet_norms) if len(planet_norms) > 0 else 0) if self.exp.conf.manager.feature_average_norm else None,
            sum(planet_norms) if self.exp.conf.manager.feature_cumulative_norm else None,
            self.exp.agent.history_embedding.detach() if self.exp.conf.manager.feature_history_embedding else None
        )

        action_distribution, value_estimation = self(features)

        action, clipped_action = action_distribution.sample()
        logprob = action_distribution.log_prob(action).detach()

        self.batch_features = torch.cat((self.batch_features, features.unsqueeze(dim=0)))
        self.batch_actions = torch.cat((self.batch_actions, action.unsqueeze(dim=0)))
        self.batch_old_logprobs = torch.cat((self.batch_old_logprobs, logprob.unsqueeze(dim=0)))

        self.batch_gaussian_mean.add(action_distribution.mean)
        self.batch_gaussian_stddev.add(action_distribution.stddev)

        self.episode_estimated_values.append(value_estimation.detach())

        return clipped_action

    def forward(self, features):
        processed_features = self.shared_network(features)

        action_mean = self.gaussian_mean(processed_features)
        action_stddev = torch.exp(self.gaussian_log_stddev)

        action_distribution = ClippedNormal(
            action_mean,
            action_stddev,
            0 if self.exp.conf.manager.lower_bounded_actions else None,
            features[..., 0] if self.exp.conf.manager.upper_bounded_actions else None
        )

        value_estimation = self.critic(processed_features)

        return action_distribution, value_estimation

    def finish_episode(self):
        reversed_episode_costs = list(reversed(self.episode_costs))
        reversed_estimated_values = list(reversed(self.episode_estimated_values))
        reversed_advantages = []
        reversed_target_values = []

        for i in range(len(self.episode_costs)):
            reward = -reversed_episode_costs[i]

            advantage = reward - reversed_estimated_values[i] + (reversed_estimated_values[i - 1] if i > 0 else 0)
            reversed_advantages.append(advantage)

            target_value = reward + (reversed_target_values[i - 1] if i > 0 else 0)
            reversed_target_values.append(target_value)

        advantages = torch.FloatTensor(list(reversed(reversed_advantages)))
        self.batch_advantages = torch.cat((self.batch_advantages, advantages.unsqueeze(dim=1)))

        target_values = torch.FloatTensor(list(reversed(reversed_target_values)))
        self.batch_target_values = torch.cat((self.batch_target_values, target_values.unsqueeze(dim=1)))

        self.episode_costs = []
        self.episode_estimated_values = []

    def finish_batch(self):
        batch_policy_loss = Accumulator()
        batch_unclipped_policy_loss = Accumulator()
        batch_value_estimation_loss = Accumulator()
        batch_value_estimation_error = Accumulator()
        batch_total_loss = Accumulator()

        for i_epoch in range(self.exp.conf.manager.n_ppo_epochs):
            distribution, value_estimation = self(self.batch_features)
            logprob = distribution.log_prob(self.batch_actions)

            ratio = torch.exp(logprob - self.batch_old_logprobs)
            unclipped_action_gain = ratio * self.batch_advantages
            clipped_action_gain = torch.clamp(ratio, 1 - self.exp.conf.manager.ppo_clip, 1 + self.exp.conf.manager.ppo_clip) * self.batch_advantages
            action_loss = -1 * torch.min(unclipped_action_gain, clipped_action_gain)

            value_estimation_loss = (value_estimation - self.batch_target_values).pow(2)

            total_loss = (action_loss + self.exp.conf.manager.c_value_estimation_loss * value_estimation_loss)

            if self.exp.train_model:
                self.optimizer.zero_grad()
                total_loss.sum().backward()

                self.exp.log("manager_batch_norm", gradient_norm(self.parameters()))
                if self.exp.conf.manager.max_gradient_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.manager.max_gradient_norm)
                    self.exp.log("manager_batch_clipped_norm", gradient_norm(self.parameters()))

                self.optimizer.step()

            batch_policy_loss.add(action_loss.mean().item())
            batch_unclipped_policy_loss.add(-1 * unclipped_action_gain.mean().item())
            batch_value_estimation_loss.add(value_estimation_loss.mean().item())
            batch_value_estimation_error.add((value_estimation - self.batch_target_values).abs().mean().item())
            batch_total_loss.add(total_loss.mean().item())

        # self.exp.log("manager_mean_entropy", self.batch_entropy.average().item())
        self.exp.log("manager_mean_task_cost", self.batch_task_cost.average())  # hele episode - is accumulator voor - wordt al gevuld | DONE
        self.exp.log("manager_mean_ponder_cost", self.batch_ponder_cost.average())  # hele episode - is accumulator voor - wordt al gevuld | DONE
        self.exp.log("manager_mean_policy_loss", batch_policy_loss.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_unclipped_policy_loss", batch_unclipped_policy_loss.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_value_estimation_loss", batch_value_estimation_loss.average())  # per step - kan met data | DONE
        # self.exp.log("manager_mean_entropy_loss", self.batch_entropy_loss.average().item())  # per step - kan met data
        self.exp.log("manager_mean_total_loss", batch_total_loss.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_value_estimation_error", batch_value_estimation_error.average())  # per step - moet nog gemaakt worden | DONE

        self.exp.log("manager_mean_threshold", self.batch_threshold.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_action", self.batch_actions.mean().item())  # per step - kan met data | DONE
        self.exp.log("manager_mean_final_threshold", self.batch_final_threshold.average())  # hele episode - is accumulator voor - wordt al gevuld | DONE
        self.exp.log("manager_mean_gaussian_mean", self.batch_gaussian_mean.average())  # per step - moet nog gemaakt worden (accumulator?) | DONE
        self.exp.log("manager_mean_gaussian_stddev", self.batch_gaussian_stddev.average())  # per step - moet nog gemaakt worden (accumulator?) | DONE

        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_final_threshold = Accumulator()
        self.batch_threshold = Accumulator()
        self.batch_gaussian_mean = Accumulator()
        self.batch_gaussian_stddev = Accumulator()

        self.batch_features = torch.FloatTensor(0, self.batch_features.shape[1])
        self.batch_advantages = torch.FloatTensor(0, 1)
        self.batch_actions = torch.FloatTensor(0, 1)
        self.batch_old_logprobs = torch.FloatTensor(0, 1)
        self.batch_target_values = torch.FloatTensor(0, 1)

    def store_model(self):
        torch.save(self.state_dict(), self.exp.file_path('manager_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('manager_optimizer_state_dict'))

    def load_model(self):
        self.load_state_dict(torch.load(self.exp.file_path("manager_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("manager_optimizer_state_dict")))


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
