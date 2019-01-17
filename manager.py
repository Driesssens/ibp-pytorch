from utilities import *
from torch.distributions import Categorical
import os


class Manager(torch.nn.Module):

    def __init__(self,
                 parent,
                 hidden_layer_sizes=(100, 100),
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 entropy_factor=0.2
                 ):
        super().__init__()

        self.parent = parent

        state_vector_length = 5 * (1 + parent.environment.n_planets)  # (mass, xy position, xy velocity) * n_objects

        if not self.parent.use_ship_mass:
            state_vector_length -= 1

        history_embedding_length = len(parent.history_embedding)
        route_vector_length = len(self.parent.routes_of_strategy)

        self.neural_net = make_mlp_with_relu(
            input_size=state_vector_length + history_embedding_length,
            hidden_layer_sizes=hidden_layer_sizes,
            output_size=route_vector_length,
            final_relu=False
        )

        self.episode_log_probabilities = []
        self.episode_entropies = []
        self.episode_costs = []

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.max_gradient_norm = max_gradient_norm
        self.entropy_factor = entropy_factor

        self.batch_entropy = Accumulator()
        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_policy_loss = Accumulator()
        self.batch_total_loss = Accumulator()

    def forward(self):
        input_tensor = tensor_from(
            self.parent.environment.agent_ship.encode_state(self.parent.use_ship_mass),
            self.parent.environment.planet_state_vector(),
            self.parent.history_embedding.detach())

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
            last_action_value = cost if not reversed_action_values else reversed_action_values[-1]
            reversed_action_values.append(last_action_value + cost)

        action_values = list(reversed(reversed_action_values))

        for i_action in range(len(action_values)):
            policy_loss = tensor_from(action_values[i_action]) * self.episode_log_probabilities[i_action]
            self.batch_policy_loss.add(policy_loss)

            entropy = self.episode_entropies[i_action]
            self.batch_entropy.add(entropy)

            total_loss = policy_loss - self.entropy_factor * entropy
            self.batch_total_loss.add(total_loss)

            total_loss.backward()

        self.episode_log_probabilities = []
        self.episode_entropies = []
        self.episode_costs = []

    def finish_batch(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)

        self.parent.log("manager_batch_norm", norm)

        self.optimizer.step()
        self.optimizer.zero_grad()

        self.parent.log("manager_mean_entropy", self.batch_entropy.average().item())
        self.parent.log("manager_mean_task_cost", self.batch_task_cost.average())
        self.parent.log("manager_mean_ponder_cost", self.batch_ponder_cost.average())
        self.parent.log("manager_mean_policy_loss", self.batch_policy_loss.average().item())
        self.parent.log("manager_mean_total_loss", self.batch_total_loss.average().item())

        self.batch_entropy = Accumulator()
        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_policy_loss = Accumulator()
        self.batch_total_loss = Accumulator()

    def store(self):
        torch.save(self.state_dict(), os.path.join(self.parent.experiment_folder, 'manager_state_dict'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.parent.experiment_folder, 'manager_optimizer_state_dict'))

    def load(self, experiment_name):
        self.load_state_dict(torch.load(os.path.join("storage", experiment_name, "manager_state_dict")))
        self.optimizer.load_state_dict(torch.load(os.path.join("storage", experiment_name, "manager_optimizer_state_dict")))
