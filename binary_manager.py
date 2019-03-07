from utilities import *
from torch.distributions import Categorical, Normal, Bernoulli
from spaceship_environment import Planet, Ship

if False:
    from experiment import Experiment


class BinaryManager(torch.nn.Module):
    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        feature_input_size = sum([
            self.exp.conf.controller.object_embedding_length if self.exp.conf.manager.feature_controller_embedding else 0,
            1 if self.exp.conf.manager.feature_norm else 0,
            7 if self.exp.conf.manager.feature_state else 0,  # type (2) + mass (1) + position (2) + velocity (2)
            self.exp.conf.history_embedding_length if self.exp.conf.manager.feature_history_embedding else 0,
        ])

        self.shared_network = make_mlp_with_relu(
            input_size=feature_input_size,
            hidden_layer_sizes=self.exp.conf.manager.hidden_layer_sizes,
            output_size=0,  # no output layer; actor and critic heads will be the output
            final_relu=True
        )

        self.bernoulli_logit = torch.nn.Linear(self.exp.conf.manager.hidden_layer_sizes[-1], 1)
        self.critic = torch.nn.Linear(self.exp.conf.manager.hidden_layer_sizes[-1], 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.exp.conf.manager.learning_rate)

        self.episode_costs = []
        self.episode_estimated_values = []

        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_planet_p = Accumulator()
        self.batch_ship_p = Accumulator()

        self.batch_features = torch.FloatTensor(0, feature_input_size)
        self.batch_actions = torch.FloatTensor(0, 1)
        self.batch_old_logprobs = torch.FloatTensor(0, 1)
        self.batch_advantages = torch.FloatTensor(0, 1)
        self.batch_target_values = torch.FloatTensor(0, 1)

    def act(self, object_embeddings, objects):
        filter_indices = []
        filter_indices_this_action = []

        for i in range(len(object_embeddings)):
            if object_embeddings[i] is None:
                filter_indices.append(False)
                continue

            features = tensor_from(
                object_embeddings[i] if self.exp.conf.manager.feature_controller_embedding else None,
                object_embeddings[i].norm().item() if self.exp.conf.manager.feature_norm else None,
                tensor_from(
                    0 if isinstance(objects[i], Planet) else 1,
                    1 if isinstance(objects[i], Planet) else 0,
                    objects[i].mass,
                    objects[i].encode_state(False)
                ).detach() if self.exp.conf.manager.feature_state else None,
                self.exp.agent.history_embedding.detach() if self.exp.conf.manager.feature_history_embedding else None,
            )

            action_distribution, value_estimation = self(features)

            action = action_distribution.sample()
            filter_indices.append(action.item() == 1)
            filter_indices_this_action.append(action.item() == 1)

            logprob = action_distribution.log_prob(action).detach()

            self.batch_features = torch.cat((self.batch_features, features.unsqueeze(dim=0)))
            self.batch_actions = torch.cat((self.batch_actions, action.unsqueeze(dim=0)))
            self.batch_old_logprobs = torch.cat((self.batch_old_logprobs, logprob.unsqueeze(dim=0)))

            if isinstance(objects[i], Planet):
                self.batch_planet_p.add(action_distribution.probs)
            elif isinstance(objects[i], Ship):
                self.batch_ship_p.add(action_distribution.probs)

            self.episode_estimated_values.append(value_estimation.detach())

        return filter_indices, filter_indices_this_action

    def forward(self, features):
        processed_features = self.shared_network(features)

        logit = self.bernoulli_logit(processed_features)

        action_distribution = Bernoulli(logits=logit)
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
        self.exp.log("manager_mean_planet_p", self.batch_planet_p.average())
        self.exp.log("manager_mean_ship_p", self.batch_ship_p.average())

        self.exp.log("manager_mean_policy_loss", batch_policy_loss.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_unclipped_policy_loss", batch_unclipped_policy_loss.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_value_estimation_loss", batch_value_estimation_loss.average())  # per step - kan met data | DONE
        # self.exp.log("manager_mean_entropy_loss", self.batch_entropy_loss.average().item())  # per step - kan met data
        self.exp.log("manager_mean_total_loss", batch_total_loss.average())  # per step - kan met data | DONE
        self.exp.log("manager_mean_value_estimation_error", batch_value_estimation_error.average())  # per step - moet nog gemaakt worden | DONE

        self.exp.log("manager_mean_action", self.batch_actions.mean().item())  # per step - kan met data | DONE

        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()

        self.batch_task_cost = Accumulator()
        self.batch_ponder_cost = Accumulator()
        self.batch_planet_p = Accumulator()
        self.batch_ship_p = Accumulator()

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
