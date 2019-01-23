from utilities import *
from spaceship_environment import Ship, Planet
from typing import List
from copy import deepcopy

if False:
    from experiment import Experiment


class Imaginator(torch.nn.Module):

    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        self.relation_module = make_mlp_with_relu(
            input_size=(1 if self.exp.conf.use_ship_mass else 0) + 1 + 2,  # ship mass + planet mass + difference vector between ship and planet xy position
            hidden_layer_sizes=self.exp.conf.imaginator.relation_module_layer_sizes,
            output_size=self.exp.conf.imaginator.effect_embedding_length,
            final_relu=False
        )

        self.object_module = make_mlp_with_relu(
            input_size=(1 if self.exp.conf.use_ship_mass else 0) + 2 + 2 + self.exp.conf.imaginator.effect_embedding_length,  # ship mass + ship xy velocity + action + effect embedding
            hidden_layer_sizes=self.exp.conf.imaginator.object_module_layer_sizes,
            output_size=2,  # imagined velocity
            final_relu=False
        )

        self.batch_loss = Accumulator()
        self.batch_evaluation = Accumulator()
        self.batch_task_cost = Accumulator()

        self.optimizer = torch.optim.Adam(self.parameters(), self.exp.conf.imaginator.learning_rate)

    def imagine(self, ship: Ship, planets: List[Planet], action, differentiable_trajectory=False):
        imagined_ship_trajectory = [deepcopy(ship)]
        imagined_ship_trajectory[-1].wrap_in_tensors()

        for i_physics_step in range(self.exp.env.n_steps_per_action):
            current_state = imagined_ship_trajectory[-1]

            imagined_state = deepcopy(current_state)
            imagined_state.xy_position = current_state.xy_position + self.exp.env.euler_method_step_size * current_state.xy_velocity

            imagined_velocity = self(
                current_state,
                planets,
                action if i_physics_step == 0 else np.zeros(2)
            )

            imagined_state.xy_velocity = imagined_velocity
            imagined_ship_trajectory.append(imagined_state)

            if not differentiable_trajectory:
                current_state.detach_and_to_numpy()

        target_position = torch.zeros(2)
        imagined_task_cost = torch.nn.functional.mse_loss(imagined_ship_trajectory[-1].xy_position.unsqueeze(0), target_position.unsqueeze(0))

        if not differentiable_trajectory:
            imagined_ship_trajectory[-1].detach_and_to_numpy()

        if self.exp.conf.fuel_price == 0:
            imagined_fuel_cost = tensor_from(0)
        else:
            imagined_fuel_cost = torch.clamp((torch.norm(tensor_from(action)) - self.exp.conf.fuel_cost_threshold) * self.exp.conf.fuel_price, min=0)

        return imagined_ship_trajectory, imagined_task_cost, imagined_fuel_cost

    def forward(self, ship: Ship, planets: List[Planet], action):
        if self.exp.conf.use_ship_mass:
            effect_embeddings = [self.relation_module(tensor_from(ship.mass, planet.mass, tensor_from(ship.xy_position) - tensor_from(planet.xy_position))) for planet in planets]
        else:
            effect_embeddings = [self.relation_module(tensor_from(planet.mass, tensor_from(ship.xy_position) - tensor_from(planet.xy_position))) for planet in planets]

        aggregate_effect_embedding = torch.mean(torch.stack(effect_embeddings), dim=0)

        if self.exp.conf.use_ship_mass:
            imagined_velocity = self.object_module(tensor_from(
                ship.mass,
                ship.xy_velocity / self.exp.conf.imaginator.velocity_normalization_factor,
                action / self.exp.conf.imaginator.action_normalization_factor,
                aggregate_effect_embedding
            ))
        else:
            imagined_velocity = self.object_module(tensor_from(
                ship.xy_velocity / self.exp.conf.imaginator.velocity_normalization_factor,
                action / self.exp.conf.imaginator.action_normalization_factor,
                aggregate_effect_embedding
            ))
        return imagined_velocity

    def accumulate_loss(self, ship_trajectory: List[Ship], planets: List[Planet], action):
        for i_physics_step in range(len(ship_trajectory) - 1):
            previous_state = ship_trajectory[i_physics_step]
            imagined_velocity = self(previous_state, planets, action if i_physics_step == 0 else np.zeros(2))
            target_velocity = ship_trajectory[i_physics_step + 1].xy_velocity

            loss = torch.nn.functional.mse_loss(imagined_velocity.unsqueeze(0), tensor_from(target_velocity).unsqueeze(0), reduction='sum')

            self.batch_loss.add(loss)

    def evaluate(self, old_ship_state: Ship, planets: List[Planet], action, actual_new_ship_state: Ship):
        estimated_trajectory, critic_evaluation, estimated_fuel_cost = self.imagine(old_ship_state, planets, action)
        imagined_final_position = estimated_trajectory[-1].xy_position
        actual_final_position = actual_new_ship_state.xy_position

        evaluation = np.square(imagined_final_position - actual_final_position).mean()
        self.batch_evaluation.add(evaluation)

        task_cost = np.square(actual_final_position).mean()
        self.batch_task_cost.add(task_cost)

        return estimated_trajectory, critic_evaluation, estimated_fuel_cost

    def finish_batch(self):
        mean_loss = self.batch_loss.average()
        self.exp.log("imaginator_mean_loss", mean_loss.item())

        if self.exp.train_model:
            self.optimizer.zero_grad()

            if self.exp.conf.imaginator.batch_loss_sum:
                self.batch_loss.cumulative_value.backward()
            else:
                mean_loss.backward()

            self.exp.log("imaginator_batch_norm", gradient_norm(self.parameters()))

            if self.exp.conf.imaginator.max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.imaginator.max_gradient_norm)
                self.exp.log("imaginator_batch_clipped_norm", gradient_norm(self.parameters()))

            self.optimizer.step()

        self.batch_loss = Accumulator()

        self.exp.log("imaginator_mean_final_position_error", self.batch_evaluation.average())
        self.batch_evaluation = Accumulator()

        self.exp.log("controller_and_memory_mean_task_cost", self.batch_task_cost.average())
        self.batch_task_cost = Accumulator()

    def store_model(self):
        torch.save(self.state_dict(), self.exp.file_path('imaginator_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('imaginator_optimizer_state_dict'))

    def load_model(self):
        self.load_state_dict(torch.load(self.exp.file_path("imaginator_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("imaginator_optimizer_state_dict")))
