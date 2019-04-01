from utilities import *
from spaceship_environment import Ship, Planet, polar2cartesian, cartesian2polar
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
        self.batch_l2_loss = Accumulator()
        self.batch_evaluation = Accumulator()
        self.batch_task_cost = Accumulator()

        self.batch_important_p = Accumulator()
        self.batch_n_planets_in_each_imagination = Accumulator()
        self.batch_f_planets_in_each_imagination = Accumulator()

        self.optimizer = torch.optim.Adam(self.parameters(), self.exp.conf.imaginator.learning_rate)

    def imagine(self, ship: Ship, planets: List[Planet], action, differentiable_trajectory=False):
        imagined_ship_trajectory = [deepcopy(ship)]
        imagined_ship_trajectory[-1].wrap_in_tensors()

        filtered_planets = self.hard_attention(ship, planets) if self.is_self_filtering(per_imag=False) else planets

        if not self.is_self_filtering(per_imag=True):
            n_important = len([planet for planet in filtered_planets if not planet.is_secondary])
            self.batch_important_p.add(n_important / self.exp.conf.n_planets)
            self.batch_n_planets_in_each_imagination.add(len(filtered_planets))
            self.batch_f_planets_in_each_imagination.add(len(filtered_planets) / len(planets) if len(planets) != 0 else 1)

        for i_physics_step in range(self.exp.env.n_steps_per_action):
            current_state = imagined_ship_trajectory[-1]

            imagined_state = deepcopy(current_state)
            imagined_state.xy_position = current_state.xy_position + self.exp.env.euler_method_step_size * current_state.xy_velocity

            plan = self.hard_attention(current_state, planets) if self.is_self_filtering(per_imag=True) else filtered_planets

            if self.is_self_filtering(per_imag=True):
                n_important = len([planet for planet in plan if not planet.is_secondary])
                self.batch_important_p.add(n_important / self.exp.conf.n_planets)
                self.batch_n_planets_in_each_imagination.add(len(plan))
                self.batch_f_planets_in_each_imagination.add(len(plan) / len(planets) if len(planets) != 0 else 1)

            imagined_velocity, _ = self(
                current_state,
                plan,
                action if i_physics_step == 0 else np.zeros(2)
            )

            imagined_state.xy_velocity = imagined_velocity
            imagined_ship_trajectory.append(imagined_state)

            if not differentiable_trajectory:
                current_state.detach_and_to_numpy()

        target_position = torch.zeros(2) if len(self.exp.env.beacons) == 0 else torch.FloatTensor(self.exp.env.beacons[0].xy_position)
        imagined_task_cost = torch.nn.functional.mse_loss(imagined_ship_trajectory[-1].xy_position.unsqueeze(0), target_position.unsqueeze(0))

        if not differentiable_trajectory:
            imagined_ship_trajectory[-1].detach_and_to_numpy()

        if self.exp.conf.fuel_price == 0:
            imagined_fuel_cost = tensor_from(0)
        else:
            imagined_fuel_cost = torch.clamp((torch.norm(tensor_from(action)) - self.exp.conf.fuel_cost_threshold) * self.exp.conf.fuel_price, min=0)

        return imagined_ship_trajectory, imagined_task_cost, imagined_fuel_cost

    def filter(self, ship: Ship, planets: List[Planet], threshold):
        with torch.no_grad():
            effect_embeddings = [
                self.relation_module(tensor_from(
                    ship.mass if self.exp.conf.use_ship_mass else None,
                    planet.mass,
                    tensor_from(ship.xy_position) - tensor_from(planet.xy_position)
                )) for planet in planets
            ]

        norms = [effect_embedding.norm().item() for effect_embedding in effect_embeddings]

        return [planet for (planet, norm) in zip(planets, norms) if norm > threshold]

    def embed(self, ship: Ship, planets: List[Planet]):
        effect_embeddings = [
            self.relation_module(tensor_from(
                ship.mass if self.exp.conf.use_ship_mass else None,
                planet.mass,
                tensor_from(ship.xy_position) - tensor_from(planet.xy_position)
            )) for planet in planets
        ]

        return effect_embeddings

    def forward(self, ship: Ship, planets: List[Planet], action):
        if self.exp.conf.imaginator_ignores_secondary:
            planets = [planet for planet in planets if not planet.is_secondary]

        effect_embeddings = self.embed(ship, planets)

        if hasattr(self, 'measure_imaginator_planet_embedding_introspection') and not isinstance(action, np.ndarray):
            for i, embedding in enumerate(effect_embeddings):
                planet = self.exp.env.planets[i]

                actual_radius = np.linalg.norm(ship.xy_position.detach().numpy() - planet.xy_position)
                pretended_radius = actual_radius

                pretended_xy_distance = planet.xy_position - ship.xy_position.detach().numpy()
                minimal_radius = planet.mass

                if pretended_radius < minimal_radius:
                    pretended_radius = minimal_radius
                    actual_angle, actual_radius = cartesian2polar(pretended_xy_distance[0], pretended_xy_distance[1])
                    pretended_xy_distance = np.array(polar2cartesian(actual_angle, pretended_radius))

                xy_gravitational_force = self.exp.env.gravitational_constant * planet.mass * ship.mass * pretended_xy_distance / pretended_radius ** 3
                gravitational_force_magnitude = np.linalg.norm(xy_gravitational_force)

                self.embeddings.append(embedding.detach().numpy())
                self.metrics.append([planet.mass, actual_radius, gravitational_force_magnitude])

        if len(effect_embeddings) == 0:
            aggregate_effect_embedding = torch.zeros(self.exp.conf.imaginator.effect_embedding_length)
            mean_embedding_norm = 0
        else:
            stacked = torch.stack(effect_embeddings)
            mean_embedding_norm = stacked.norm(dim=1).mean()
            aggregate_effect_embedding = torch.mean(stacked, dim=0)

        imagined_velocity = self.object_module(tensor_from(
            ship.mass if self.exp.conf.use_ship_mass else None,
            ship.xy_velocity / self.exp.conf.imaginator.velocity_normalization_factor,
            action / self.exp.conf.imaginator.action_normalization_factor,
            aggregate_effect_embedding
        ))

        return imagined_velocity, mean_embedding_norm

    def accumulate_loss(self, ship_trajectory: List[Ship], planets: List[Planet], action):
        only_filtered = ((self.exp.conf.imaginator.adahan_threshold is not None) or (self.exp.conf.imaginator.simplehan_threshold is not None)) \
                        and self.exp.conf.imaginator.l2_on_filtered

        scale_filtered = ((self.exp.conf.imaginator.adahan_threshold is not None) or (self.exp.conf.imaginator.simplehan_threshold is not None)) \
                         and self.exp.conf.imaginator.l2_scaled_by_filtered

        filtered_planets = self.hard_attention(ship_trajectory[0], planets) if self.is_self_filtering(per_imag=False) else planets

        if not (only_filtered or scale_filtered):
            self.batch_l2_loss.add(torch.stack(self.embed(ship_trajectory[0], planets)).norm(dim=1).mean() * self.exp.conf.imaginator.c_l2_loss)
            # print("this no4 happens")

        if only_filtered and self.is_self_filtering(per_imag=False):
            self.batch_l2_loss.add(torch.stack(self.embed(ship_trajectory[0], filtered_planets)).norm(dim=1).sum() / len(planets) * self.exp.conf.imaginator.c_l2_loss)
            # print("this no3 happens")

        if scale_filtered and self.is_self_filtering(per_imag=False):
            self.batch_l2_loss.add(torch.stack(self.embed(ship_trajectory[0], planets)).norm(dim=1).mean() * self.exp.conf.imaginator.c_l2_loss * len(filtered_planets) / len(planets))
            # print("this no2 happens")

        for i_physics_step in range(len(ship_trajectory) - 1):
            previous_state = ship_trajectory[i_physics_step]

            if self.is_self_filtering(per_imag=True):
                plans = self.hard_attention(previous_state, planets)

                if only_filtered and i_physics_step == 0:
                    self.batch_l2_loss.add(torch.stack(self.embed(previous_state, plans)).norm(dim=1).sum() / len(planets) * self.exp.conf.imaginator.c_l2_loss)
                    # print("this no1 happens")

                if scale_filtered and i_physics_step == 0:
                    self.batch_l2_loss.add(torch.stack(self.embed(ship_trajectory[0], planets)).norm(dim=1).mean() * self.exp.conf.imaginator.c_l2_loss * len(plans) / len(planets))
                    # print("this happens")
            else:
                plans = filtered_planets

            imagined_velocity, mean_embedding_norm = self(
                previous_state,
                plans,
                action if i_physics_step == 0 else np.zeros(2)
            )

            target_velocity = ship_trajectory[i_physics_step + 1].xy_velocity

            loss = torch.nn.functional.mse_loss(imagined_velocity.unsqueeze(0), tensor_from(target_velocity).unsqueeze(0), reduction='sum')

            self.batch_loss.add(loss)
            # self.batch_l2_loss.add(mean_embedding_norm * self.exp.conf.imaginator.c_l2_loss)
            # self.batch_l2_loss.add(torch.stack(self.embed(ship_trajectory[0], planets)).norm(dim=1).mean() * self.exp.conf.imaginator.c_l2_loss)

    def evaluate(self, old_ship_state: Ship, planets: List[Planet], action, actual_new_ship_state: Ship):
        estimated_trajectory, critic_evaluation, estimated_fuel_cost = self.imagine(old_ship_state, planets, action)
        imagined_final_position = estimated_trajectory[-1].xy_position
        actual_final_position = actual_new_ship_state.xy_position

        evaluation = np.square(imagined_final_position - actual_final_position).mean()
        self.batch_evaluation.add(evaluation)

        if len(self.exp.env.beacons) == 0:
            task_cost = np.square(actual_final_position).mean()
        else:
            task_cost = np.square(actual_final_position - self.exp.env.beacons[0].xy_position).mean()

        self.batch_task_cost.add(task_cost)

        if self.exp.agent.has_curator():
            f_planets_kept = len(planets) / len(self.exp.env.planets)
            ponder_cost = f_planets_kept * self.exp.conf.manager.ponder_price
            self.exp.agent.manager.episode_costs.append(ponder_cost + evaluation)
            self.exp.agent.manager.batch_ponder_cost.add(ponder_cost)
            self.exp.agent.manager.batch_task_cost.add(ponder_cost + evaluation)

            if hasattr(self.exp.agent, 'measure_performance'):
                self.exp.agent.manager_mean_task_cost_measurements.append(ponder_cost + evaluation)

            self.exp.agent.batch_n_planets_in_each_imagination.add(len(planets))
            self.exp.agent.batch_f_planets_in_each_imagination.add(f_planets_kept)

        if hasattr(self.exp.agent, 'measure_performance'):
            self.exp.agent.imaginator_mean_final_position_error_measurements.append(evaluation)
            self.exp.agent.controller_and_memory_mean_task_cost_measurements.append(task_cost)

        return estimated_trajectory, critic_evaluation, estimated_fuel_cost

    def finish_batch(self):
        mean_loss = self.batch_loss.average()
        mean_l2_loss = self.batch_l2_loss.average()
        self.exp.log("imaginator_mean_loss", mean_loss.item())
        self.exp.log("imaginator_mean_l2_loss", mean_l2_loss.item())
        self.exp.log("imaginator_mean_total_loss", mean_loss.item() + mean_l2_loss.item())

        if self.exp.train_model:
            self.optimizer.zero_grad()

            if self.exp.conf.imaginator.batch_loss_sum:
                (self.batch_loss.cumulative_value + self.batch_l2_loss.cumulative_value).backward()
            else:
                (mean_loss + mean_l2_loss).backward()

            self.exp.log("imaginator_batch_norm", gradient_norm(self.parameters()))

            if self.exp.conf.imaginator.max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.imaginator.max_gradient_norm)
                self.exp.log("imaginator_batch_clipped_norm", gradient_norm(self.parameters()))

            self.optimizer.step()

        self.batch_loss = Accumulator()
        self.batch_l2_loss = Accumulator()

        self.exp.log("imaginator_mean_final_position_error", self.batch_evaluation.average())
        self.batch_evaluation = Accumulator()

        self.exp.log("controller_and_memory_mean_task_cost", self.batch_task_cost.average())
        self.batch_task_cost = Accumulator()

        self.exp.log("imaginator_mean_n_planets", self.batch_n_planets_in_each_imagination.average())
        self.exp.log("imaginator_mean_f_planets", self.batch_f_planets_in_each_imagination.average())
        self.exp.log("imaginator_mean_important_p", self.batch_important_p.average())

        self.batch_important_p = Accumulator()
        self.batch_n_planets_in_each_imagination = Accumulator()
        self.batch_f_planets_in_each_imagination = Accumulator()

    def store_model(self):
        torch.save(self.state_dict(), self.exp.file_path('imaginator_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('imaginator_optimizer_state_dict'))

    def load_model(self):
        self.load_state_dict(torch.load(self.exp.file_path("imaginator_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("imaginator_optimizer_state_dict")))

    def is_self_filtering(self, per_imag=None):
        sf = (self.exp.conf.imaginator.han_n_top_objects is not None) \
             or (self.exp.conf.imaginator.adahan_threshold is not None) \
             or (self.exp.conf.imaginator.simplehan_threshold is not None)

        if per_imag is None:
            return sf
        elif per_imag is True:
            return sf and self.exp.conf.imaginator.han_per_imagination
        else:
            return sf and not self.exp.conf.imaginator.han_per_imagination

    def hard_attention(self, ship, planets):
        norms = [x.detach().norm() for x in self.embed(ship, planets)]

        if self.exp.conf.imaginator.han_n_top_objects is not None:
            filter_indices = [False] * len(planets)

            top_indices = sorted(range(len(norms)), key=lambda i: norms[i])[-self.exp.conf.imaginator.han_n_top_objects:]

            for ind in top_indices:
                filter_indices[ind] = True
        elif self.exp.conf.imaginator.adahan_threshold is not None:
            zeros_and_ones = torch.nn.functional.softmax(torch.FloatTensor(norms)) >= self.exp.conf.imaginator.adahan_threshold / len(norms)
            filter_indices = [x.item() == 1 for x in zeros_and_ones]
        elif self.exp.conf.imaginator.simplehan_threshold is not None:
            zeros_and_ones = torch.FloatTensor(norms) >= self.exp.conf.imaginator.simplehan_threshold * torch.FloatTensor(norms).mean()
            filter_indices = [x.item() == 1 for x in zeros_and_ones]

        filtered_planets = [planet for (planet, filter_value) in zip(planets, filter_indices) if filter_value]

        return filtered_planets
