import json
import time
from copy import deepcopy

from controller_and_memory import ControllerAndMemory
from imaginator import Imaginator
from manager import Manager, PPOManager
from spaceship_environment import polar2cartesian
from configuration import route_as_vector, Routes
from utilities import *

if False:
    from experiment import Experiment


class ImaginationBasedPlanner:
    def __init__(self, experiment: 'Experiment'):
        self.exp = experiment  # type: 'Experiment'

        self.i_episode = 0
        self.history_embedding = torch.zeros(self.exp.conf.history_embedding_length)

        self.imaginator = None  # type: Imaginator
        self.controller_and_memory = None  # type: ControllerAndMemory
        self.manager = None  # type: Manager

        self.batch_n_imaginations_per_action = Accumulator()
        self.batch_start_time = time.perf_counter()
        self.batch_action_magnitude = Accumulator()

        self.batch_n_planets_in_each_imagination = Accumulator()
        self.batch_f_planets_in_each_imagination = Accumulator()
        self.batch_n_planets_in_final_imagination = Accumulator()
        self.batch_f_planets_in_final_imagination = Accumulator()

    def act(self):
        i_imagination = 0

        last_imagined_ship_state = self.exp.env.agent_ship
        last_filtered_planets = self.exp.env.planets
        last_threshold = None
        filter_indices = None

        for _ in range(self.exp.conf.max_imaginations_per_action):
            if self.has_normal_manager():
                i_route, threshold = self.manager.act()
                route = self.exp.conf.routes_of_strategy[i_route] if i_route is not None else Routes.IMAGINE_FROM_REAL_STATE

                if route is Routes.ACT:
                    self.manager.episode_costs.append(0)
                else:
                    self.manager.episode_costs.append(self.exp.conf.manager.ponder_price)
            else:
                route = Routes.IMAGINE_FROM_REAL_STATE

            if route is Routes.ACT:
                break
            elif route is Routes.IMAGINE_FROM_REAL_STATE:
                last_imagined_ship_state = self.exp.env.agent_ship
            elif route is Routes.IMAGINE_FROM_LAST_IMAGINATION:
                pass

            if self.controller_and_memory is not None:
                imagined_action = self.controller_and_memory.controller(last_imagined_ship_state)
            else:
                imagined_action = self.dummy_action()

            filtered_planets = self.exp.env.planets

            if self.has_normal_manager() and threshold is not None:
                print(threshold)
                filtered_planets = self.imaginator.filter(last_imagined_ship_state, filtered_planets, threshold)
                self.manager.batch_threshold.add(threshold)
                last_threshold = threshold

            imagined_trajectory, imagined_loss, imagined_fuel_cost = self.imaginator.imagine(
                last_imagined_ship_state,
                filtered_planets,
                imagined_action,
                differentiable_trajectory=True
            )

            self.exp.env.add_imagined_ship_trajectory(imagined_trajectory)

            if self.has_ppo_manager():
                objects = [imagined_trajectory[-1]] + self.exp.env.planets

                if filter_indices is None:
                    filter_indices = [True] * len(objects)

                for i in range(len(filter_indices)):
                    if not filter_indices[i]:
                        objects[i] = None

                object_embeddings = self.controller_and_memory.memory.get_object_embeddings(objects)

                threshold = self.manager.act(list(filter(lambda x: x is not None, object_embeddings)))
                last_threshold = threshold

                filter_indices = [filter_value and threshold < embedding.norm().item() for filter_value, embedding in zip(filter_indices, object_embeddings)]

            if self.controller_and_memory is not None:
                self.history_embedding = self.controller_and_memory.memory(
                    route=route_as_vector(self.exp.conf.imagination_strategy, route),
                    actual_state=self.exp.env.agent_ship,
                    last_imagined_state=last_imagined_ship_state,
                    action=imagined_action,
                    new_state=imagined_trajectory[-1],
                    reward=-(imagined_loss.unsqueeze(0) + imagined_fuel_cost),
                    i_action=self.exp.env.i_action,
                    i_imagination=i_imagination,
                    filter_indices=filter_indices
                )

            if self.has_normal_manager() and threshold is not None:
                n_kept_planets = len(filtered_planets)
                fraction_kept_planets = n_kept_planets / len(self.exp.env.planets)
                self.manager.episode_costs[-1] *= (n_kept_planets + 1) / (len(self.exp.env.planets) + 1)

                self.batch_n_planets_in_each_imagination.add(n_kept_planets)
                self.batch_f_planets_in_each_imagination.add(fraction_kept_planets)

            if self.has_ppo_manager():
                n_kept_objects = len(list(filter(lambda x: x, filter_indices)))  # count amount of True
                fraction_kept_objects = n_kept_objects / len(filter_indices)
                self.manager.episode_costs.append(self.exp.conf.manager.ponder_price * (n_kept_objects + 1) / (len(filter_indices) + 1))
                self.batch_n_planets_in_each_imagination.add(n_kept_objects)
                self.batch_f_planets_in_each_imagination.add(fraction_kept_objects)

            last_imagined_ship_state = imagined_trajectory[-1]

            i_imagination += 1

        if self.controller_and_memory is not None:
            selected_action = self.controller_and_memory.controller(self.exp.env.agent_ship)
            detached_action = selected_action.detach().numpy()
        else:
            detached_action = self.dummy_action()

        old_ship_state = deepcopy(self.exp.env.agent_ship)
        actual_trajectory, actual_fuel_cost, actual_task_cost = self.perform_action(detached_action)
        new_ship_state = actual_trajectory[-1]

        if (self.controller_and_memory is not None) and (not self.exp.env.i_action == self.exp.env.n_actions_per_episode):
            self.history_embedding = self.controller_and_memory.memory(
                route=route_as_vector(self.exp.conf.imagination_strategy, Routes.ACT),
                actual_state=old_ship_state,
                last_imagined_state=old_ship_state,
                action=selected_action,
                new_state=new_ship_state,
                reward=-(actual_task_cost + actual_fuel_cost),
                i_action=self.exp.env.i_action - 1,
                i_imagination=i_imagination,
            )

        if self.manager is not None:
            internal_cost = sum(self.manager.episode_costs)
            self.manager.batch_ponder_cost.add(internal_cost)

            external_cost = actual_task_cost + actual_fuel_cost
            self.manager.episode_costs[-1] += external_cost

            task_cost = internal_cost + external_cost
            self.manager.batch_task_cost.add(task_cost)

            if hasattr(self, 'measure_performance'):
                self.manager_mean_task_cost_measurements.append(task_cost)

            if last_threshold is not None:
                self.manager.batch_final_threshold.add(last_threshold)

        self.imaginator.accumulate_loss(actual_trajectory, last_filtered_planets, detached_action)

        if self.controller_and_memory is not None:
            estimated_trajectory, critic_evaluation, fuel_cost = self.imaginator.evaluate(old_ship_state, self.exp.env.planets, selected_action, new_ship_state)
            self.controller_and_memory.accumulate_loss(critic_evaluation, fuel_cost)
        else:
            estimated_trajectory, _, _ = self.imaginator.evaluate(old_ship_state, self.exp.env.planets, detached_action, new_ship_state)

        self.exp.env.add_estimated_ship_trajectory(estimated_trajectory)
        self.batch_n_imaginations_per_action.add(i_imagination)
        self.batch_action_magnitude.add(np.linalg.norm(detached_action))

        if self.manager is not None:
            self.batch_n_planets_in_final_imagination.add(len(filtered_planets))
            self.batch_f_planets_in_final_imagination.add(len(filtered_planets) / len(self.exp.env.planets))

    def perform_action(self, action, compute_cost_independently=True):
        resultant_fuel_cost = None
        resultant_task_loss = None
        resultant_trajectory = [deepcopy(self.exp.env.agent_ship)]

        for i_physics_step in range(self.exp.env.n_steps_per_action):
            _, reward, done, _ = self.exp.env.step(action if i_physics_step == 0 else np.zeros(2))

            resultant_trajectory.append(deepcopy(self.exp.env.agent_ship))

            if i_physics_step == 0:
                resultant_fuel_cost = -reward
            elif done:
                resultant_task_loss = -reward

        if compute_cost_independently:
            computed_task_cost = np.square(resultant_trajectory[-1].xy_position).mean()
            computed_fuel_cost = max(0, (np.linalg.norm(action) - self.exp.conf.fuel_cost_threshold) * self.exp.conf.fuel_price)

            return resultant_trajectory, computed_task_cost, computed_fuel_cost
        else:
            return resultant_trajectory, resultant_fuel_cost, resultant_task_loss

    def finish_episode(self):
        if self.manager is not None:
            self.manager.finish_episode()

        if (self.i_episode + 1) % self.exp.conf.n_episodes_per_batch == 0:
            self.finish_batch()

        self.history_embedding = torch.zeros(self.history_embedding.shape)

        if self.controller_and_memory is not None:
            self.controller_and_memory.memory.reset_state()

        self.i_episode += 1

    def has_normal_manager(self):
        return self.manager is not None and isinstance(self.manager, Manager)

    def has_ppo_manager(self):
        return self.manager is not None and isinstance(self.manager, PPOManager)

    def finish_batch(self):
        self.imaginator.finish_batch()

        if self.controller_and_memory is not None:
            self.controller_and_memory.finish_batch()

        if self.manager is not None:
            self.manager.finish_batch()

        self.exp.log("mean_n_imaginations_per_action", self.batch_n_imaginations_per_action.average())
        self.batch_n_imaginations_per_action = Accumulator()

        self.exp.log("mean_real_action_magnitude", self.batch_action_magnitude.average())
        self.batch_action_magnitude = Accumulator()

        if self.manager is not None:
            self.exp.log("mean_n_planets_in_each_imagination", self.batch_n_planets_in_each_imagination.average())
            self.exp.log("mean_f_planets_in_each_imagination", self.batch_f_planets_in_each_imagination.average())
            self.exp.log("mean_n_planets_in_final_imagination", self.batch_n_planets_in_final_imagination.average())
            self.exp.log("mean_f_planets_in_final_imagination", self.batch_f_planets_in_final_imagination.average())

        self.batch_n_planets_in_each_imagination = Accumulator()
        self.batch_f_planets_in_each_imagination = Accumulator()
        self.batch_n_planets_in_final_imagination = Accumulator()
        self.batch_f_planets_in_final_imagination = Accumulator()

        self.exp.log("get_num_threads", torch.get_num_threads())

        if self.exp.store_model:
            self.store_model()

        now = time.perf_counter()
        minutes_passed = (now - self.batch_start_time) / 60
        episodes_per_minute = self.exp.conf.n_episodes_per_batch / minutes_passed
        self.exp.log("episodes_per_minute", episodes_per_minute)
        self.batch_start_time = now

    @property
    def i_episode_of_batch(self):
        return self.i_episode % self.exp.conf.n_episodes_per_batch

    def store_model(self):
        with open(self.exp.file_path('training_status.json'), 'w') as file:
            json.dump(self.training_status, file, indent=2)

        self.imaginator.store_model()

        if self.controller_and_memory is not None:
            self.controller_and_memory.store_model()

        if self.manager is not None:
            self.manager.store_model()

    def load_model(self):
        with open(self.exp.file_path('training_status.json')) as file:
            training_status = json.load(file)
            self.i_episode = training_status['i_episode'] + 1

    @property
    def training_status(self):
        return {'i_episode': self.i_episode}

    def dummy_action(self):
        angle = np.pi * np.random.uniform(0, 2)
        radius = np.random.uniform(*self.exp.conf.dummy_action_magnitude_interval)
        action = np.array(polar2cartesian(angle, radius))
        return action
