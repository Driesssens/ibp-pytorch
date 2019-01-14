from enum import Enum, IntEnum
from collections import defaultdict

import numpy as np
import torch
import tensorboardX
import os
import datetime
import time
from copy import deepcopy
import json

from spaceship_environment import SpaceshipEnvironment, polar2cartesian
from controller_and_memory import ControllerAndMemory
from imaginator import Imaginator
from manager import Manager
from memory import Memory
from utilities import *


class ImaginationStrategies(Enum):
    ONE_STEP = 1
    N_STEP = 2
    TREE = 3


class ImaginationBasedPlanner:
    def __init__(self,
                 environment: SpaceshipEnvironment,
                 history_embedding_length=100,
                 max_imaginations_per_action=3,
                 imagination_strategy=ImaginationStrategies.TREE,
                 n_episodes_per_batch=20,
                 experiment_name=None,
                 tensorboard=False,
                 train=True,
                 use_controller_and_memory=True,
                 dummy_action_magnitude_interval=(0, 8),
                 use_ship_mass=False,
                 fuel_price=0,
                 refresh_each_batch=False,
                 large_backprop=False,
                 ):
        self.environment = environment

        self.max_imaginations_per_action = max_imaginations_per_action
        self.imagination_strategy = imagination_strategy

        self.n_episodes_per_batch = n_episodes_per_batch

        self.i_episode = 0
        self.i_action_of_episode = 0
        self.history_embedding = torch.randn(history_embedding_length)
        self.episode_metrics = defaultdict(int)

        self.experiment_name = experiment_name if experiment_name is not None else datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.experiment_folder = 'storage\{}'.format(self.experiment_name)

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

        if tensorboard:
            self.tensorboard_writer = tensorboardX.SummaryWriter(self.experiment_folder)
        else:
            self.tensorboard_writer = None

        self.train = train

        # self.manager = Manager(self, history_embedding_length)

        self.use_ship_mass = use_ship_mass
        self.fuel_price = fuel_price
        self.refresh_each_batch = refresh_each_batch
        self.large_backprop = large_backprop

        if use_controller_and_memory:
            self.controller_and_memory = ControllerAndMemory(self)
        else:
            self.controller_and_memory = None
            self.dummy_action_magnitude_interval = dummy_action_magnitude_interval

        self.imaginator = Imaginator(self)
        self.batch_start_time = time.perf_counter()
        self.batch_action_magnitude = Accumulator()

    def refresh(self):
        if self.controller_and_memory is not None:
            self.controller_and_memory = ControllerAndMemory(self)
            self.controller_and_memory.load(self.experiment_name)

        self.imaginator = Imaginator(self)
        self.imaginator.load(self.experiment_name)

        if self.tensorboard_writer is not None:
            self.tensorboard_writer = tensorboardX.SummaryWriter(self.experiment_folder)

    def act(self):
        i_imagination = 0

        last_imagined_ship_state = self.environment.agent_ship
        planets_vector = self.environment.planet_state_vector()

        for _ in range(self.max_imaginations_per_action):
            # i_route = self.manager(np.concatenate([actual_ship_vector] + planet_vectors), self.history_embedding)
            # route = routes_of_strategy[self.imagination_strategy][i_route]

            route = Routes.IMAGINE_FROM_REAL_STATE

            if route is Routes.ACT:
                break
            if route is Routes.IMAGINE_FROM_REAL_STATE:
                last_imagined_ship_state = self.environment.agent_ship
            if route is Routes.IMAGINE_FROM_LAST_IMAGINATION:
                pass

            if self.controller_and_memory is not None:
                imagined_action = self.controller_and_memory.controller(last_imagined_ship_state)
            else:
                imagined_action = self.dummy_action()

            imagined_trajectory, imagined_loss = self.imaginator.imagine(
                last_imagined_ship_state,
                self.environment.planets,
                imagined_action,
                differentiable_trajectory=True
            )

            self.environment.add_imagined_ship_trajectory(imagined_trajectory)

            if self.controller_and_memory is not None:
                self.history_embedding = self.controller_and_memory.memory(
                    route=route_as_vector(self.imagination_strategy, route),
                    actual_state=np.concatenate([self.environment.agent_ship.encode_state(self.use_ship_mass), planets_vector]),
                    last_imagined_state=np.concatenate([last_imagined_ship_state.encode_state(self.use_ship_mass), planets_vector]),
                    action=imagined_action,
                    new_state=tensor_from([imagined_trajectory[-1].encode_state(self.use_ship_mass), tensor_from(planets_vector)]),
                    reward=-imagined_loss.unsqueeze(0),
                    i_action=self.i_action_of_episode,
                    i_imagination=i_imagination
                )

            last_imagined_ship_state = imagined_trajectory[-1]

            i_imagination += 1

        if self.controller_and_memory is not None:
            selected_action = self.controller_and_memory.controller(self.environment.agent_ship)
            detached_action = selected_action.detach().numpy()
        else:
            detached_action = self.dummy_action()

        old_ship_state = deepcopy(self.environment.agent_ship)
        actual_trajectory, actual_fuel_cost, actual_task_loss = self.perform_action(detached_action)
        new_ship_state = actual_trajectory[-1]

        ### REDO THIS AT BETTER PLACE
        actual_task_cost = np.square(new_ship_state.xy_position).mean()
        actual_fuel_cost = max(0, (np.linalg.norm(detached_action) - 8) * self.fuel_price)
        ### REDO THIS AT BETTER PLACE

        if self.controller_and_memory is not None:
            self.history_embedding = self.controller_and_memory.memory(
                route=route_as_vector(self.imagination_strategy, Routes.ACT),
                actual_state=np.concatenate([old_ship_state.encode_state(self.use_ship_mass), planets_vector]),
                last_imagined_state=np.concatenate([old_ship_state.encode_state(self.use_ship_mass), planets_vector]),
                action=selected_action,
                new_state=np.concatenate([new_ship_state.encode_state(self.use_ship_mass), planets_vector]),
                reward=-(actual_task_cost + actual_fuel_cost),
                i_action=self.i_action_of_episode,
                i_imagination=i_imagination
            )

        self.imaginator.compute_loss(actual_trajectory, self.environment.planets, detached_action)

        if self.controller_and_memory is not None:
            estimated_trajectory, critic_evaluation, fuel_cost = self.imaginator.evaluate(old_ship_state, self.environment.planets, selected_action, new_ship_state)
            self.controller_and_memory.accumulate_loss(critic_evaluation, fuel_cost)
        else:
            estimated_trajectory, _, _ = self.imaginator.evaluate(old_ship_state, self.environment.planets, detached_action, new_ship_state)

        self.environment.add_estimated_ship_trajectory(estimated_trajectory)

        self.batch_action_magnitude.add(np.linalg.norm(detached_action))

    def perform_action(self, action):
        fuel_cost = None
        task_loss = None
        resultant_trajectory = [deepcopy(self.environment.agent_ship)]

        for i_physics_step in range(self.environment.n_steps_per_action):
            _, reward, done, _ = self.environment.step(action if i_physics_step == 0 else np.zeros(2))

            resultant_trajectory.append(deepcopy(self.environment.agent_ship))

            if i_physics_step == 0:
                fuel_cost = -reward
            elif done:
                task_loss = -reward

        return resultant_trajectory, fuel_cost, task_loss

    def finish_episode(self):
        # for metric, value in self.episode_metrics.items():
        #     self.tensorboard_writer.add_scalar(metric, value, self.i_episode)
        #     print("{}: {}".format(self.i_episode, value))
        # self.episode_metrics = defaultdict(int)

        if self.train and (self.i_episode + 1) % self.n_episodes_per_batch == 0:
            self.imaginator.finish_batch()

            if self.controller_and_memory is not None:
                self.controller_and_memory.finish_batch()

            now = time.perf_counter()
            seconds_passed = now - self.batch_start_time
            minutes_passed = seconds_passed / 60
            episodes_per_minute = self.n_episodes_per_batch / minutes_passed
            self.log("episodes_per_minute", episodes_per_minute)
            self.batch_start_time = now

            self.log("mean_real_action_magnitude", self.batch_action_magnitude.average())
            self.batch_action_magnitude = Accumulator()

            self.store()

            if self.refresh_each_batch:
                self.refresh()

        self.history_embedding = torch.randn(self.history_embedding.shape)

        if self.controller_and_memory is not None:
            self.controller_and_memory.memory.reset_state()

        self.i_episode += 1

    @property
    def i_episode_of_batch(self):
        return self.i_episode % self.n_episodes_per_batch

    def log(self, name, value):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, self.i_episode)

    def store(self):
        with open(self.experiment_folder + '\\training_status.json', 'w') as file:
            json.dump(self.training_status, file)

        self.imaginator.store()

        if self.controller_and_memory is not None:
            self.controller_and_memory.store()

    def load(self, experiment_name):
        try:
            with open(self.experiment_folder + '\\training_status.json') as file:
                training_status = json.load(file)
                self.i_episode = training_status['i_episode'] + 1
        except:
            pass

        self.imaginator.load(experiment_name)

        try:
            self.controller_and_memory.load(experiment_name)
        except:
            pass

    @property
    def routes_of_strategy(self):
        return routes_of_strategy[self.imagination_strategy]

    @property
    def training_status(self):
        return {'i_episode': self.i_episode}

    def dummy_action(self):
        angle = np.pi * np.random.uniform(0, 2)
        radius = np.random.uniform(*self.dummy_action_magnitude_interval)
        action = np.array(polar2cartesian(angle, radius))
        return action


class Routes(IntEnum):
    ACT = 1
    IMAGINE_FROM_REAL_STATE = 2
    IMAGINE_FROM_LAST_IMAGINATION = 3


routes_of_strategy = {
    ImaginationStrategies.ONE_STEP: [Routes.ACT, Routes.IMAGINE_FROM_REAL_STATE],
    ImaginationStrategies.N_STEP: [Routes.ACT, Routes.IMAGINE_FROM_LAST_IMAGINATION],
    ImaginationStrategies.TREE: [Routes.ACT, Routes.IMAGINE_FROM_REAL_STATE, Routes.IMAGINE_FROM_LAST_IMAGINATION]
}


def route_as_vector(strategy, route):
    boolean_vector_encoding = np.array(routes_of_strategy[strategy]) == route
    integer_vector_encoding = boolean_vector_encoding.astype(int)
    return integer_vector_encoding
