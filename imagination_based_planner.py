from enum import Enum, IntEnum
from collections import defaultdict

import numpy as np
import torch
import tensorboardX
import os
import datetime
from copy import deepcopy

from spaceship_environment import SpaceshipEnvironment
from controller import Controller
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
                 ):
        self.environment = environment

        # self.manager = Manager(self, history_embedding_length)
        # self.controller = Controller(self, history_embedding_length)
        self.imaginator = Imaginator(self)
        # self.memory = Memory(self)

        self.max_imaginations_per_action = max_imaginations_per_action
        self.imagination_strategy = imagination_strategy

        self.n_episodes_per_batch = n_episodes_per_batch

        self.i_episode = 0
        self.i_action_of_episode = 0
        self.history_embedding = torch.randn(history_embedding_length)
        self.episode_metrics = defaultdict(int)

        self.experiment_name = experiment_name if experiment_name is not None else datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        self.experiment_folder = 'storage\{}'.format(self.experiment_name)

        if tensorboard:
            if not os.path.exists(self.experiment_folder):
                os.makedirs(self.experiment_folder)
            self.tensorboard_writer = tensorboardX.SummaryWriter(self.experiment_folder)
        else:
            self.tensorboard_writer = None

    def act(self):
        i_imagination = 0
        last_imagined_ship_state = self.environment.agent_ship

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

            # imagined_action = self.controller(last_imagined_ship_state, self.environment.planets, self.history_embedding)
            imagined_action = np.zeros(2)

            imagined_trajectory = self.imaginator.imagine(
                last_imagined_ship_state,
                self.environment.planets,
                imagined_action
            )

            self.environment.add_imagined_ship_trajectory([(ship_state.x, ship_state.y) for ship_state in imagined_trajectory])

            # self.history_embedding = self.memory(
            #     route=route_as_vector(self.imagination_strategy, route),
            #     actual_state=np.concatenate([actual_ship_vector] + planet_vectors),
            #     last_imagined_state=np.concatenate([last_imagined_ship_vector] + planet_vectors),
            #     action=imagined_action,
            #     new_state=np.concatenate([new_imagined_ship_vector] + planet_vectors),
            #     reward=-(imagined_fuel_cost + imagined_task_loss) if self.last_action_of_episode() else -imagined_fuel_cost,
            #     i_action=self.i_action_of_episode,
            #     i_imagination=i_imagination
            # )

            last_imagined_ship_state = imagined_trajectory[-1]

            i_imagination += 1

        # selected_action = self.controller(np.concatenate([last_imagined_ship_vector] + planet_vectors), self.history_embedding)
        # detached_action = selected_action.detach().numpy()
        detached_action = np.zeros(2)

        actual_trajectory, actual_fuel_cost, actual_task_loss = self.perform_action(detached_action)

        # self.history_embedding = self.memory(
        #     route=route_as_vector(self.imagination_strategy, Routes.ACT),
        #     actual_state=np.concatenate([actual_ship_vector] + planet_vectors),
        #     last_imagined_state=np.concatenate([actual_ship_vector] + planet_vectors),
        #     action=selected_action,
        #     new_state=np.concatenate([environment.agent_ship.encode_state()] + planet_vectors),
        #     reward=-(actual_fuel_cost + actual_task_loss) if actual_task_loss is not None else -actual_fuel_cost,
        #     i_action=self.i_action_of_episode,
        #     i_imagination=i_imagination
        # )

        self.imaginator.compute_loss(actual_trajectory, self.environment.planets, detached_action)

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

    def new_episode(self):
        # for metric, value in self.episode_metrics.items():
        #     self.tensorboard_writer.add_scalar(metric, value, self.i_episode)
        #     print("{}: {}".format(self.i_episode, value))
        # self.episode_metrics = defaultdict(int)

        if (self.i_episode + 1) % self.n_episodes_per_batch == 0:
            self.imaginator.finish_batch()

        self.history_embedding = torch.randn(self.history_embedding.shape)

        # self.memory.reset_state()
        self.i_episode += 1

    def log(self, name, value):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, self.i_episode)

    def store(self):
        self.imaginator.store()

    def load(self, experiment_name):
        self.imaginator.load(experiment_name)


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
