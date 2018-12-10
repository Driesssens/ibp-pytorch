from enum import Enum, IntEnum

import numpy as np
import torch

from controller import Controller
from imaginator import Imaginator
from manager import Manager
from memory import Memory


class ImaginationStrategies(Enum):
    ONE_STEP = 1
    N_STEP = 2
    TREE = 3


class ImaginationBasedPlanner:
    def __init__(self,
                 n_planets=5,
                 object_vector_length=5,
                 action_vector_length=2,
                 history_embedding_length=100,
                 n_actions_per_episode=3,
                 n_physics_steps_per_action=12,
                 max_imaginations_per_action=3,
                 imagination_strategy=ImaginationStrategies.TREE,
                 only_imagine_agent_ship_velocity=True
                 ):
        state_vector_length = object_vector_length * (n_planets + 1)
        route_vector_length = len(routes_of_strategy[imagination_strategy])

        self.manager = Manager(state_vector_length, history_embedding_length, route_vector_length)
        self.controller = Controller(state_vector_length, history_embedding_length, action_vector_length)
        self.imaginator = Imaginator(object_vector_length, action_vector_length, only_imagine_agent_ship_velocity, n_physics_steps_per_action)
        self.memory = Memory(route_vector_length, state_vector_length, action_vector_length)

        self.max_imaginations_per_action = max_imaginations_per_action
        self.imagination_strategy = imagination_strategy
        self.n_actions_per_episode = n_actions_per_episode

        self.history_embedding = torch.randn(history_embedding_length)
        self.i_action = 0

        self.last_actual_agent_ship_vector = None
        self.last_i_imagination = None

    def act(self, agent_ship, planets, agent_ship_trajectory=None):
        imagined_trajectories = []
        planet_vectors = [planet.encode_state() for planet in planets]

        actual_agent_ship_vector = agent_ship.encode_state()
        last_imagined_agent_ship_vector = actual_agent_ship_vector

        for i_imagination in range(self.max_imaginations_per_action):
            i_route = self.manager(np.concatenate([actual_agent_ship_vector] + planet_vectors), self.history_embedding)
            route = routes_of_strategy[self.imagination_strategy][i_route]

            if route is Routes.ACT:
                break
            if route is Routes.IMAGINE_FROM_REAL_STATE:
                last_imagined_agent_ship_vector = actual_agent_ship_vector
            if route is Routes.IMAGINE_FROM_LAST_IMAGINATION:
                pass

            imagined_action = self.controller(np.concatenate([last_imagined_agent_ship_vector] + planet_vectors), self.history_embedding)

            new_imagined_agent_ship_vector, imagined_fuel_cost, imagined_task_loss, imagined_trajectory = self.imaginator(
                last_imagined_agent_ship_vector,
                planet_vectors,
                imagined_action
            )

            self.history_embedding = self.memory(
                route=route_as_vector(self.imagination_strategy, route),
                actual_state=np.concatenate([actual_agent_ship_vector] + planet_vectors),
                last_imagined_state=np.concatenate([last_imagined_agent_ship_vector] + planet_vectors),
                action=imagined_action,
                new_state=np.concatenate([new_imagined_agent_ship_vector] + planet_vectors),
                reward=-(imagined_fuel_cost + imagined_task_loss) if self.last_action_of_episode() else -imagined_fuel_cost,
                i_action=self.i_action,
                i_imagination=i_imagination
            )

            imagined_trajectories.append(imagined_trajectory)

            last_imagined_agent_ship_vector = new_imagined_agent_ship_vector

        selected_action = self.controller(np.concatenate([last_imagined_agent_ship_vector] + planet_vectors), self.history_embedding)

        self.i_action += 1
        self.last_actual_agent_ship_vector = actual_agent_ship_vector
        self.last_i_imagination = len(imagined_trajectories) - 1

        return selected_action.detach(), imagined_trajectories

    def observe_results(self, action, agent_ship, planets, reward):
        planet_vectors = [planet.encode_state() for planet in planets]
        new_agent_ship_vector = agent_ship.encode_state()

        self.history_embedding = self.memory(
            route=route_as_vector(self.imagination_strategy, Routes.ACT),
            actual_state=np.concatenate([self.last_actual_agent_ship_vector] + planet_vectors),
            last_imagined_state=np.concatenate([self.last_actual_agent_ship_vector] + planet_vectors),
            action=action,
            new_state=np.concatenate([new_agent_ship_vector] + planet_vectors),
            reward=reward,
            i_action=self.i_action - 1,
            i_imagination=self.last_i_imagination
        )

        self.last_actual_agent_ship_vector = None
        self.last_i_imagination = None

    def new_episode(self):
        self.history_embedding = torch.randn(self.history_embedding.shape)
        self.i_action = 0

        self.memory.reset_state()

    def last_action_of_episode(self):
        return self.i_action == self.n_actions_per_episode - 1


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
