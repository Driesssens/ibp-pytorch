from manager import Manager
from imaginator import Imaginator
from memory import Memory
from imagination_based_planner import ImaginationBasedPlanner
from spaceship_environment import SpaceshipEnvironment
import numpy as np


def print_test_name(name):
    print("-----------")
    print(name)
    print("-----------")


def test_forward_manager():
    print_test_name("test_forward_manager")

    object_vector_length = 5
    n_planets = 5
    state_vector_length = object_vector_length * (n_planets + 1)

    history_embedding_length = 100
    route_vector_length = 3

    manager = Manager(
        state_vector_length=state_vector_length,
        history_embedding_length=history_embedding_length,
        route_vector_length=route_vector_length
    )

    state_vector = np.zeros(state_vector_length)
    history_vector = np.zeros(history_embedding_length)

    for _ in range(12):
        print("Route: {}".format(manager(state_vector, history_vector)))


def test_forward_imaginator():
    print_test_name("test_forward_imaginator")

    object_vector_length = 5
    action_vector_length = 2
    imaginator = Imaginator(
        object_vector_length=object_vector_length,
        action_vector_length=action_vector_length,
        only_imagine_agent_ship_velocity=True,
        n_physics_steps_per_action=12
    )

    ship_vector = np.zeros(object_vector_length)
    planet_vectors = [np.zeros(object_vector_length) for _ in range(5)]
    action_vector = np.zeros(action_vector_length)

    print(imaginator(ship_vector, planet_vectors, action_vector))


def test_forward_memory():
    print_test_name("test_forward_memory")

    object_vector_length = 5
    n_planets = 5
    state_vector_length = object_vector_length * (n_planets + 1)
    action_vector_length = 2

    memory = Memory(route_vector_length=3, state_vector_length=state_vector_length, action_vector_length=action_vector_length)

    ship_vector = np.zeros(object_vector_length)
    planet_vectors = [np.zeros(object_vector_length) for _ in range(n_planets)]
    state_vector = np.concatenate([ship_vector] + planet_vectors)
    action_vector = np.zeros(action_vector_length)

    for i_episode in range(2):
        memory.reset_state()

        for i_action in range(3):
            for i_imagination in range(2):
                history_embedding = memory(np.array([0, 1, 0]), state_vector, state_vector, action_vector, state_vector, 0.8, i_action, i_imagination)
                print("Ep {}, j={}, k={}: {}".format(i_episode, i_action, i_imagination, history_embedding))


def test_forward_ibp():
    print_test_name("test_forward_ibp")

    n_actions_per_episode = 3
    n_planets = 2

    environment = SpaceshipEnvironment(n_planets=n_planets, n_actions_per_episode=n_actions_per_episode)
    agent = ImaginationBasedPlanner(n_planets=n_planets, n_actions_per_episode=n_actions_per_episode)

    for i_episode in range(2):
        state = environment.reset()
        agent.new_episode()

        for i_action in range(n_actions_per_episode):
            action, imagined_trajectories = agent.act(state['agent_ship'], state['planets'])
            cumulative_reward = 0

            for i_physics_step in range(environment.n_steps_per_action):
                if i_physics_step == 0:
                    state, reward, done, _ = environment.step(action)
                    cumulative_reward += reward
                else:
                    state, reward, done, _ = environment.step()
                    cumulative_reward += reward

            agent.observe_results(action, state['agent_ship'], state['planets'], cumulative_reward)

            print("Imagined {} times before action {} in episode {}".format(len(imagined_trajectories), i_action, i_episode))


test_forward_manager()
test_forward_imaginator()
test_forward_memory()
test_forward_ibp()
