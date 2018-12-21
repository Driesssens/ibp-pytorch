from manager import Manager
from imaginator import Imaginator
from memory import Memory
from imagination_based_planner import ImaginationBasedPlanner
from spaceship_environment import SpaceshipEnvironment, cartesian2polar, polar2cartesian
import numpy as np
import torch
from spaceship_environment import AcademicPapers
import time


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


def test_forward_ibp(evaluate_model=None):
    name = "lr0.003-div5.0-explicit-mean-dynamic_state"
    print_test_name("evaluating {}".format(evaluate_model) if evaluate_model is not None else name)

    n_actions_per_episode = 3
    n_planets = 3

    n_steps_per_action = 12
    euler_step = 0.05

    environment = SpaceshipEnvironment(n_steps_per_action=n_steps_per_action, euler_method_step_size=euler_step, n_planets=n_planets, n_actions_per_episode=n_actions_per_episode, render_after_each_step=False if evaluate_model is None else True, implicit_euler=False, store_episode_as_gif=True,
                                       render_window_size=400)
    agent = ImaginationBasedPlanner(n_physics_steps_per_action=n_steps_per_action, euler_method_step_size=euler_step, n_planets=n_planets, n_actions_per_episode=n_actions_per_episode, tensorboard=True if evaluate_model is None else False, n_episodes_per_minibatch=20,
                                    n_minibatches_per_learning_rate_shrinkage=1000, learning_rate_shrink_factor=0.8, imaginator_learning_rate=0.003,
                                    name=evaluate_model if evaluate_model is not None else name)

    if evaluate_model is not None:
        agent.load_imaginator(evaluate_model)

    for i_episode in range(8500):
        environment.reset()

        if evaluate_model is None:
            agent.new_episode()

        for i_action in range(n_actions_per_episode):
            imagined_trajectories = agent.act(environment)

            # print("Imagined {} times before action {} in episode {}".format(len(imagined_trajectories), i_action, i_episode))

        # if i_episode == 7500:
        #     environment.render_after_each_step = True
        #     agent.store_imaginator()

    if evaluate_model is None:
        agent.store_imaginator()


def test_new_style():
    experiment_name = "test_new_style_2_faster2_stable-1400-2"

    n_planets = 3

    environment = SpaceshipEnvironment(n_planets, render_after_each_step=True)
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=1, n_episodes_per_batch=40)
    agent.load("test_new_style_2_faster2_stable-1400-2")

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        # if i_episode == 8000:
        #     environment.render_after_each_step = True
    # agent.store()


def analyze_moving_planets(name):
    print_test_name("analyze_moving_planets")

    n_actions_per_episode = 3
    n_planets = 1

    environment = SpaceshipEnvironment(n_planets=n_planets, n_actions_per_episode=n_actions_per_episode, render_after_each_step=True, n_seconds_sleep_per_render=0.01)

    agent = ImaginationBasedPlanner(n_planets=n_planets, n_actions_per_episode=n_actions_per_episode, tensorboard=False, n_episodes_per_minibatch=20, n_minibatches_per_learning_rate_shrinkage=10, learning_rate_shrink_factor=0.85)

    agent.load_imaginator(name)

    for _ in range(1000):
        # print("")

        environment.reset()
        # addition = (np.random.rand(2) - np.array([0.5, 0.5])) / 2
        addition = (np.random.rand(1) - np.array([0.5])) * 2
        addition = np.array([np.pi * 2 / 18])

        for _ in range(40):

            new_imagined_ship_vector, imagined_trajectory = agent.imaginator.imagine(
                environment.agent_ship.encode_state(),
                [planet.encode_state() for planet in environment.planets],
                np.zeros(2)
            )

            environment.add_imagined_ship_trajectory([(ship_vector[1], ship_vector[2]) for ship_vector in imagined_trajectory][::-1])

            for planet in environment.planets:
                distance_vector = planet.xy_position - environment.agent_ship.xy_position
                angle, radius = cartesian2polar(distance_vector[0], distance_vector[1])

                angle += addition

                x, y = polar2cartesian(angle, radius)

                planet.xy_position = np.concatenate((x, y)) + environment.agent_ship.xy_position
                # print(np.linalg.norm(distance_vector))
                environment.update_zoom_factor(planet.x, planet.y)


# test_forward_manager()
# test_forward_imaginator()
# test_forward_memory()
test_new_style()
