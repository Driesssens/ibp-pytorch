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
            agent.finish_episode()

        for i_action in range(n_actions_per_episode):
            imagined_trajectories = agent.act(environment)

            # print("Imagined {} times before action {} in episode {}".format(len(imagined_trajectories), i_action, i_episode))

        # if i_episode == 7500:
        #     environment.render_after_each_step = True
        #     agent.store_imaginator()

    if evaluate_model is None:
        agent.store_imaginator()


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


def evalu(name):
    print("evaluate")
    experiment_name = name
    print("system: {}".format(name))

    n_planets = 3

    environment = SpaceshipEnvironment(n_planets, render_after_each_step=True, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=500, train=False,
                                    use_controller_and_memory=True, dummy_action_magnitude_interval=(0, 10), use_ship_mass=True)
    agent.imaginator.action_normalization_factor = 0.5
    agent.load(name)

    for i_episode in range(14000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def test_new_style():
    print("test_imaginator_fixed_mass")
    experiment_name = "test_imaginator_dummy-action-0-10_actnorm-0.5_lr-0.001_batch-100_reg0_no-mass-0.25"
    print("experiment: {}".format(experiment_name))

    n_planets = 3

    environment = SpaceshipEnvironment(n_planets, render_after_each_step=False, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=True, max_imaginations_per_action=0, n_episodes_per_batch=100, train=True, use_controller_and_memory=False, dummy_action_magnitude_interval=(0, 10), use_ship_mass=False)
    agent.load(experiment_name)
    agent.imaginator.action_normalization_factor = 0.5

    # for g in agent.imaginator.optimizer.param_groups:
    #     g['lr'] = 0.002

    for i_episode in range(50000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def big_test():
    print("more backprop on bootstrapped EM")
    # print("EM: {}".format("test_imaginator_dummy-action-0-10_actnorm-0.5_lr-0.001_batch-100_reg0_no-mass-0.25"))

    experiment_name = "morebackprop_bootstrapped_batch200_imag-2_EMlr0.001_others-default_no-reg_no-mass-0.25_fuel-0.0004"

    n_planets = 3

    environment = SpaceshipEnvironment(n_planets, render_after_each_step=False, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=True, max_imaginations_per_action=2, n_episodes_per_batch=250, train=True, use_controller_and_memory=True, use_ship_mass=False, fuel_price=0.0004, refresh_each_batch=False)
    agent.load(experiment_name)
    # agent.imaginator.load("test_imaginator_dummy-action-0-10_actnorm-0.5_lr-0.001_batch-100_reg0_no-mass-0.25")

    for i_episode in range(100000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


def bug_test():
    print("bugtest")

    experiment_name = "bugtest"

    n_planets = 3

    environment = SpaceshipEnvironment(n_planets, render_after_each_step=False, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=False, max_imaginations_per_action=2, n_episodes_per_batch=200, train=True, use_controller_and_memory=True)
    agent.imaginator.action_normalization_factor = 0.5

    for i_episode in range(100000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()


# test_new_style()
# evalu("5_full-system_pre-trained_batch200_imag-2_EMlr0.001_others-default_no-reg_fixedmass-0.25")
big_test()
# bug_test()