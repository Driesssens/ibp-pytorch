from manager import Manager
from imaginator import Imaginator
from memory import Memory
from imagination_based_planner import ImaginationBasedPlanner
from spaceship_environment import SpaceshipEnvironment, cartesian2polar, polar2cartesian
import numpy as np
import torch
from spaceship_environment import AcademicPapers
import time

def big_test():
    print("batch test")
    # print("EM: {}".format("test_imaginator_dummy-action-0-10_actnorm-0.5_lr-0.001_batch-100_reg0_no-mass-0.25"))

    experiment_name = "batchtest_bootstrapped_batch200_imag-2_EMlr0.001_others-default_no-reg_no-mass-0.25_fuel-0.0004"

    n_planets = 3

    environment = SpaceshipEnvironment(n_planets, render_after_each_step=False, agent_ship_random_mass_interval=(0.25, 0.25))
    agent = ImaginationBasedPlanner(environment, experiment_name=experiment_name, tensorboard=True, max_imaginations_per_action=2, n_episodes_per_batch=250, train=True, use_controller_and_memory=True, use_ship_mass=False, fuel_price=0.0004, refresh_each_batch=False)
    # agent.load(experiment_name)
    # agent.imaginator.load("test_imaginator_dummy-action-0-10_actnorm-0.5_lr-0.001_batch-100_reg0_no-mass-0.25")

    for i_episode in range(100000):
        environment.reset()

        for i_action in range(environment.n_actions_per_episode):
            agent.act()

        agent.finish_episode()
