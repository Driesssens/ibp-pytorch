from spaceship_environment import SpaceshipEnvironment, GravityCap, polar2cartesian
import time
import numpy as np
from spaceship_environment import AcademicPapers, GravityCap

game = SpaceshipEnvironment(
    default_settings_from=AcademicPapers.LearningModelBasedPlanningFromScratch,
    n_planets=3,
    n_actions_per_episode=3,
    store_episode_as_gif=False,
    # render_window_size=400,
    implicit_euler=False,
    euler_scale=1,
    render_after_each_step=True,
    agent_ship_random_mass_interval=(0.25, 0.25),
    with_beacons=False,
    beacon_probability=0.0
)

game.reset()

action_required = False

while True:

    magnitude = 8

    if action_required:
        angle = np.pi * np.random.uniform(0, 2)
        radius = magnitude
        action = polar2cartesian(angle, radius)
        observation, reward, done, _ = game.step(action)

        # observation, reward, done, _ = game.step([np.random.uniform(-magnitude, magnitude), np.random.uniform(-magnitude, magnitude)])
    else:
        observation, reward, done, _ = game.step([0, 0])

        action_required = observation["action_required"]

    if done:
        game.reset()
