from spaceship_environment import SpaceshipEnvironment, GravityCap, polar2cartesian
import time
import numpy as np
from spaceship_environment import AcademicPapers, GravityCap

game = SpaceshipEnvironment(
    default_settings_from=AcademicPapers.LearningModelBasedPlanningFromScratch,
    n_planets=3,
    n_actions_per_episode=1,
    store_episode_as_gif=False,
    # render_window_size=400,
    implicit_euler=False,
    euler_scale=1,
    render_after_each_step=True,
    agent_ship_random_mass_interval=(0.25, 0.25),
    n_secondary_planets=0,
    with_beacons=True,
    beacon_probability=1,
    cap_gravity=GravityCap.Realistic,
    # render_window_size=333
    render_window_size=215,
    n_ice_rocks=4
)

# configuration = GeneralConfiguration(
#     fuel_price=0,
#     max_imaginations_per_action=4,
#     gravity_cap=GravityCap.Realistic,
#     with_beacons=True,
#     beacon_probability=0 + (0.5 if settings['game'] == 'beac' else 0) + (0.5 if settings['game'] == 'all' else 0),
#     n_ice_rocks=0 + (4 if settings['game'] == '4ext' else 0) + (4 if settings['game'] == 'all' else 0) + (10 if settings['game'] == '10ext' else 0),
#     controller=controller_configuration,
#     imaginator=ImaginatorConfiguration()
# )

game.mass_to_pixel_ratio /= 2
game.reset()

action_required = False

while True:

    magnitude = 0

    if action_required:
        angle = np.pi * np.random.uniform(0, 2)
        radius = magnitude
        action = polar2cartesian(angle, radius)
        observation, reward, done, _ = game.step(action)

        # observation, reward, done, _ = game.step([np.random.uniform(-magnitude, magnitude), np.random.uniform(-magnitude, magnitude)])
    else:
        observation, reward, done, _ = game.step([0, 0])

        action_required = observation["action_required"]

    # print(game.objs())
    # print([obj.ide for obj in game.objs()])

    if done:
        time.sleep(0.5)

        game.reset()
