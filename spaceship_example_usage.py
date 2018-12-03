from spaceship_environment import SpaceshipEnvironment
import time
import numpy as np
from spaceship_environment import AcademicPapers

game = SpaceshipEnvironment(
    n_planets=5,
    n_actions_per_episode=3,
    store_episode_as_gif=True,
    render_window_size=500
)

game.reset()
game.render()

action_required = False

while True:
    game.render()
    time.sleep(0.05)

    magnitude = 50000

    if action_required:
        observation, reward, done, _ = game.step([np.random.uniform(-magnitude, magnitude), np.random.uniform(-magnitude, magnitude)])
    else:
        observation, reward, done, _ = game.step([0, 0])

        action_required = observation["action_required"]

    game.render()

    if done:
        game.reset()
        game.render()
