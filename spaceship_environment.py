import gym
from enum import Enum
import numpy as np
import arcade
import pyglet
from pyglet.gl import *
import os
import imageio
from datetime import datetime


class AcademicPapers(Enum):
    LearningModelBasedPlanningFromScratch = 1
    MetacontrolForAdaptiveImaginationBasedOptimization = 2


SETTINGS_FROM_PAPERS = {
    AcademicPapers.MetacontrolForAdaptiveImaginationBasedOptimization: {
        "n_actions_per_episode": 1,
        "euler_method_step_size": 0.05,
        "gravitational_constant": 1000000,
        "damping_constant": 0.1,
        "fuel_price": None,
        "fuel_cost_threshold": None,
        "agent_ship_random_mass_interval": (1, 9),
        "agent_ship_random_radial_distance_interval": (150, 250),
        "planets_random_mass_interval": (20, 50),
        "planets_random_radial_distance_interval": (100, 250),
        "sun_mass": 100,
        "sun_random_radial_distance_interval": (100, 200)
    },
    AcademicPapers.LearningModelBasedPlanningFromScratch: {
        "n_actions_per_episode": 1,
        "euler_method_step_size": 0.05,
        "gravitational_constant": "UNKNOWN",  # todo: ???
        "damping_constant": "UNKNOWN",  # todo: ???
        "fuel_price": 0.0002,
        "fuel_cost_threshold": 8,
        "agent_ship_random_mass_interval": (0.004, 0.36),
        "agent_ship_random_radial_distance_interval": (0.6, 1.0),
        "planets_random_mass_interval": (0.08, 0.4),
        "planets_random_radial_distance_interval": (0.4, 1.0),
        "sun_mass": None,
        "sun_random_radial_distance_interval": None
    }
}

DEFAULT = object()  # Set init argument to DEFAULT to get setting from the chosen paper's experimental setup


class SpaceshipEnvironment(gym.Env):
    """An OpenAI Gym environment of the Spaceship task.

    In this continuous control task, a spaceship floats amongst a handful of stationary planets. The agent's goal is to get the spaceship as close as possible to the goal position (the center at x=0, y=0) by the end of the episode. It can fire its thrusters every few timesteps with a certain force and direction. Apart from that, the ship is subject to the gravitational pull of the planets (which themselves don't move).

    Optionally, a fuel cost can be set, such that the agent needs to reach the goal using a minimum amount of thruster force.

    Observation:
        Dict(3) {
            'action_required' (boolean): Whether this is a step in which the agent can execute an action, i.e. fire the ships thrusters. When False, actions passed to step() will be ignored.
            'agent_ship' (`AgentShip` object): The current state of the agent ship.
            'planets' (list of `Planet` objects: The current state of the planets.
        }

        You can get Numpy vector representations with `AgentShip.encode_as_vector()` and `Planet.encode_as_vector()`.

    Actions:
        Continuous(2) [
            0   Thruster force in the x direction
            1	Thruster force in the y direction
        ]

    Example usage:
        game = SpaceshipEnvironment(n_planets = 3)

        observation = game.reset()
        game.render()

        agent = YourAlgorithm()

        while True:
            if observation["action_required"]:  # Indicates whether this is a timestep at which the thrusters can be fired
                action = agent.forward(observation)
            else:
                action = None

            observation, reward, done, _ = game.step(action)
            game.render()

            if done:
                observation = game.reset()
                game.render()
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 n_planets,
                 default_settings_from=AcademicPapers.MetacontrolForAdaptiveImaginationBasedOptimization,
                 n_actions_per_episode=DEFAULT,
                 n_steps_per_action=12,
                 euler_method_step_size=DEFAULT,
                 euler_method_step_resolution_scale=1,
                 gravitational_constant=DEFAULT,
                 damping_constant=DEFAULT,
                 fuel_price=DEFAULT,
                 fuel_cost_threshold=DEFAULT,
                 agent_ship_random_mass_interval=DEFAULT,
                 agent_ship_random_radial_distance_interval=DEFAULT,
                 planets_random_mass_interval=DEFAULT,
                 planets_random_radial_distance_interval=DEFAULT,
                 sun_mass=DEFAULT,
                 sun_random_radial_distance_interval=DEFAULT,
                 render_window_size=900,
                 store_episode_as_gif=False,
                 gif_file_name=""):
        """
        Args:
            n_planets (int): The number of planets randomly positioned in each episode.
            default_settings_from (enum): The academic paper of which the settings are used for arguments set to `DEFAULT`. You can overrule any settings you want; anything you don't set will be as in this paper's experiments.
            n_actions_per_episode (int): The number of (real, non-imagined) actions the agent executes per episode.
            n_steps_per_action (int): If e.g. 12, the agent executes an action at one timestep and then waits for 11 steps of physics simulations before executing the next.
            euler_method_step_size (float): The time resolution (smaller is more fine-grained / less movement per step) at which the physics are simulated.
            gravitational_constant (float): The magnitude of gravity. See C.2 in "Metacontrol for adaptive imagination-based optimization".
            damping_constant (float): Controls inertia; dampens the effect of gravity on the spaceship. See C.2 in "Metacontrol for adaptive imagination-based optimization".
            fuel_price (float): Adds linearly to the agent's loss when it uses a force larger than `fuel_cost_threshold`: fuel cost = max(0, `fuel_price` * (force - `fuel_cost_threshold`)).
            fuel_cost_threshold (float): See above.
            agent_ship_random_mass_interval (2-tuple of floats): The interval from which the agent ship's mass will be uniformly sampled for each episode.
            agent_ship_random_radial_distance_interval (2-tuple of floats): The interval at which the agent's initial position will be sampled for each episode. First, the radial distance from the goal position is uniformly sampled from `agent_ship_random_radial_distance_interval`. Next, the angle is uniformly sampled from [0, 360].
            planets_random_mass_interval (2-tuple of floats): See above.
            planets_random_radial_distance_interval (2-tuple of floats): See above.
            sun_mass (float): When not None, the first planet's mass will not be randomly sampled, but set to `sun_mass`.
            sun_random_radial_distance_interval (float): When not None, the first planet's distance will be sampled from `sun_random_radial_distance_interval` instead of `planets_random_radial_distance_interval`.
            render_window_size (int): The height and width of the window when rendering (window is always square).
            store_episode_as_gif (boolean): When True, all finished episodes are stored as animated GIFs.
            gif_file_name (string): When not None, all animated GIF file names will be prefixed with `gif_file_name`. Otherwise, a time stamp will be used.
        """
        parameters = locals().copy()

        # Bit of an unconventional init method, but it enables two sets of default argument values depending on `default_settings_from`. Any argument set to `DEFAULT` will get its value from the dict above.
        for parameter, value in parameters.items():
            if value is DEFAULT:
                self.__setattr__(parameter, SETTINGS_FROM_PAPERS[default_settings_from][parameter])
            else:
                self.__setattr__(parameter, value)

        self.agent_ship = None
        self.planets = None
        self.i_step = None
        self.episode_cumulative_loss = None

        self.render_window = None
        self.lowest_zoom_factor_this_episode = None

        # A list of lists of xy-positions: one list for every `n_steps_per_action` steps. For rendering the ship's trajectory through space.
        self.past_agent_ship_positions = None

        # Determines how zoomed out the rendering initially is.
        self.minimally_visible_world_size = 2.5 * max([
            radius_interval[1] for radius_interval
            in [self.agent_ship_random_radial_distance_interval, self.planets_random_radial_distance_interval, self.sun_random_radial_distance_interval]
            if radius_interval is not None
        ])

        self.mass_to_pixel_ratio = 50 / max([
            mass for mass
            in [self.agent_ship_random_mass_interval[1], self.planets_random_mass_interval[1], self.sun_mass]
            if mass is not None
        ])

        self.minimally_visible_world_radius = self.minimally_visible_world_size / 2
        self.render_window_radius = render_window_size / 2

        if not self.gif_file_name:
            self.gif_file_name = 'experiment_{date:%Y%m%d_%H%M%S}'.format(date=datetime.now())

        self.gif_file_name += '-'

        self.i_episode = -1

    def reset(self):
        self.agent_ship = AgentShip(
            random_mass_interval=self.agent_ship_random_mass_interval,
            random_radial_distance_interval=self.agent_ship_random_radial_distance_interval
        )

        self.planets = [
            Planet(
                random_mass_interval=self.planets_random_mass_interval,
                random_radial_distance_interval=self.planets_random_radial_distance_interval
            )
            for _ in range(self.n_planets)
        ]

        if self.n_planets > 0 and self.sun_random_radial_distance_interval:
            self.planets[0] = Planet(
                random_mass_interval=self.planets_random_mass_interval,
                random_radial_distance_interval=self.sun_random_radial_distance_interval
            )

        if self.n_planets > 0 and self.sun_mass:
            self.planets[0].mass = self.sun_mass

        self.i_step = 0
        self.lowest_zoom_factor_this_episode = 1
        self.past_agent_ship_positions = []
        self.episode_cumulative_loss = 0

        self.i_episode += 1

        return self.observation()

    def observation(self):
        return {
            'action_required': self.first_step_of_action(),
            'agent_ship': self.agent_ship,
            'planets': self.planets
        }

    def step(self, xy_thrust_force):
        if not self.first_step_of_action():
            # Ignore the action when this is a timestep at which the thrusters may not be fired.
            xy_thrust_force = np.zeros(2)
        else:
            # This is the first timestep after a new action; initialize the list of past positions for trajectory visualization.
            self.past_agent_ship_positions.append([(self.agent_ship.x, self.agent_ship.y)])

        xy_gravitational_forces = [
            self.gravitational_constant * planet.mass * self.agent_ship.mass
            / np.linalg.norm(self.agent_ship.xy_position - planet.xy_position) ** 3
            * (planet.xy_position - self.agent_ship.xy_position)
            for planet in self.planets
        ]

        xy_acceleration = (sum(xy_gravitational_forces) - self.damping_constant * self.agent_ship.xy_velocity + xy_thrust_force) / self.agent_ship.mass

        self.agent_ship.xy_position += self.euler_method_step_size / self.euler_method_step_resolution_scale * self.agent_ship.xy_velocity
        self.agent_ship.xy_velocity += self.euler_method_step_size / self.euler_method_step_resolution_scale * xy_acceleration

        self.past_agent_ship_positions[-1].append((self.agent_ship.x, self.agent_ship.y))

        self.i_step += 1

        fuel_cost = max(
            0,
            (np.absolute(xy_thrust_force).sum() - self.fuel_cost_threshold) * self.fuel_price
        ) if self.fuel_price else 0

        if self.done():
            task_loss = np.linalg.norm(self.agent_ship.xy_position)
            loss = fuel_cost + task_loss
        else:
            loss = fuel_cost

        self.episode_cumulative_loss += loss

        return self.observation(), -loss, self.done(), {}  # Negated loss as OpenAI Gym convention is to return reward.

    def render(self, mode='human', close=False):
        if mode == 'human':
            pass
        else:
            return super().render(mode=mode)  # Raises exception. Gym convention to do it like this apparently.

        if close:
            if self.render_window:
                self.render_window.close()

        if self.render_window is None:
            self.render_window = arcade.Window(self.render_window_size, self.render_window_size)

        arcade.start_render()
        pyglet.clock.tick()

        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()

            arcade.draw_lrtb_rectangle_filled(0, self.render_window_size, self.render_window_size, 0, arcade.color.BLACK)  # Otherwise GIF file will have transparent background

            for planet in self.planets:
                arcade.draw_circle_outline(self.screen_position(planet.x), self.screen_position(planet.y), self.screen_size(planet.mass), arcade.color.RED)

            for i_trajectory, trajectory in enumerate(self.past_agent_ship_positions):
                # Draw the agent ship's past trajectory this episode as a green trailing line. Green gets lighter after each successive action

                for i_from in range(len(trajectory) - 1):
                    i_to = i_from + 1

                    extra_darkness = 1 if i_from == 0 else 0  # This part of the trajectory is before the new action kicks in, so needs darker shade of green
                    darkness = 0 + 1 * (i_trajectory + 1 - extra_darkness) / self.n_actions_per_episode

                    arcade.draw_line(
                        self.screen_position(trajectory[i_from][0]),
                        self.screen_position(trajectory[i_from][1]),
                        self.screen_position(trajectory[i_to][0]),
                        self.screen_position(trajectory[i_to][1]),
                        (int(darkness * 0), int(darkness * 255), int(darkness * 0)),
                        max(1, self.screen_size(self.agent_ship.mass))
                    )

            arcade.draw_circle_filled(self.screen_position(self.agent_ship.x), self.screen_position(self.agent_ship.y), max(1, self.screen_size(self.agent_ship.mass)), arcade.color.WHITE)

        arcade.finish_render()

        if self.store_episode_as_gif:
            self.store_gif_frame()

            if self.done():
                self.store_gif_animation()

    def seed(self, given_seed=None):
        if given_seed is None:
            np.random.seed()
            return np.random.get_state()
        elif isinstance(given_seed, tuple):
            np.random.set_state(given_seed)
            return given_seed
        elif isinstance(given_seed, int):
            np.random.seed(int)
            return given_seed
        else:
            raise ValueError("Argument given_seed needs to be None, a tuple, or an integer.")

    def first_step_of_action(self):
        # Indicates whether this is the first of `n_steps_per_action`, so thrusters may be fired and the agent needs to select an action.
        return self.i_step % self.n_steps_per_action == 0

    def done(self):
        # Indicates whether the current episode is finished.
        return self.i_step >= self.n_actions_per_episode * self.n_steps_per_action

    def screen_position(self, position):
        # Converts environment x and y positions to pixel positions on screen for rendering.
        return position / self.minimally_visible_world_radius * self.zoom_factor() * self.render_window_radius + self.render_window_radius

    def screen_size(self, mass):
        # Gives screen pixel size according to object's mass.
        return mass * self.mass_to_pixel_ratio * self.zoom_factor()

    def zoom_factor(self):
        # Zoom factor to ensure the agent ship is always visible on screen.
        furthest_point = max(abs(self.agent_ship.x), abs(self.agent_ship.y))
        border = self.minimally_visible_world_radius - self.minimally_visible_world_radius * 0.1
        zoom_factor = min(self.lowest_zoom_factor_this_episode, border / furthest_point)
        self.lowest_zoom_factor_this_episode = zoom_factor
        return zoom_factor

    def store_gif_frame(self):
        # Stores GIF frames to be compiled to one animated GIF after episode is done. You can delete the temporary_gif_frames folder afterwards if you don't need the frames.
        if not os.path.exists('temporary_gif_frames\{}ep{}'.format(self.gif_file_name, self.i_episode)):
            os.makedirs('temporary_gif_frames\{}ep{}'.format(self.gif_file_name, self.i_episode))

        pyglet.image.get_buffer_manager().get_color_buffer().save('temporary_gif_frames\{}ep{}\st{:03}.png'.format(self.gif_file_name, self.i_episode, self.i_step))

    def store_gif_animation(self):
        file_names = sorted(os.listdir('temporary_gif_frames\{}ep{}'.format(self.gif_file_name, self.i_episode)))
        frames = [imageio.imread('temporary_gif_frames\{}ep{}\{}'.format(self.gif_file_name, self.i_episode, file_name)) for file_name in file_names]

        if not os.path.exists('gifs'):
            os.makedirs('gifs')

        imageio.mimsave('gifs\{}ep{}-loss_{:.2f}.gif'.format(self.gif_file_name, self.i_episode, self.episode_cumulative_loss), frames)


class SpaceObject:
    type = None

    class Types(Enum):
        AGENT_SHIP = 1
        PLANET = 2

    def __init__(self, random_mass_interval=None, random_radial_distance_interval=None, mass=None, xy_position=None):
        if random_mass_interval:
            self.mass = np.random.uniform(random_mass_interval[0], random_mass_interval[1])
        elif mass:
            self.mass = mass
        else:
            raise ValueError("Either random_mass_interval or mass must be set.")

        if random_radial_distance_interval:
            angle = np.pi * np.random.uniform(0, 2)
            radius = np.random.uniform(random_radial_distance_interval[0], random_radial_distance_interval[1])
            self.xy_position = np.array(polar2cartesian(angle, radius))
        elif xy_position:
            self.xy_position = xy_position
        else:
            raise ValueError("Either random_radial_distance_interval or xy_position must be set.")

        self.xy_velocity = np.zeros(2)

    def encode_type_one_hot(self):
        type_one_hot_encoding = np.zeros(len(SpaceObject.Types))
        type_one_hot_encoding[self.type] = 1
        return type_one_hot_encoding

    def encode_static_properties(self):
        return np.array([self.mass])

    def encode_dynamic_properties(self):
        return np.concatenate((self.xy_position, self.xy_velocity))

    def encode_as_vector(self):
        return np.concatenate((self.encode_type_one_hot(), self.encode_static_properties(), self.encode_dynamic_properties()))

    @property
    def x(self):
        return self.xy_position[0]

    @property
    def y(self):
        return self.xy_position[1]


class AgentShip(SpaceObject):
    type = SpaceObject.Types.AGENT_SHIP


class Planet(SpaceObject):
    type = SpaceObject.Types.PLANET


def cartesian2polar(x, y):
    angle = np.arctan2(y, x)
    radius = np.hypot(x, y)
    return angle, radius


def polar2cartesian(angle, radius):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y