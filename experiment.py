import json
import os
import itertools
import tensorboardX

from configuration import *
from imagination_based_planner import ImaginationBasedPlanner
from spaceship_environment import SpaceshipEnvironment


class Experiment:
    @classmethod
    def new(cls, configuration: GeneralConfiguration, name, path=('storage', 'misc',)):
        new_experiment = cls()
        new_experiment.name = name
        new_experiment.path = path
        new_experiment.conf = configuration

        agent = ImaginationBasedPlanner(new_experiment)

        if configuration.imaginator is not None:
            agent.imaginator = configuration.imaginator.the_class(new_experiment)

        if configuration.controller is not None:
            agent.controller_and_memory = configuration.controller.the_class(new_experiment)

        if configuration.manager is not None:
            agent.manager = configuration.manager.the_class(new_experiment)

        new_experiment.agent = agent

        os.makedirs(new_experiment.directory_path())

        with open(new_experiment.file_path('configuration.json'), 'x') as file:
            json.dump(new_experiment.conf.to_dict(), file, indent=2)

        if new_experiment.conf.imaginator is not None:
            with open(new_experiment.file_path('configuration_imaginator.json'), 'x') as file:
                json.dump(new_experiment.conf.imaginator.to_dict(), file, indent=2)

        if new_experiment.conf.controller is not None:
            with open(new_experiment.file_path('configuration_controller.json'), 'x') as file:
                json.dump(new_experiment.conf.controller.to_dict(), file, indent=2)

        if new_experiment.conf.manager is not None:
            with open(new_experiment.file_path('configuration_manager.json'), 'x') as file:
                json.dump(new_experiment.conf.manager.to_dict(), file, indent=2)

        return new_experiment

    @classmethod
    def load(cls, name, path=('storage', 'misc',)):
        loaded_experiment = cls()
        loaded_experiment.name = name
        loaded_experiment.path = path

        with open(loaded_experiment.file_path('configuration.json')) as file:
            general_settings = json.load(file)

        imaginator_settings = None
        if general_settings['imaginator'] is not None:
            with open(loaded_experiment.file_path('configuration_imaginator.json')) as file:
                imaginator_settings = json.load(file)

        controller_settings = None
        if general_settings['controller'] is not None:
            with open(loaded_experiment.file_path('configuration_controller.json')) as file:
                controller_settings = json.load(file)

        manager_settings = None
        if general_settings['manager'] is not None:
            with open(loaded_experiment.file_path('configuration_manager.json')) as file:
                manager_settings = json.load(file)

        configuration = GeneralConfiguration.from_dict(general_settings, imaginator_settings, controller_settings, manager_settings)
        loaded_experiment.conf = configuration

        agent = ImaginationBasedPlanner(loaded_experiment)
        agent.load_model()

        if configuration.imaginator is not None:
            agent.imaginator = configuration.imaginator.the_class(loaded_experiment)
            agent.imaginator.load_model()

        if configuration.controller is not None:
            agent.controller_and_memory = configuration.controller.the_class(loaded_experiment)
            agent.controller_and_memory.load_model()

        if configuration.manager is not None:
            agent.manager = configuration.manager.the_class(loaded_experiment)
            agent.manager.load_model()

        loaded_experiment.agent = agent

        return loaded_experiment

    def __init__(self):
        self.conf = None  # type: GeneralConfiguration
        self.name = None
        self.path = None
        self.agent = None  # type: ImaginationBasedPlanner
        self.env = None  # type: SpaceshipEnvironment

        self.tensorboard_writer = None  # type: tensorboardX.SummaryWriter
        self.train_model = None
        self.store_model = None

    def directory_path(self):
        return os.path.join(*self.path, self.name)

    def file_path(self, file_name):
        return os.path.join(self.directory_path(), file_name)

    def train(self, n_episodes=-1):
        print("training {} for {} episodes".format(self.name, n_episodes))

        self.initialize_environment()

        self.env.render_after_each_step = False
        self.train_model = True
        self.store_model = True

        self.tensorboard_writer = tensorboardX.SummaryWriter(self.directory_path())

        for i_episode in itertools.count():
            self.env.reset()

            for i_action in range(self.conf.n_actions_per_episode):
                self.agent.act()

            self.agent.finish_episode()

            if i_episode == n_episodes:
                break

    def render(self):
        print("rendering {}".format(self.name))

        self.initialize_environment()

        self.env.render_after_each_step = True
        self.train_model = False
        self.store_model = False

        self.tensorboard_writer = None

        while True:
            self.env.reset()

            for i_action in range(self.conf.n_actions_per_episode):
                self.agent.act()

            self.agent.finish_episode()

    def initialize_environment(self):
        self.env = SpaceshipEnvironment(
            n_planets=self.conf.n_planets,
            n_actions_per_episode=self.conf.n_actions_per_episode,
            fuel_price=self.conf.fuel_price,
            fuel_cost_threshold=self.conf.fuel_cost_threshold,
            agent_ship_random_mass_interval=self.conf.agent_ship_random_mass_interval,
            agent_ship_random_radial_distance_interval=self.conf.agent_ship_random_radial_distance_interval,
            planets_random_mass_interval=self.conf.planets_random_mass_interval,
            planets_random_radial_distance_interval=self.conf.planets_random_radial_distance_interval,
        )

    def log(self, name, value):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar(name, value, self.agent.i_episode)
