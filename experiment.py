import json
import os
import itertools
import tensorboardX
import torch
import numpy as np
import time
from configuration import *
from imagination_based_planner import ImaginationBasedPlanner
from spaceship_environment import SpaceshipEnvironment


class Experiment:
    @classmethod
    def new(cls,
            configuration: GeneralConfiguration,
            name,
            path=('storage', 'home', 'misc',),
            pretrained_imaginator=None,
            pretrained_controller=None
            ):
        new_experiment = cls()
        new_experiment.name = name
        new_experiment.path = path
        new_experiment.conf = configuration

        agent = ImaginationBasedPlanner(new_experiment)

        if pretrained_controller is not None and pretrained_imaginator is None:
            pretrained_imaginator = pretrained_controller

        if configuration.imaginator is not None:
            agent.imaginator = configuration.imaginator.the_class(new_experiment)
        elif pretrained_imaginator is not None:
            loaded = cls.load(pretrained_imaginator[0], pretrained_imaginator[1])
            new_experiment.conf.imaginator = loaded.conf.imaginator
            agent.imaginator = loaded.agent.imaginator
            agent.imaginator.exp = new_experiment

        if configuration.controller is not None:
            agent.controller_and_memory = configuration.controller.the_class(new_experiment)
        elif pretrained_controller is not None:
            loaded = cls.load(pretrained_imaginator[0], pretrained_imaginator[1])
            new_experiment.conf.controller = loaded.conf.controller
            agent.controller_and_memory = loaded.agent.controller_and_memory
            agent.controller_and_memory.exp = new_experiment
            agent.controller_and_memory.controller.exp = new_experiment
            agent.controller_and_memory.memory.exp = new_experiment

        if configuration.manager is not None:
            agent.manager = configuration.manager.the_class(new_experiment)

        agent.history_embedding = torch.zeros(new_experiment.conf.history_embedding_length)
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
    def load(cls, name, path=('storage', 'home', 'misc',), initialized_and_silenced=False, specific_instance=None):
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

        if specific_instance is None:
            # HACKY STUFF
            try:
                with open(loaded_experiment.file_path('training_status.json')) as file:
                    training_status = json.load(file)
                    root_episode = training_status['i_episode'] + 1
                    # print("root_episode: {}".format(root_episode))
            except:
                root_episode = 0

            subfolders = [int(f.name) for f in os.scandir(loaded_experiment.directory_path()) if f.is_dir()]

            if len(subfolders) == 0:
                last_episode = root_episode
            else:
                # print("subfolders: {}".format(subfolders))
                last_episode = str(max(root_episode, max(subfolders)))
                # print("last_episode: {}".format(last_episode))

            old_path = loaded_experiment.path
            old_name = loaded_experiment.name

            if root_episode < int(last_episode):
                loaded_experiment.path += (loaded_experiment.name,)
                loaded_experiment.name = '{}'.format(last_episode)

            # print("path and name: {}, {}".format(loaded_experiment.path, loaded_experiment.name))
            # / HACKY STUFF
        else:
            old_path = loaded_experiment.path
            old_name = loaded_experiment.name

            loaded_experiment.path += (loaded_experiment.name,)
            loaded_experiment.name = '{}'.format(str(specific_instance))

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

        # HACKY STUFF
        loaded_experiment.path = old_path
        loaded_experiment.name = old_name
        # print("path and name: {}, {}".format(loaded_experiment.path, loaded_experiment.name))
        # / HACKY STUFF

        if initialized_and_silenced:
            loaded_experiment.initialize_and_silence()

        return loaded_experiment

    def initialize_and_silence(self):
        self.initialize_environment()
        self.env.reset()
        self.env.render_after_each_step = False
        self.train_model = False
        self.store_model = False
        self.tensorboard_writer = None

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

    def train(self, n_episodes=-1, measure_performance_every_n_episodes=2000, measure_performance_n_sample_episodes=1000, continuous_store=False, sparse_report=False):
        if sparse_report and not continuous_store:
            print("Warning: sparse report should only be used with continuous store. Will now disable sparse report")
            sparse_report = False

        print("training {} for {} episodes".format(self.name, n_episodes))

        self.initialize_environment()

        self.env.render_after_each_step = False
        self.train_model = True
        self.store_model = not sparse_report

        self.tensorboard_writer = tensorboardX.SummaryWriter(self.directory_path())

        first = True

        for i_episode in itertools.count():
            self.env.reset()

            for i_action in range(self.conf.n_actions_per_episode):
                self.agent.act()

            if sparse_report and not ((self.agent.i_episode + 1) % measure_performance_every_n_episodes == 0):
                old_tensorboard_writer = self.tensorboard_writer
                self.tensorboard_writer = None

            self.agent.finish_episode()

            if sparse_report and not self.agent.i_episode % measure_performance_every_n_episodes == 0:
                self.tensorboard_writer = old_tensorboard_writer

            if i_episode == n_episodes:
                break

            if measure_performance_every_n_episodes != 0:
                if self.agent.i_episode % measure_performance_every_n_episodes == 0:
                    self.measure_performance(measure_performance_n_sample_episodes, first)
                    first = False

                    if continuous_store:
                        old_path = self.path
                        old_name = self.name

                        self.agent.i_episode -= 1

                        self.path += (self.name,)
                        self.name = '{}'.format(self.agent.i_episode)
                        os.makedirs(self.directory_path())

                        self.agent.store_model(training_status_update=True)

                        self.agent.i_episode += 1

                        self.path = old_path
                        self.name = old_name

    def render(self, slp=0):
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
            time.sleep(slp)

    def evaluate(self, n_episodes=-1):
        print("evaluating {} for {} episodes".format(self.name, n_episodes))

        self.initialize_environment()

        self.env.render_after_each_step = False
        self.train_model = False
        self.store_model = False

        self.tensorboard_writer = None

        with torch.no_grad():
            for i_episode in itertools.count():
                self.env.reset()

                for i_action in range(self.conf.n_actions_per_episode):
                    self.agent.act()

                self.agent.finish_episode()

                if i_episode == n_episodes:
                    break

    def measure_performance(self, n_episodes, first=False):
        stashed_train_model = self.train_model
        stashed_store_model = self.store_model
        stashed_tensorboard_writer = self.tensorboard_writer
        stashed_i_episode = self.agent.i_episode

        self.train_model = False
        self.store_model = False
        self.tensorboard_writer = None

        self.agent.measure_performance = True
        self.agent.imaginator_mean_final_position_error_measurements = []
        self.agent.controller_and_memory_mean_task_cost_measurements = []
        self.agent.manager_mean_task_cost_measurements = []

        with torch.no_grad():
            for _ in range(n_episodes):
                self.env.reset()

                for i_action in range(self.conf.n_actions_per_episode):
                    self.agent.act()

                self.agent.finish_episode()

        self.agent.i_episode = stashed_i_episode
        self.store_model = stashed_store_model
        self.train_model = stashed_train_model
        self.tensorboard_writer = stashed_tensorboard_writer

        if first:
            layout = {}

            for thing in ('imaginator', 'controller', 'manager'):
                layout[thing] = {
                    'extremes': ['Margin', ['{}/mean'.format(thing), '{}/min'.format(thing), '{}/max'.format(thing)]],
                    'percentile_10': ['Margin', ['{}/mean'.format(thing), '{}/10_percentile'.format(thing), '{}/90_percentile'.format(thing)]],
                    'percentile_25': ['Margin', ['{}/mean'.format(thing), '{}/25_percentile'.format(thing), '{}/75_percentile'.format(thing)]],
                }

            self.tensorboard_writer.add_custom_scalars(layout)

        if self.conf.imaginator is not None:
            imaginator_performance = np.stack(self.agent.imaginator_mean_final_position_error_measurements)

            self.log('imaginator/mean', imaginator_performance.mean())
            self.log('imaginator/min', imaginator_performance.min())
            self.log('imaginator/max', imaginator_performance.max())
            self.log('imaginator/10_percentile', np.percentile(imaginator_performance, 10))
            self.log('imaginator/90_percentile', np.percentile(imaginator_performance, 90))
            self.log('imaginator/25_percentile', np.percentile(imaginator_performance, 25))
            self.log('imaginator/75_percentile', np.percentile(imaginator_performance, 75))
            self.log('imaginator/variance', imaginator_performance.var())

        if self.conf.controller is not None:
            controller_performance = np.stack(self.agent.controller_and_memory_mean_task_cost_measurements)

            self.log('controller/mean', controller_performance.mean())
            self.log('controller/min', controller_performance.min())
            self.log('controller/max', controller_performance.max())
            self.log('controller/10_percentile', np.percentile(controller_performance, 10))
            self.log('controller/90_percentile', np.percentile(controller_performance, 90))
            self.log('controller/25_percentile', np.percentile(controller_performance, 25))
            self.log('controller/75_percentile', np.percentile(controller_performance, 75))
            self.log('controller/variance', controller_performance.var())

        if self.conf.manager is not None and not self.agent.manager_delayed():
            manager_performance = np.stack(self.agent.manager_mean_task_cost_measurements)

            self.log('manager/mean', manager_performance.mean())
            self.log('manager/min', manager_performance.min())
            self.log('manager/max', manager_performance.max())
            self.log('manager/10_percentile', np.percentile(manager_performance, 10))
            self.log('manager/90_percentile', np.percentile(manager_performance, 90))
            self.log('manager/25_percentile', np.percentile(manager_performance, 25))
            self.log('manager/75_percentile', np.percentile(manager_performance, 75))
            self.log('manager/variance', manager_performance.var())

        del self.agent.measure_performance
        del self.agent.imaginator_mean_final_position_error_measurements
        del self.agent.controller_and_memory_mean_task_cost_measurements
        del self.agent.manager_mean_task_cost_measurements

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
            n_secondary_planets=self.conf.n_secondary_planets,
            secondary_planets_random_mass_interval=self.conf.secondary_planets_random_mass_interval,
            secondary_planets_random_radial_distance_interval=self.conf.secondary_planets_random_radial_distance_interval,
            with_beacons=self.conf.with_beacons,
            beacon_probability=self.conf.beacon_probability,
            beacon_radial_distance_interval=self.conf.beacon_radial_distance_interval,
            cap_gravity=self.conf.gravity_cap
        )

    def log(self, name, value):
        if self.tensorboard_writer is not None and value is not None:
            self.tensorboard_writer.add_scalar(name, value, self.agent.i_episode)
