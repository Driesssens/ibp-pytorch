from enum import Enum, IntEnum
import numpy as np

from controller_and_memory import ControllerAndMemory
from imaginator import Imaginator
from manager import Manager


class ImaginationStrategies(Enum):
    ONE_STEP = 1
    N_STEP = 2
    TREE = 3


class Configuration:
    the_class = None

    @classmethod
    def from_dict(cls, settings):
        instance = cls(**settings)
        return instance

    def to_dict(self):
        return self.__dict__


class ControllerConfiguration(Configuration):
    the_class = ControllerAndMemory

    def __init__(self,
                 learning_rate=0.003,
                 max_gradient_norm=10,
                 hidden_layer_sizes=(100, 100),
                 immediate_mode=True
                 ):
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.hidden_layer_sizes = hidden_layer_sizes
        self.immediate_mode = immediate_mode


class ImaginatorConfiguration(Configuration):
    the_class = Imaginator

    def __init__(self,
                 relation_module_layer_sizes=(150, 150, 150, 150),
                 effect_embedding_length=100,
                 object_module_layer_sizes=(100,),
                 velocity_normalization_factor=7.5,
                 action_normalization_factor=0.5,
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 batch_loss_sum=False
                 ):
        self.relation_module_layer_sizes = relation_module_layer_sizes
        self.effect_embedding_length = effect_embedding_length
        self.object_module_layer_sizes = object_module_layer_sizes
        self.velocity_normalization_factor = velocity_normalization_factor
        self.action_normalization_factor = action_normalization_factor
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.batch_loss_sum = batch_loss_sum


class ManagerConfiguration(Configuration):
    the_class = Manager

    def __init__(self,
                 hidden_layer_sizes=(100, 100),
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 entropy_factor=0.2,
                 ponder_price=0.05
                 ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.entropy_factor = entropy_factor
        self.ponder_price = ponder_price


class GeneralConfiguration:
    @classmethod
    def from_dict(cls, general_settings, imaginator_settings=None, controller_settings=None, manager_settings=None) -> 'GeneralConfiguration':
        instance = cls(**general_settings)

        if not isinstance(instance.imagination_strategy, ImaginationStrategies):
            instance.imagination_strategy = ImaginationStrategies(instance.imagination_strategy)

        if instance.imaginator is not None:
            instance.imaginator = eval(instance.imaginator)(**imaginator_settings)

        if instance.controller is not None:
            instance.controller = eval(instance.controller)(**controller_settings)

        if instance.manager is not None:
            instance.manager = eval(instance.manager)(**manager_settings)

        return instance

    def to_dict(self):
        settings = dict(self.__dict__)

        settings['imagination_strategy'] = settings['imagination_strategy'].value

        for key in ('imaginator', 'controller', 'manager'):
            if settings[key] is not None:
                settings[key] = settings[key].__class__.__name__

        return settings

    def __init__(self,
                 n_planets=3,
                 n_actions_per_episode=1,
                 fuel_price=0.0004,
                 fuel_cost_threshold=8,
                 agent_ship_random_mass_interval=(0.25, 0.25),
                 agent_ship_random_radial_distance_interval=(0.6, 1.0),
                 planets_random_mass_interval=(0.08, 0.4),
                 planets_random_radial_distance_interval=(0.4, 1.0),
                 history_embedding_length=100,
                 max_imaginations_per_action=3,
                 imagination_strategy=ImaginationStrategies.ONE_STEP,
                 n_episodes_per_batch=20,
                 dummy_action_magnitude_interval=(0, 10),
                 use_ship_mass=False,
                 imaginator=None,
                 controller=None,
                 manager=None
                 ):
        self.n_planets = n_planets
        self.n_actions_per_episode = n_actions_per_episode
        self.fuel_price = fuel_price
        self.fuel_cost_threshold = fuel_cost_threshold
        self.agent_ship_random_mass_interval = agent_ship_random_mass_interval
        self.agent_ship_random_radial_distance_interval = agent_ship_random_radial_distance_interval
        self.planets_random_mass_interval = planets_random_mass_interval
        self.planets_random_radial_distance_interval = planets_random_radial_distance_interval
        self.history_embedding_length = history_embedding_length
        self.max_imaginations_per_action = max_imaginations_per_action
        self.imagination_strategy = imagination_strategy
        self.n_episodes_per_batch = n_episodes_per_batch
        self.dummy_action_magnitude_interval = dummy_action_magnitude_interval
        self.use_ship_mass = use_ship_mass
        self.imaginator = imaginator  # type: ImaginatorConfiguration
        self.controller = controller  # type: ControllerConfiguration
        self.manager = manager  # type: ManagerConfiguration

    @property
    def routes_of_strategy(self):
        return routes_of_strategy[self.imagination_strategy]


class Routes(IntEnum):
    ACT = 1
    IMAGINE_FROM_REAL_STATE = 2
    IMAGINE_FROM_LAST_IMAGINATION = 3


routes_of_strategy = {
    ImaginationStrategies.ONE_STEP: [Routes.ACT, Routes.IMAGINE_FROM_REAL_STATE],
    ImaginationStrategies.N_STEP: [Routes.ACT, Routes.IMAGINE_FROM_LAST_IMAGINATION],
    ImaginationStrategies.TREE: [Routes.ACT, Routes.IMAGINE_FROM_REAL_STATE, Routes.IMAGINE_FROM_LAST_IMAGINATION]
}


def route_as_vector(strategy, route):
    boolean_vector_encoding = np.array(routes_of_strategy[strategy]) == route
    integer_vector_encoding = boolean_vector_encoding.astype(int)
    return integer_vector_encoding
