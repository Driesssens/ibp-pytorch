from enum import Enum, IntEnum
import numpy as np
from typing import Union

from controller_and_memory import ControllerAndMemory, SetControllerAndFlatMemory, SetControllerAndSetMemory
from imaginator import Imaginator
from full_imaginator import FullImaginator
from manager import Manager, PPOManager
from binary_manager import BinaryManager, BinomialManager
from curator import Curator
from spaceship_environment import GravityCap


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
                 immediate_mode=True,
                 only_ship_state=False,
                 use_route=True,
                 use_actual_state=True,
                 use_last_imagined_state=True,
                 use_action=True,
                 use_new_state=True,
                 use_reward=True,
                 use_i_action=True,
                 use_i_imagination=True,
                 ):
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.hidden_layer_sizes = hidden_layer_sizes
        self.immediate_mode = immediate_mode
        self.only_ship_state = only_ship_state
        self.use_route = use_route
        self.use_actual_state = use_actual_state
        self.use_last_imagined_state = use_last_imagined_state
        self.use_action = use_action
        self.use_new_state = use_new_state
        self.use_reward = use_reward
        self.use_i_action = use_i_action
        self.use_i_imagination = use_i_imagination


class SetControllerAndFlatMemoryConfiguration(Configuration):
    the_class = SetControllerAndFlatMemory

    def __init__(self,
                 learning_rate=0.003,
                 max_gradient_norm=10,
                 relation_module_layer_sizes=(150, 150, 150, 150),
                 effect_embedding_length=100,
                 hide_ship_state=False,
                 control_module_layer_sizes=(100,),
                 velocity_normalization_factor=1,
                 immediate_mode=True,
                 only_ship_state=False,
                 use_route=True,
                 use_actual_state=True,
                 use_last_imagined_state=True,
                 use_action=True,
                 use_new_state=True,
                 use_reward=True,
                 use_i_action=True,
                 use_i_imagination=True,
                 selu=False
                 ):
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.relation_module_layer_sizes = relation_module_layer_sizes
        self.effect_embedding_length = effect_embedding_length
        self.hide_ship_state = hide_ship_state
        self.control_module_layer_sizes = control_module_layer_sizes
        self.velocity_normalization_factor = velocity_normalization_factor
        self.immediate_mode = immediate_mode
        self.only_ship_state = only_ship_state
        self.use_route = use_route
        self.use_actual_state = use_actual_state
        self.use_last_imagined_state = use_last_imagined_state
        self.use_action = use_action
        self.use_new_state = use_new_state
        self.use_reward = use_reward
        self.use_i_action = use_i_action
        self.use_i_imagination = use_i_imagination

        self.selu = selu


class SetControllerAndSetMemoryConfiguration(Configuration):
    the_class = SetControllerAndSetMemory

    def __init__(self,
                 learning_rate=0.003,
                 max_gradient_norm=10,
                 object_function_layer_sizes=(150, 150, 150, 150),
                 object_embedding_length=50,
                 aggregate_function_layer_sizes=(100,),
                 aggregate_embedding_length=50,
                 control_module_layer_sizes=(100,),
                 velocity_normalization_factor=1,
                 immediate_mode=True,
                 use_action=True,
                 use_i_imagination=True,
                 hide_ship_state=True,
                 selu=False,
                 relu_after_aggregate_function=False,
                 memoryless=False,
                 effect_embedding_length=None,  # do not use
                 leaky=False,
                 prelu=False,
                 max_action=None,
                 han_n_top_objects=None,
                 adahan_threshold=None,
                 simplehan_threshold=None,
                 han_per_imagination=False
                 ):
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.immediate_mode = immediate_mode

        self.effect_embedding_length = 0
        self.hide_ship_state = hide_ship_state
        self.control_module_layer_sizes = control_module_layer_sizes

        self.object_function_layer_sizes = object_function_layer_sizes
        self.aggregate_function_layer_sizes = aggregate_function_layer_sizes
        self.object_embedding_length = object_embedding_length
        self.aggregate_embedding_length = aggregate_embedding_length

        self.velocity_normalization_factor = velocity_normalization_factor
        self.use_i_imagination = use_i_imagination
        self.use_action = use_action

        self.relu_after_aggregate_function = relu_after_aggregate_function
        self.memoryless = memoryless

        self.selu = selu
        self.leaky = leaky
        self.prelu = prelu

        self.max_action = max_action

        self.han_n_top_objects = han_n_top_objects
        self.adahan_threshold = adahan_threshold
        self.han_per_imagination = han_per_imagination
        self.simplehan_threshold = simplehan_threshold


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
                 batch_loss_sum=False,
                 han_n_top_objects=None,
                 adahan_threshold=None,
                 simplehan_threshold=None,
                 han_per_imagination=False,
                 c_l2_loss=0,
                 l2_on_filtered=False,
                 l2_scaled_by_filtered=False
                 ):
        self.relation_module_layer_sizes = relation_module_layer_sizes
        self.effect_embedding_length = effect_embedding_length
        self.object_module_layer_sizes = object_module_layer_sizes
        self.velocity_normalization_factor = velocity_normalization_factor
        self.action_normalization_factor = action_normalization_factor
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.batch_loss_sum = batch_loss_sum

        self.han_n_top_objects = han_n_top_objects
        self.adahan_threshold = adahan_threshold
        self.han_per_imagination = han_per_imagination
        self.simplehan_threshold = simplehan_threshold

        self.c_l2_loss = c_l2_loss
        self.l2_on_filtered = l2_on_filtered
        self.l2_scaled_by_filtered = l2_scaled_by_filtered


class FullImaginatorConfiguration(Configuration):
    the_class = FullImaginator

    def __init__(self,
                 relation_module_layer_sizes=(150, 150, 150, 150),
                 effect_embedding_length=100,
                 object_module_layer_sizes=(100,),
                 velocity_normalization_factor=7.5,
                 action_normalization_factor=0.5,
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 batch_loss_sum=False,
                 han_n_top_objects=None,
                 adahan_threshold=None,
                 simplehan_threshold=None,
                 han_per_imagination=False,
                 c_l2_loss=0
                 ):
        self.relation_module_layer_sizes = relation_module_layer_sizes
        self.effect_embedding_length = effect_embedding_length
        self.object_module_layer_sizes = object_module_layer_sizes
        self.velocity_normalization_factor = velocity_normalization_factor
        self.action_normalization_factor = action_normalization_factor
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.batch_loss_sum = batch_loss_sum

        self.han_n_top_objects = han_n_top_objects
        self.adahan_threshold = adahan_threshold
        self.han_per_imagination = han_per_imagination
        self.simplehan_threshold = simplehan_threshold

        self.c_l2_loss = c_l2_loss


class ManagerConfiguration(Configuration):
    the_class = Manager

    def __init__(self,
                 hidden_layer_sizes=(100, 100),
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 entropy_factor=0.2,
                 ponder_price=0.05,
                 manage_n_imaginations=True,
                 manage_planet_filtering=False
                 ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.entropy_factor = entropy_factor
        self.ponder_price = ponder_price
        self.manage_n_imaginations = manage_n_imaginations
        self.manage_planet_filtering = manage_planet_filtering


class PPOManagerConfiguration(Configuration):
    the_class = PPOManager

    def __init__(self,
                 hidden_layer_sizes=(64, 64),
                 initial_gaussian_stddev=1,
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 ponder_price=0.05,
                 n_ppo_epochs=5,
                 ppo_clip=0.2,
                 c_value_estimation_loss=0.5,
                 lower_bounded_actions=False,
                 upper_bounded_actions=False,
                 per_imagination=True,
                 n_steps_delay=0,
                 feature_average_norm=True,
                 feature_cumulative_norm=True,
                 feature_history_embedding=True
                 ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.initial_gaussian_stddev = initial_gaussian_stddev
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.ponder_price = ponder_price
        self.n_ppo_epochs = n_ppo_epochs
        self.ppo_clip = ppo_clip
        self.c_value_estimation_loss = c_value_estimation_loss
        self.lower_bounded_actions = lower_bounded_actions
        self.upper_bounded_actions = upper_bounded_actions
        self.per_imagination = per_imagination
        self.n_steps_delay = n_steps_delay
        self.feature_average_norm = feature_average_norm
        self.feature_cumulative_norm = feature_cumulative_norm
        self.feature_history_embedding = feature_history_embedding and per_imagination


class BinaryManagerConfiguration(Configuration):
    the_class = BinaryManager

    def __init__(self,
                 hidden_layer_sizes=(64, 64),
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 ponder_price=0.05,
                 n_ppo_epochs=5,
                 ppo_clip=0.2,
                 c_value_estimation_loss=0.5,
                 per_imagination=True,
                 feature_controller_embedding=True,
                 feature_norm=True,
                 feature_state=True,
                 feature_history_embedding=True
                 ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.ponder_price = ponder_price
        self.n_ppo_epochs = n_ppo_epochs
        self.ppo_clip = ppo_clip
        self.c_value_estimation_loss = c_value_estimation_loss
        self.per_imagination = per_imagination
        self.feature_controller_embedding = feature_controller_embedding
        self.feature_norm = feature_norm,
        self.feature_state = feature_state
        self.feature_history_embedding = feature_history_embedding and per_imagination


class BinomialManagerConfiguration(Configuration):
    the_class = BinomialManager

    def __init__(self,
                 object_function_layer_sizes=(150, 150, 150, 150),
                 object_embedding_length=50,
                 state_function_layer_sizes=(),
                 state_embedding_length=100,
                 value_layer_sizes=(64,),
                 policy_layer_sizes=(64, 64),
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 ponder_price=0.05,
                 n_ppo_epochs=5,
                 ppo_clip=0.2,
                 c_value_estimation_loss=0.5,
                 per_imagination=True,
                 n_steps_delay=0,
                 c_entropy_bonus=0,
                 feature_controller_embedding=True,
                 feature_norm=True,
                 feature_state=True,
                 feature_n_objects=True,
                 feature_history_embedding=True,
                 feature_state_embedding=True
                 ):
        self.object_function_layer_sizes = object_function_layer_sizes
        self.object_embedding_length = object_embedding_length
        self.state_function_layer_sizes = state_function_layer_sizes
        self.state_embedding_length = state_embedding_length
        self.value_layer_sizes = value_layer_sizes
        self.policy_layer_sizes = policy_layer_sizes

        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.ponder_price = ponder_price
        self.n_ppo_epochs = n_ppo_epochs
        self.ppo_clip = ppo_clip
        self.c_value_estimation_loss = c_value_estimation_loss
        self.per_imagination = per_imagination
        self.n_steps_delay = n_steps_delay
        self.c_entropy_bonus = c_entropy_bonus

        self.feature_controller_embedding = feature_controller_embedding
        self.feature_norm = feature_norm,
        self.feature_state = feature_state
        self.feature_n_objects = feature_n_objects
        self.feature_history_embedding = feature_history_embedding and per_imagination
        self.feature_state_embedding = feature_state_embedding


class CuratorConfiguration(Configuration):
    the_class = Curator

    def __init__(self,
                 object_function_layer_sizes=(150, 150, 150, 150),
                 object_embedding_length=50,
                 state_function_layer_sizes=(),
                 state_embedding_length=100,
                 value_layer_sizes=(64,),
                 policy_layer_sizes=(64, 64),
                 learning_rate=0.001,
                 max_gradient_norm=10,
                 ponder_price=0.05,
                 n_ppo_epochs=5,
                 ppo_clip=0.2,
                 c_value_estimation_loss=0.5,
                 per_physics_step=True,
                 feature_imaginator_embedding=True,
                 feature_norm=True,
                 feature_state=True,
                 feature_n_objects=True,
                 feature_ship_state=True,
                 feature_control=True,
                 feature_state_embedding=True
                 ):
        self.object_function_layer_sizes = object_function_layer_sizes
        self.object_embedding_length = object_embedding_length
        self.state_function_layer_sizes = state_function_layer_sizes
        self.state_embedding_length = state_embedding_length
        self.value_layer_sizes = value_layer_sizes
        self.policy_layer_sizes = policy_layer_sizes

        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.ponder_price = ponder_price
        self.n_ppo_epochs = n_ppo_epochs
        self.ppo_clip = ppo_clip
        self.c_value_estimation_loss = c_value_estimation_loss
        self.per_physics_step = per_physics_step

        self.feature_imaginator_embedding = feature_imaginator_embedding
        self.feature_norm = feature_norm,
        self.feature_state = feature_state
        self.feature_n_objects = feature_n_objects
        self.feature_ship_state = feature_ship_state
        self.feature_control = feature_control
        self.feature_state_embedding = feature_state_embedding


class GeneralConfiguration:
    @classmethod
    def from_dict(cls, general_settings, imaginator_settings=None, controller_settings=None, manager_settings=None) -> 'GeneralConfiguration':
        instance = cls(**general_settings)

        if not isinstance(instance.imagination_strategy, ImaginationStrategies):
            instance.imagination_strategy = ImaginationStrategies(instance.imagination_strategy)

        if not isinstance(instance.gravity_cap, GravityCap):
            instance.gravity_cap = GravityCap(instance.gravity_cap)

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
                 n_secondary_planets=0,
                 secondary_planets_random_mass_interval=(0.00, 0.00),
                 secondary_planets_random_radial_distance_interval=(1.8, 2.0),
                 history_embedding_length=None,
                 _history_embedding_length=100,
                 max_imaginations_per_action=3,
                 imagination_strategy=ImaginationStrategies.ONE_STEP,
                 n_episodes_per_batch=20,
                 dummy_action_magnitude_interval=(0, 10),
                 use_ship_mass=False,
                 imaginator=None,
                 controller=None,
                 manager=None,
                 imaginator_ignores_secondary=False,
                 with_beacons=False,
                 beacon_probability=1,
                 beacon_radial_distance_interval=(0.0, 1.5),
                 gravity_cap=GravityCap.Low
                 ):
        self.n_planets = n_planets
        self.n_actions_per_episode = n_actions_per_episode
        self.fuel_price = fuel_price
        self.fuel_cost_threshold = fuel_cost_threshold
        self.agent_ship_random_mass_interval = agent_ship_random_mass_interval
        self.agent_ship_random_radial_distance_interval = agent_ship_random_radial_distance_interval
        self.planets_random_mass_interval = planets_random_mass_interval
        self.planets_random_radial_distance_interval = planets_random_radial_distance_interval
        self.n_secondary_planets = n_secondary_planets
        self.secondary_planets_random_mass_interval = secondary_planets_random_mass_interval
        self.secondary_planets_random_radial_distance_interval = secondary_planets_random_radial_distance_interval
        self._history_embedding_length = _history_embedding_length if history_embedding_length is None else history_embedding_length
        self.max_imaginations_per_action = max_imaginations_per_action
        self.imagination_strategy = imagination_strategy
        self.n_episodes_per_batch = n_episodes_per_batch
        self.dummy_action_magnitude_interval = dummy_action_magnitude_interval
        self.use_ship_mass = use_ship_mass
        self.imaginator = imaginator  # type: ImaginatorConfiguration
        self.controller = controller  # type: Union[ControllerConfiguration, SetControllerAndFlatMemoryConfiguration, SetControllerAndSetMemoryConfiguration]
        self.manager = manager  # type: Union[ManagerConfiguration, PPOManagerConfiguration, BinaryManagerConfiguration, BinomialManagerConfiguration, CuratorConfiguration]
        self.imaginator_ignores_secondary = imaginator_ignores_secondary
        self.with_beacons = with_beacons
        self.beacon_probability = beacon_probability
        self.beacon_radial_distance_interval = beacon_radial_distance_interval
        self.gravity_cap = gravity_cap

    @property
    def routes_of_strategy(self):
        return routes_of_strategy[self.imagination_strategy]

    @property
    def history_embedding_length(self):
        if self.controller is not None:
            if hasattr(self.controller, 'memoryless'):
                if self.controller.memoryless:
                    return sum([
                        self.controller.aggregate_embedding_length if len(self.controller.aggregate_function_layer_sizes) > 0 else self.controller.object_embedding_length,
                        2 if self.controller.use_action else 0,
                        1 if self.controller.use_i_imagination else 0,
                    ])

        return self._history_embedding_length


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
