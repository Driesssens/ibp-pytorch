import json
import time
from copy import deepcopy

from controller_and_memory import ControllerAndMemory
from imaginator import Imaginator
from full_imaginator import FullImaginator
from manager import Manager, PPOManager
from spaceship_environment import polar2cartesian
from configuration import route_as_vector, Routes
from utilities import *
from binary_manager import BinaryManager, BinomialManager
from curator import Curator

if False:
    from experiment import Experiment


class ImaginationBasedPlanner:
    def __init__(self, experiment: 'Experiment'):
        self.exp = experiment  # type: 'Experiment'

        self.i_episode = 0
        self.history_embedding = torch.zeros(self.exp.conf.history_embedding_length)

        self.imaginator = None  # type: Imaginator
        self.controller_and_memory = None  # type: ControllerAndMemory
        self.manager = None  # type: Manager

        self.batch_start_time = time.perf_counter()
        self.batch_action_magnitude = Accumulator()

        self.batch_n_planets_in_each_imagination = Accumulator()
        self.batch_f_planets_in_each_imagination = Accumulator()

        self.batch_ship_p = Accumulator()

    def act(self):

        if self.exp.conf.controller.manual_ship_selection:
            filter_indices = [False] * self.exp.env.n_obj()
            filter_indices[0] = True
        else:
            filter_indices = [True] * self.exp.env.n_obj()

        if self.controller_and_memory is not None:
            object_embeddings = None

            if self.uses_filters() and self.exp.conf.controller.filter_before_imagining:
                filter_indices, object_embeddings = self.filter(self.exp.env.objs())

            if not self.exp.conf.controller.blind_first_action:
                self.history_embedding = self.controller_and_memory.memory(
                    action=np.zeros(2),
                    objects=self.exp.env.objs(),
                    i_imagination=0,
                    filter_indices=filter_indices,
                    object_embeddings=object_embeddings
                )

        for i_imagination in range(self.exp.conf.max_imaginations_per_action):
            proposed_action = self.controller_and_memory.controller(self.exp.env.agent_ship) if self.controller_and_memory is not None else self.dummy_action()

            imagined_objects, _ = self.imaginator.imagine(
                self.exp.env.agent_ship,
                proposed_action,
                filter_indices,
                differentiable_trajectory=True,
                record=i_imagination == self.exp.conf.max_imaginations_per_action - 1
            )

            if self.controller_and_memory is not None:
                object_embeddings = None

                if i_imagination == 0 and self.uses_filters() and not self.exp.conf.controller.filter_before_imagining:
                    filter_indices, object_embeddings = self.filter(imagined_objects)

                self.history_embedding = self.controller_and_memory.memory(
                    action=proposed_action,
                    objects=imagined_objects,
                    i_imagination=i_imagination + (1 if self.exp.conf.controller.blind_first_action else 0),
                    filter_indices=filter_indices,
                    object_embeddings=object_embeddings
                )

        if self.controller_and_memory is not None:
            selected_action = self.controller_and_memory.controller(self.exp.env.agent_ship)
            detached_action = selected_action.detach().numpy()
        else:
            selected_action = self.dummy_action()
            detached_action = selected_action

        if hasattr(self, 'measure_analyze_actions'):
            self.actual_actions.append(np.linalg.norm(detached_action))

        old_ship_state = deepcopy(self.exp.env.agent_ship)
        actual_trajectory, actual_task_cost = self.perform_action(detached_action)
        new_ship_state = actual_trajectory[-1]

        if self.has_ppo_manager():
            internal_cost = sum(self.manager.episode_costs)
            self.manager.batch_ponder_cost.add(internal_cost)

            external_cost = actual_task_cost
            self.manager.episode_costs[-1] += external_cost

            task_cost = internal_cost + external_cost
            self.manager.batch_task_cost.add(task_cost)

            if hasattr(self, 'measure_performance'):
                self.manager_mean_task_cost_measurements.append(task_cost)

        if isinstance(self.imaginator, FullImaginator):
            self.imaginator.accumulate_loss(actual_trajectory, detached_action, filter_indices)

            # subjects = [old_ship_state] + self.exp.env.planets + self.exp.env.beacons
            # influencerss = [[x for x in subjects if x is not subject] for subject in subjects]
            #
            # embeddingss = []
            # normss = []
            #
            # for subject, influencers in zip(subjects, influencerss):
            #
            # embeddingss = [[x.detach().norm() for x in self.imaginator.embed(subject, influencers)] for subject, influencers in zip(subjects, influencerss)]
            #
            # if self.exp.conf.imaginator.simplehan_threshold is not None:
            #     zeros_and_ones = torch.FloatTensor(norms) >= self.exp.conf.imaginator.simplehan_threshold * torch.FloatTensor(norms).mean()
            #     filter_indices = [x.item() == 1 for x in zeros_and_ones]
            #
            # for subject in self.exp.env.planets:
            #     influencers = [x for x in self.exp.env.planets if x is not subject] + [old_ship_state]
            #     self.imaginator.accumulate_loss([subject] * len(actual_trajectory), influencers, detached_action)
            #
            #     if self.exp.conf.max_imaginations_per_action == 0:
            #         estimated_final_position = self.imaginator.imagine(subject, influencers, detached_action)[0][-1].xy_position
            #         final_prediction_error = np.square(estimated_final_position - subject.xy_position).mean()
            #         self.imaginator.batch_evaluation.add(final_prediction_error)
            #         self.imaginator.batch_static_evaluation.add(final_prediction_error)
        else:
            if filter_indices[0]:
                self.imaginator.accumulate_loss(actual_trajectory, self.exp.env.planets, detached_action)

        critic_evaluation = self.imaginator.evaluate(
            old_ship_state,
            selected_action if self.controller_and_memory is not None else detached_action,
            new_ship_state,
            filter_indices
        )

        if self.controller_and_memory is not None:
            self.controller_and_memory.accumulate_loss(critic_evaluation)

        self.batch_action_magnitude.add(np.linalg.norm(detached_action))
        self.batch_n_planets_in_each_imagination.add(filter_indices.count(True))
        self.batch_f_planets_in_each_imagination.add(filter_indices.count(True) / self.exp.env.n_obj())

    def filter(self, objects):
        object_embeddings = self.controller_and_memory.memory.get_object_embeddings(objects)
        detached_object_embeddings = [embedding.detach() for embedding in object_embeddings]

        if self.has_ppo_manager():
            if isinstance(self.manager, BinaryManager) or isinstance(self.manager, BinomialManager):
                filter_indices, _ = self.manager.act(detached_object_embeddings, objects)
            else:
                threshold = self.manager.act(object_embeddings, objects)
                self.manager.batch_threshold.add(threshold)
                filter_indices = [threshold < embedding.norm().item() for embedding in detached_object_embeddings]

            f_kept_objects = filter_indices.count(True) / len(filter_indices)
            self.manager.episode_costs.append(self.exp.conf.manager.ponder_price * f_kept_objects * self.exp.conf.max_imaginations_per_action)

        if self.is_self_filtering():
            norms = [embedding.norm() for embedding in detached_object_embeddings]

            if self.exp.conf.controller.han_n_top_objects is not None:
                filter_indices = [False] * len(objects)
                top_indices = sorted(range(len(norms)), key=lambda i: norms[i])[-self.exp.conf.controller.han_n_top_objects:]
                for ind in top_indices:
                    filter_indices[ind] = True

            elif self.exp.conf.controller.simplehan_threshold is not None:
                norms_tensor = torch.FloatTensor(norms)
                zeros_and_ones = norms_tensor >= self.exp.conf.controller.simplehan_threshold * norms_tensor.mean()
                filter_indices = [x.item() == 1 for x in zeros_and_ones]

            if len(self.exp.env.beacons) == 0:
                if filter_indices[0]:
                    self.batch_ship_p.add(1)
                else:
                    self.batch_ship_p.add(0)
            else:
                if filter_indices[0] and filter_indices[-1]:
                    self.batch_ship_p.add(1)
                elif filter_indices[0] or filter_indices[-1]:
                    self.batch_ship_p.add(0.5)
                else:
                    self.batch_ship_p.add(0)

        return filter_indices, object_embeddings

    def perform_action(self, action):
        resultant_trajectory = [deepcopy(self.exp.env.agent_ship)]

        for i_physics_step in range(self.exp.env.n_steps_per_action):
            self.exp.env.step(action if i_physics_step == 0 else np.zeros(2))
            resultant_trajectory.append(deepcopy(self.exp.env.agent_ship))

        actual_final_position = resultant_trajectory[-1].xy_position

        if len(self.exp.env.beacons) == 0:
            task_cost = np.square(actual_final_position).mean()
        else:
            task_cost = np.square(actual_final_position - self.exp.env.beacons[0].xy_position).mean()

        return resultant_trajectory, task_cost

    def finish_episode(self):
        if self.manager is not None:
            self.manager.finish_episode()

        if (self.i_episode + 1) % self.exp.conf.n_episodes_per_batch == 0:
            self.finish_batch()

        self.history_embedding = torch.zeros(self.history_embedding.shape)

        if self.controller_and_memory is not None:
            self.controller_and_memory.memory.reset_state()

        self.i_episode += 1

    def uses_filters(self):
        return self.has_ppo_manager() or self.is_self_filtering()

    def is_self_filtering(self):
        return self.controller_and_memory is not None and (
                (self.exp.conf.controller.han_n_top_objects is not None) or
                (self.exp.conf.controller.adahan_threshold is not None) or
                (self.exp.conf.controller.simplehan_threshold is not None)
        )

    def has_normal_manager(self):
        return self.manager is not None and isinstance(self.manager, Manager)

    def has_ppo_manager(self):
        has_it = self.manager is not None and (isinstance(self.manager, PPOManager) or isinstance(self.manager, BinaryManager) or isinstance(self.manager, BinomialManager))
        return has_it and self.i_episode >= self.exp.conf.manager.n_steps_delay

    def finish_batch(self):
        self.imaginator.finish_batch()

        if self.controller_and_memory is not None:
            self.controller_and_memory.finish_batch()

        if self.manager is not None:
            self.manager.finish_batch()

        self.exp.log("mean_real_action_magnitude", self.batch_action_magnitude.average())
        self.batch_action_magnitude = Accumulator()

        if self.is_self_filtering():
            self.exp.log("manager_mean_ship_p", self.batch_ship_p.average())
            self.batch_ship_p = Accumulator()

        self.exp.log("mean_n_planets_in_each_imagination", self.batch_n_planets_in_each_imagination.average())
        self.exp.log("mean_f_planets_in_each_imagination", self.batch_f_planets_in_each_imagination.average())
        self.batch_n_planets_in_each_imagination = Accumulator()
        self.batch_f_planets_in_each_imagination = Accumulator()

        self.exp.log("get_num_threads", torch.get_num_threads())

        if self.exp.store_model:
            self.store_model()

        now = time.perf_counter()
        minutes_passed = (now - self.batch_start_time) / 60
        episodes_per_minute = self.exp.conf.n_episodes_per_batch / minutes_passed
        self.exp.log("episodes_per_minute", episodes_per_minute)
        self.batch_start_time = now

    @property
    def i_episode_of_batch(self):
        return self.i_episode % self.exp.conf.n_episodes_per_batch

    def store_model(self, training_status_update=True):
        if training_status_update:
            with open(self.exp.file_path('training_status.json'), 'w') as file:
                json.dump(self.training_status, file, indent=2)

        self.imaginator.store_model()

        if self.controller_and_memory is not None:
            self.controller_and_memory.store_model()

        if self.manager is not None:
            self.manager.store_model()

    def load_model(self):
        try:
            with open(self.exp.file_path('training_status.json')) as file:
                training_status = json.load(file)
                self.i_episode = training_status['i_episode'] + 1
        except FileNotFoundError:
            self.i_episode = int(self.exp.name)

    @property
    def training_status(self):
        return {'i_episode': self.i_episode}

    def dummy_action(self):
        angle = np.pi * np.random.uniform(0, 2)
        radius = np.random.uniform(*self.exp.conf.dummy_action_magnitude_interval)
        action = np.array(polar2cartesian(angle, radius))
        return action
