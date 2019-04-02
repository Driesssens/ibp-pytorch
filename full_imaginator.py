from utilities import *
from spaceship_environment import Ship, Planet, polar2cartesian, cartesian2polar, SpaceObject, Beacon
from typing import List
from copy import deepcopy

if False:
    from experiment import Experiment


class FullImaginator(torch.nn.Module):

    def __init__(self, experiment: 'Experiment'):
        super().__init__()

        self.exp = experiment

        type_length = 3 if self.exp.conf.with_beacons else 2

        self.relation_module = make_mlp_with_relu(
            input_size=2 * type_length + 1 + 1 + 2,  # type A + type B + mass A + mass B + difference vector between xy positions
            hidden_layer_sizes=self.exp.conf.imaginator.relation_module_layer_sizes,
            output_size=self.exp.conf.imaginator.effect_embedding_length,
            final_relu=False
        )

        self.object_module = make_mlp_with_relu(
            input_size=type_length + 1 + 2 + 2 + self.exp.conf.imaginator.effect_embedding_length,  # type + mass + xy velocity + action + effect embedding
            hidden_layer_sizes=self.exp.conf.imaginator.object_module_layer_sizes,
            output_size=2,  # imagined velocity
            final_relu=False
        )

        self.batch_loss = Accumulator()
        self.batch_evaluation = Accumulator()
        self.batch_static_evaluation = Accumulator()
        self.batch_dynamic_evaluation = Accumulator()
        self.batch_task_cost = Accumulator()

        self.optimizer = torch.optim.Adam(self.parameters(), self.exp.conf.imaginator.learning_rate)

    def imagine_single(self, subject: SpaceObject, influencers: List[SpaceObject], action, estimated=None):
        imagined_ship_trajectory = [deepcopy(subject)]
        imagined_ship_trajectory[-1].wrap_in_tensors()

        for i_physics_step in range(self.exp.env.n_steps_per_action):
            current_state = imagined_ship_trajectory[-1]

            imagined_state = deepcopy(current_state)
            imagined_state.xy_position = current_state.xy_position + self.exp.env.euler_method_step_size * current_state.xy_velocity

            imagined_velocity = self(
                current_state,
                influencers,
                action if i_physics_step == 0 else np.zeros(2)
            )

            imagined_state.xy_velocity = imagined_velocity
            imagined_ship_trajectory.append(imagined_state)

        if estimated is not None:
            if estimated:
                self.exp.env.add_estimated_ship_trajectory(imagined_ship_trajectory)
            else:
                self.exp.env.add_imagined_ship_trajectory(imagined_ship_trajectory)

        return imagined_ship_trajectory[-1]

    def imagine(self, ship: Ship, action, filter_indices, differentiable_trajectory=False, estimated=False, record=False):
        actual_objects = [ship] + self.exp.env.planets + self.exp.env.beacons
        imagined_objects = []

        for actual_object, filter_value in zip(actual_objects, filter_indices):
            if not filter_value:
                imagined_objects.append(None)
            else:
                predicted_result = self.imagine_single(actual_object, [obj for obj in actual_objects if obj is not actual_object], action, estimated=estimated if actual_object is ship else None)

                if actual_object is not ship and record:
                    estimated_final_position = predicted_result.xy_position
                    final_prediction_error = np.square(estimated_final_position.detach().numpy() - actual_object.xy_position).mean()
                    self.batch_evaluation.add(final_prediction_error)
                    self.batch_static_evaluation.add(final_prediction_error)

                imagined_objects.append(predicted_result)

        if filter_indices[0] and (filter_indices[-1] or len(self.exp.env.beacons) == 0):
            target_position = torch.zeros(2) if len(self.exp.env.beacons) == 0 else imagined_objects[-1].xy_position
            imagined_task_cost = torch.nn.functional.mse_loss(imagined_objects[0].xy_position.unsqueeze(0), target_position.unsqueeze(0))
        else:
            imagined_task_cost = None

        return imagined_objects, imagined_task_cost

    def embed(self, subject: SpaceObject, influencers: List[SpaceObject]):
        effect_embeddings = [
            self.relation_module(tensor_from(
                self.type_tensor(subject),
                subject.mass,
                self.type_tensor(influencer),
                influencer.mass,
                tensor_from(subject.xy_position) - tensor_from(influencer.xy_position)
            )) for influencer in influencers
        ]

        return effect_embeddings

    def type_tensor(self, of: SpaceObject):
        return tensor_from(
            1 if isinstance(of, Ship) else 0,
            1 if isinstance(of, Planet) else 0,
            (1 if isinstance(of, Beacon) else 0) if self.exp.conf.with_beacons else None,
        )

    def forward(self, subject: SpaceObject, influencers: List[SpaceObject], action):
        if self.exp.conf.imaginator_ignores_secondary:
            influencers = [planet for planet in influencers if not planet.is_secondary]

        effect_embeddings = self.embed(subject, influencers)

        if len(effect_embeddings) == 0:
            aggregate_effect_embedding = torch.zeros(self.exp.conf.imaginator.effect_embedding_length)
        else:
            aggregate_effect_embedding = torch.mean(torch.stack(effect_embeddings), dim=0)

        imagined_velocity = self.object_module(tensor_from(
            self.type_tensor(subject),
            subject.mass,
            subject.xy_velocity / self.exp.conf.imaginator.velocity_normalization_factor,
            action / self.exp.conf.imaginator.action_normalization_factor,
            aggregate_effect_embedding
        ))

        return imagined_velocity

    def accumulate_loss(self, subject_trajectory: List[SpaceObject], action, filter_indices):
        actual_objects = [subject_trajectory[0]] + self.exp.env.planets + self.exp.env.beacons

        for actual_object, filter_value in zip(actual_objects, filter_indices):
            if filter_value:
                if actual_object is actual_objects[0]:
                    traject = subject_trajectory
                else:
                    traject = [actual_object] * len(subject_trajectory)

                self.accumulate_loss_single(traject, [obj for obj in actual_objects if obj is not actual_object], action)

    def accumulate_loss_single(self, subject_trajectory: List[SpaceObject], influencers: List[SpaceObject], action):
        for i_physics_step in range(len(subject_trajectory) - 1):
            previous_state = subject_trajectory[i_physics_step]
            imagined_velocity = self(previous_state, influencers, action if i_physics_step == 0 else np.zeros(2))
            target_velocity = subject_trajectory[i_physics_step + 1].xy_velocity

            loss = torch.nn.functional.mse_loss(imagined_velocity.unsqueeze(0), tensor_from(target_velocity).unsqueeze(0), reduction='sum')

            self.batch_loss.add(loss)

    def evaluate(self, old_ship_state: Ship, action, actual_new_ship_state: Ship, filter_indices):
        #     if self.exp.conf.max_imaginations_per_action == 0:
        #         estimated_final_position = self.imaginator.imagine(subject, influencers, detached_action)[0][-1].xy_position
        #         final_prediction_error = np.square(estimated_final_position - subject.xy_position).mean()
        #         self.imaginator.batch_evaluation.add(final_prediction_error)
        #         self.imaginator.batch_static_evaluation.add(final_prediction_error)

        if self.exp.conf.max_imaginations_per_action == 0:
            filters = [True] * len(filter_indices)
        else:
            filters = [True] + [False] * len(self.exp.env.planets) + [True] * len(self.exp.env.beacons)

        imagined_objects, critic_evaluation = self.imagine(old_ship_state, action, filters, estimated=True, record=True)
        imagined_final_position = imagined_objects[0].xy_position
        actual_final_position = actual_new_ship_state.xy_position

        evaluation = np.square(imagined_final_position.detach().numpy() - actual_final_position).mean()
        self.batch_evaluation.add(evaluation)
        self.batch_dynamic_evaluation.add(evaluation)

        if len(self.exp.env.beacons) == 0:
            task_cost = np.square(actual_final_position).mean()
        else:
            task_cost = np.square(actual_final_position - self.exp.env.beacons[0].xy_position).mean()

        self.batch_task_cost.add(task_cost)

        if hasattr(self.exp.agent, 'measure_performance'):
            self.exp.agent.imaginator_mean_final_position_error_measurements.append(evaluation)
            self.exp.agent.controller_and_memory_mean_task_cost_measurements.append(task_cost)

        return critic_evaluation

    def finish_batch(self):
        mean_loss = self.batch_loss.average()
        self.exp.log("imaginator_mean_loss", mean_loss.item())

        if self.exp.train_model:
            self.optimizer.zero_grad()

            if self.exp.conf.imaginator.batch_loss_sum:
                self.batch_loss.cumulative_value.backward()
            else:
                mean_loss.backward()

            self.exp.log("imaginator_batch_norm", gradient_norm(self.parameters()))

            if self.exp.conf.imaginator.max_gradient_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.exp.conf.imaginator.max_gradient_norm)
                self.exp.log("imaginator_batch_clipped_norm", gradient_norm(self.parameters()))

            self.optimizer.step()

        self.batch_loss = Accumulator()

        self.exp.log("imaginator_mean_final_position_error", self.batch_evaluation.average())
        self.batch_evaluation = Accumulator()

        self.exp.log("imaginator_mean_dynamic_final_position_error", self.batch_dynamic_evaluation.average())
        self.batch_dynamic_evaluation = Accumulator()

        self.exp.log("imaginator_mean_static_final_position_error", self.batch_static_evaluation.average())
        self.batch_static_evaluation = Accumulator()

        self.exp.log("controller_and_memory_mean_task_cost", self.batch_task_cost.average())
        self.batch_task_cost = Accumulator()

    def store_model(self):
        torch.save(self.state_dict(), self.exp.file_path('imaginator_state_dict'))
        torch.save(self.optimizer.state_dict(), self.exp.file_path('imaginator_optimizer_state_dict'))

    def load_model(self):
        self.load_state_dict(torch.load(self.exp.file_path("imaginator_state_dict")))
        self.optimizer.load_state_dict(torch.load(self.exp.file_path("imaginator_optimizer_state_dict")))
