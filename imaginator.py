from utilities import *


class Imaginator(torch.nn.Module):
    def __init__(self,
                 object_vector_length,
                 action_vector_length,
                 only_imagine_agent_ship_velocity,
                 n_physics_steps_per_action,
                 relation_module_layer_sizes=(150, 150, 150, 150),
                 effect_embedding_length=100,
                 object_module_layer_sizes=(100,)):
        super().__init__()

        self.relation_module = make_mlp_with_relu(
            input_size=2 * object_vector_length,
            hidden_layer_sizes=relation_module_layer_sizes,
            output_size=effect_embedding_length,
            final_relu=True
        )

        state_imagination_vector_length = 2 if only_imagine_agent_ship_velocity else object_vector_length
        fuel_cost_prediction_vector_length = 1
        task_loss_prediction_vector_length = 1

        self.object_module = make_mlp_with_relu(
            input_size=object_vector_length + action_vector_length + effect_embedding_length,
            hidden_layer_sizes=object_module_layer_sizes,
            output_size=state_imagination_vector_length + fuel_cost_prediction_vector_length + task_loss_prediction_vector_length,
            final_relu=False
        )

        self.only_imagine_agent_ship_velocity = only_imagine_agent_ship_velocity
        self.n_physics_steps_per_action = n_physics_steps_per_action
        self.action_vector_length = action_vector_length

    def forward(self, agent_ship_vector, planet_vectors, action):
        imagined_trajectory = []

        for i_physics_step in range(self.n_physics_steps_per_action):
            last_imagined_state, fuel_cost_prediction, task_loss_prediction = self.physics_step(
                agent_ship_vector if i_physics_step == 0 else last_imagined_state,
                planet_vectors,
                action if i_physics_step == 0 else np.zeros(self.action_vector_length)
            )

            imagined_trajectory.append(last_imagined_state.detach().numpy())

            if i_physics_step == 0:
                imagined_fuel_cost = fuel_cost_prediction

            if i_physics_step == self.n_physics_steps_per_action - 1:
                imagined_task_loss = task_loss_prediction

        return last_imagined_state.detach(), imagined_fuel_cost.detach(), imagined_task_loss.detach(), np.array(imagined_trajectory)

    def physics_step(self, agent_ship_vector, planet_vectors, action):
        effect_embeddings = [self.relation_module(tensor_from(agent_ship_vector, planet_vector)) for planet_vector in planet_vectors]

        aggregate_effect_embedding = torch.sum(torch.stack(effect_embeddings), dim=0)
        predictions = self.object_module(tensor_from(agent_ship_vector, action, aggregate_effect_embedding))

        if self.only_imagine_agent_ship_velocity:
            state_imagination = tensor_from(agent_ship_vector[:3], predictions[:-2])
        else:
            state_imagination = predictions[:-2]

        return (
            state_imagination,  # State imagination
            predictions[-2:-1],  # Fuel cost prediction. Weird slice keeps tensor dimensionality of 1 instead of 0
            predictions[-1:]  # Task loss prediction. Weird slice keeps tensor dimensionality of 1 instead of 0
        )
