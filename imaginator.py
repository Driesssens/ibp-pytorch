from utilities import *
from spaceship_environment import SpaceshipEnvironment, Ship, Planet
from typing import List
from copy import deepcopy


class Imaginator(torch.nn.Module):

    def __init__(self,
                 parent,
                 relation_module_layer_sizes=(150, 150, 150, 150),
                 effect_embedding_length=100,
                 object_module_layer_sizes=(100,),
                 velocity_normalization_factor=7.5,
                 learning_rate=0.003,
                 max_gradient_norm=10,
                 ):
        super().__init__()

        self.relation_module = make_mlp_with_relu(
            input_size=1 + 1 + 2,  # ship mass + planet mass + difference vector between ship and planet xy position
            hidden_layer_sizes=relation_module_layer_sizes,
            output_size=effect_embedding_length,
            final_relu=False
        )

        self.object_module = make_mlp_with_relu(
            input_size=1 + 2 + 2 + effect_embedding_length,  # ship mass + ship xy velocity + action + effect embedding
            hidden_layer_sizes=object_module_layer_sizes,
            output_size=2,  # imagined velocity
            final_relu=False
        )

        self.parent = parent
        self.velocity_normalization_factor = velocity_normalization_factor

        self.batch_loss_terms = []
        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.max_gradient_norm = max_gradient_norm

    def imagine(self, ship: Ship, planets: List[Planet], action):
        imagined_ship_trajectory = [deepcopy(ship)]

        for i_physics_step in range(self.parent.environment.n_steps_per_action):
            current_state = imagined_ship_trajectory[-1]

            imagined_state = deepcopy(current_state)
            imagined_state.xy_position = current_state.xy_position + self.parent.environment.euler_method_step_size * current_state.xy_velocity

            imagined_velocity = self(
                current_state,
                planets,
                action if i_physics_step == 0 else np.zeros(2)
            ).detach().numpy()

            imagined_state.xy_velocity = imagined_velocity
            imagined_ship_trajectory.append(imagined_state)

        return imagined_ship_trajectory

    def forward(self, ship: Ship, planets: List[Planet], action):
        effect_embeddings = [self.relation_module(tensor_from(ship.mass, planet.mass, ship.xy_position - planet.xy_position)) for planet in planets]
        aggregate_effect_embedding = torch.mean(torch.stack(effect_embeddings), dim=0)
        imagined_velocity = self.object_module(tensor_from(ship.mass, ship.xy_velocity / self.velocity_normalization_factor, action, aggregate_effect_embedding))
        return imagined_velocity

    def compute_loss(self, ship_trajectory: List[Ship], planets: List[Planet], action):
        for i_physics_step in range(len(ship_trajectory) - 1):
            previous_state = ship_trajectory[i_physics_step]
            imagined_velocity = self(previous_state, planets, action if i_physics_step == 0 else np.zeros(2))
            target_velocity = ship_trajectory[i_physics_step + 1].xy_velocity

            loss = torch.nn.functional.mse_loss(imagined_velocity.unsqueeze(0), tensor_from(target_velocity).unsqueeze(0), reduction='sum')
            self.batch_loss_terms.append(loss)

    def finish_batch(self):
        stacked = torch.stack(self.batch_loss_terms)
        mean_loss = torch.mean(stacked)
        self.parent.log("imaginator_mean_loss", mean_loss.item())

        self.optimizer.zero_grad()
        mean_loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_gradient_norm)

        self.parent.log("minibatch_norm", norm)
        self.optimizer.step()

        self.batch_loss_terms = []

    def store(self):
        torch.save(self.state_dict(), self.parent.experiment_folder + '\imaginator_state_dict')

    def load(self, experiment_name):
        self.load_state_dict(torch.load("storage\{}\imaginator_state_dict".format(experiment_name)))
