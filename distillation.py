import torch
from utilities import *
import tensorboardX
import os
from spaceship_environment import Planet, Ship
from experiment import Experiment

if False:
    from experiment import Experiment


class Distiller(torch.nn.Module):
    def __init__(self, experiment: 'Experiment', name):
        super().__init__()

        self.exp = experiment

        self.network = make_mlp_with_relu(
            input_size=self.exp.conf.controller.object_embedding_length + 1,
            hidden_layer_sizes=[64, 64, 64],
            output_size=1,
            final_relu=False
        )

        # os.makedirs('runs/distillation/' + name)
        # self.writer = tensorboardX.SummaryWriter('runs/distillation/' + name)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def forward(self, input):
        logits = self.network(input)
        probs = torch.sigmoid(logits)
        return probs

    def training_do(self, n_epochs):
        self.exp.initialize_environment()
        self.exp.env.render_after_each_step = False
        self.exp.train_model = False
        self.exp.store_model = False
        self.exp.tensorboard_writer = None

        for i_epoch in range(n_epochs):
            self.exp.env.reset()

            objects = [self.exp.env.agent_ship] + self.exp.env.planets
            object_embeddings = [x.detach() if x is not None else x for x in self.exp.agent.controller_and_memory.memory.get_object_embeddings(objects)]

            object_features = torch.stack([
                tensor_from(
                    object_embedding if self.exp.conf.manager.feature_controller_embedding else None,
                    object_embedding.norm().item() if self.exp.conf.manager.feature_norm else None,
                    tensor_from(
                        0 if isinstance(objekt, Planet) else 1,
                        1 if isinstance(objekt, Planet) else 0,
                        objekt.mass,
                        objekt.encode_state(False)
                    ).detach() if self.exp.conf.manager.feature_state else None
                )
                for object_embedding, objekt in zip(object_embeddings, objects)
            ])

            action_distribution, _, _ = self.exp.agent.manager(object_features, self.exp.agent.history_embedding.detach())

            norms = [object_embedding.norm() for object_embedding in object_embeddings]
            norms = torch.stack(norms)
            # print(norms)
            object_embeddings = torch.stack(object_embeddings)
            # print(object_embeddings)
            together = torch.cat([object_embeddings, norms.unsqueeze(dim=1)], dim=1)

            targets = action_distribution.probs
            # print(together)
            outputs = self(together).squeeze()
            loss = torch.nn.functional.mse_loss(outputs, targets.detach())
            self.optimizer.zero_grad()
            loss.backward()
            print(outputs)
            print(targets.detach())
            print("epoch {}: {}".format(i_epoch, loss))
            self.optimizer.step()


def tryout():
    experiment = Experiment.load("v_2-epo_2-ponder_price_0.05-id_1", ('storage', 'lisa', 'bino2'))
    thing = Distiller(experiment, "noname")
    thing.training_do(1000)


tryout()