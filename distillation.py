import torch
from utilities import *
import tensorboardX
import os
from spaceship_environment import Planet, Ship
from experiment import Experiment
from typing import List
from experiment_line import ExperimentLine


class Distiller(torch.nn.Module):
    def __init__(self, experiment: Experiment, evaluation_experiment: Experiment, name):
        super().__init__()

        self.exp = experiment
        self.eval = evaluation_experiment

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


class SimpleDistiller(torch.nn.Module):
    def __init__(self, experiments: List[Experiment], name):
        super().__init__()

        if isinstance(experiments, Experiment):
            experiments = [experiments]

        self.exps = experiments

        self.network = make_mlp_with_relu(
            input_size=self.exps[0].conf.controller.object_embedding_length + 1,
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

    def training_do(self, n_epochs, printit=False, superprintit=False):
        for exp in self.exps:
            exp.initialize_environment()
            exp.env.render_after_each_step = False
            exp.train_model = False
            exp.store_model = False
            exp.tensorboard_writer = None

        for i_epoch in range(n_epochs):
            for exp in self.exps:
                features, targets = self.get_training_samples(exp)

                outputs = self(features).squeeze()
                loss = torch.nn.functional.mse_loss(outputs, targets.detach())

                self.optimizer.zero_grad()
                loss.backward()

                if superprintit:
                    print(outputs)
                    print(targets.detach())

                if printit:
                    print("epoch {}: {} ({})".format(i_epoch, loss, exp.name))

                self.optimizer.step()

    def get_training_samples(self, exp: Experiment):
        exp.env.reset()

        objects = [exp.env.agent_ship] + exp.env.planets
        object_embeddings = [x.detach() if x is not None else x for x in exp.agent.controller_and_memory.memory.get_object_embeddings(objects)]
        norms = [object_embedding.norm() for object_embedding in object_embeddings]
        norms = torch.stack(norms)
        object_embeddings = torch.stack(object_embeddings)
        together = torch.cat([object_embeddings, norms.unsqueeze(dim=1)], dim=1)

        targets = torch.FloatTensor([1] + [0] * len(exp.env.planets))

        return together, targets

    def evaluate(self, evaluation_experiment: Experiment, n_episodes, printit=False, superprintit=False):
        evaluation_experiment.initialize_environment()
        evaluation_experiment.env.render_after_each_step = False
        evaluation_experiment.train_model = False
        evaluation_experiment.store_model = False
        evaluation_experiment.tensorboard_writer = None

        losses = []
        for i_epoch in range(n_episodes):
            evaluation_experiment.env.reset()

            objects = [evaluation_experiment.env.agent_ship] + evaluation_experiment.env.planets
            object_embeddings = [x.detach() if x is not None else x for x in evaluation_experiment.agent.controller_and_memory.memory.get_object_embeddings(objects)]
            norms = [object_embedding.norm() for object_embedding in object_embeddings]
            norms = torch.stack(norms)
            object_embeddings = torch.stack(object_embeddings)
            together = torch.cat([object_embeddings, norms.unsqueeze(dim=1)], dim=1)

            targets = torch.FloatTensor([1] + [0] * len(evaluation_experiment.env.planets))
            outputs = self(together).squeeze()
            if superprintit:
                print(outputs)
                print(targets)
            loss = torch.nn.functional.l1_loss(outputs, targets.detach())
            losses.append(loss)

        total_loss = torch.stack(losses).mean().item()

        if printit:
            print(total_loss)

        return total_loss


class SequenceDistiller(torch.nn.Module):
    def __init__(self, lines: List[ExperimentLine], lstm_hidden=100):
        super().__init__()

        if isinstance(lines, ExperimentLine):
            lines = [lines]

        self.lines = lines

        feature_length = self.lines[0].conf.controller.object_embedding_length + 1

        self.lstm = torch.nn.LSTM(input_size=feature_length, hidden_size=lstm_hidden, batch_first=True)
        self.output = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, 1),
            torch.nn.Sigmoid()
        )

        self.lstm_hidden = lstm_hidden
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        # hidden = (torch.randn(1, inputs.shape[0], 1, self.lstm_hidden), torch.randn(1, inputs.shape[0], self.lstm_hidden))
        predictions = self.output(self.lstm(inputs)[0][:, -1])
        return predictions

    def training_do(self, n_epochs, n_objects_per_epoch, stop=None, printit=False, superprintit=False):
        for i_epoch in range(n_epochs):
            seqs = []
            targs = []

            for line in self.lines:
                seq, targ, _ = line.get_batch(n_objects_per_epoch, stop=stop, balanced=True)
                seqs.append(seq)
                targs.append(targ)

            seqs = torch.cat(seqs, dim=0)
            targs = torch.cat(targs, dim=0)

            predictions = self(seqs)
            loss = torch.nn.functional.mse_loss(predictions, targs).mean()
            simple_loss = torch.nn.functional.l1_loss(predictions, targs).mean().item()

            self.optimizer.zero_grad()
            loss.backward()

            if superprintit:
                print(predictions)
                print(targs)

            if printit:
                print("epoch {}: {}".format(i_epoch, simple_loss))

            self.optimizer.step()

    def evaluate(self, n_objects, line: ExperimentLine, stop=None, balanced=False):
        seqs, targs, types = line.get_batch(n_objects, stop=stop, balanced=balanced)
        predictions = self(seqs)

        simple_loss = torch.nn.functional.l1_loss(predictions, targs, reduction='none')
        total_loss = simple_loss.mean().item()

        planet_loss = simple_loss[types == 0].mean().item()
        ship_loss = simple_loss[types == 1].mean().item()
        beacon_loss = simple_loss[types == 2].mean().item() if line.conf.with_beacons else None

        stri = "{:0.2f} | {:0.2f} [p: {:0.2f} | s: {:0.2f}{}]".format(
            total_loss,
            (planet_loss + ship_loss + beacon_loss) / 3 if line.conf.with_beacons else (planet_loss + ship_loss) / 2,
            planet_loss,
            ship_loss,
            " | b: {:0.2f}".format(beacon_loss) if line.conf.with_beacons else ''
        )

        return stri, total_loss, planet_loss, ship_loss, beacon_loss


def tryout():
    experiment = Experiment.load("v_2-epo_2-ponder_price_0.05-id_1", ('storage', 'lisa', 'bino2'))
    thing = Distiller(experiment, "noname")
    thing.training_do(1000)


def tryout_simple():
    exp1 = Experiment.load("v_6-memoryless_True-id_5", ('storage', 'lisa', 'varia_hleak'))
    exp2 = Experiment.load("v_8-memoryless_True-id_7", ('storage', 'lisa', 'varia_hleak'))
    exp3 = Experiment.load("v_7-memoryless_True-id_6", ('storage', 'lisa', 'varia_hleak'))
    thing = SimpleDistiller([exp1, exp2], "noname")
    thing.training_do(100)
    thing.evaluate(exp3, 200)
    thing.evaluate(exp1, 200)
    thing.evaluate(exp2, 200)


def compare_mix():
    n = 20
    exp1 = Experiment.load("v_6-memoryless_True-id_5", ('storage', 'lisa', 'varia_hleak'))
    exp2 = Experiment.load("v_8-memoryless_True-id_7", ('storage', 'lisa', 'varia_hleak'))
    exp3 = Experiment.load("v_7-memoryless_True-id_6", ('storage', 'lisa', 'varia_hleak'))
    mix = [SimpleDistiller([exp1, exp2], "noname") for _ in range(n)]
    no_mix_1 = [SimpleDistiller([exp1], "noname") for _ in range(n)]
    no_mix_2 = [SimpleDistiller([exp2], "noname") for _ in range(n)]

    for model in [item for sublist in [mix, no_mix_1, no_mix_2] for item in sublist]:
        model.training_do(100)

    mix_results = [model.evaluate(exp3, 200) for model in mix]
    no_mix_1_results = [model.evaluate(exp3, 200) for model in no_mix_1]
    no_mix_2_results = [model.evaluate(exp3, 200) for model in no_mix_2]

    for res in (mix_results, no_mix_1_results, no_mix_2_results):
        print(sum(res) / len(res))
        print(sorted(res))


def train_on_one():
    eval = ExperimentLine(name="many-10_v1", path=('storage', 'home', 'memless'))

    distiller = SequenceDistiller(eval)
    distiller.training_do(100, 20, printit=True, stop=10)
    distiller.evaluate(200, eval, stop=25)


def tryout_seq():
    # line1 = ExperimentLine(name="max_action_None-fuel_0-max_imag_4-5", path=('storage', 'home', 'memless'))
    # line2 = ExperimentLine(name="max_action_None-fuel_0-max_imag_4-4", path=('storage', 'home', 'memless'))
    # line3 = ExperimentLine(name="max_action_None-fuel_0-max_imag_4-3", path=('storage', 'home', 'memless'))
    # line4 = ExperimentLine(name="han_top_1_v2", path=('storage', 'home', 'memless'))

    lines = [ExperimentLine(name=name, path=('storage', 'lisa2', 'han1')) for name in [
        "v_5-scratch_True-n_han_1-id_4",
        "v_10-scratch_True-n_han_1-id_9",
        "v_1-scratch_True-n_han_2-id_20",
        "v_3-scratch_True-n_han_2-id_22"
        # "v_6-scratch_True-n_han_1-id_5",
        # "v_1-scratch_True-n_han_1-id_0",
        # "v_8-scratch_True-n_han_1-id_7"
    ]]

    eval = ExperimentLine(name="many-10_v1", path=('storage', 'home', 'memless'))

    distiller = SequenceDistiller(lines[:2])
    distiller.training_do(100, 20, printit=True, stop=15)

    for i in range(5, 30):
        stri = distiller.evaluate(200, lines[2], stop=i)[0]
        print("{}@{}: {}".format(15, i, stri))


def check_out(line: ExperimentLine, stop=None):
    torch.set_printoptions(profile="full", linewidth=1000, precision=5)
    objects, targets, types = line.get_batch(3 if line.env.with_beacons else 2, True, stop=stop)

    print("ship ({}) for {}".format(types[2 if line.env.with_beacons else 1], line.name))
    print("")
    print(objects[2 if line.env.with_beacons else 1])
    print("")

    print("planet ({}) for {}".format(types[1 if line.env.with_beacons else 0], line.name))
    print("")
    print(objects[1 if line.env.with_beacons else 0])
    print("")

    if line.env.with_beacons:
        print("beacon ({}) for {}".format(types[0], line.name))
        print("")
        print(objects[0])
        print("")


tryout_seq()
