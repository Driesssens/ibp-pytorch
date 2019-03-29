import json
import os
import itertools
import tensorboardX
import torch
import numpy as np

from configuration import *
from imagination_based_planner import ImaginationBasedPlanner
from spaceship_environment import SpaceshipEnvironment, Planet, Ship
from experiment import Experiment


class ExperimentLine:
    def __init__(self, name, path=('storage', 'home', 'misc',)):
        self.name = name
        self.path = path

        folders = sorted([int(f.name) for f in os.scandir(self.directory_path()) if f.is_dir()])[:40]

        self.exps = [Experiment.load(self.name, self.path, initialized_and_silenced=True, specific_instance=folder) for folder in folders]
        self.env = self.exps[0].env
        self.conf = self.exps[0].conf

    def directory_path(self):
        return os.path.join(*self.path, self.name)

    def object_to_seq(self, objekt, start=0, stop=None, skip=1, as_tensor=True):
        embeddings = [exp.agent.controller_and_memory.memory.get_object_embeddings([objekt])[0].detach() for exp in (self.exps[start::skip] if stop is None else self.exps[start:stop:skip])]
        with_norms = [torch.cat([embedding, embedding.norm().unsqueeze(dim=0)]) for embedding in embeddings]

        if as_tensor:
            with_norms = torch.stack(with_norms)

        return with_norms

    def file_path(self, file_name):
        return os.path.join(self.directory_path(), file_name)

    def get_batch(self, n_objects, balanced=False, start=0, stop=None, skip=1, as_tensor=True):
        seqs = []
        targets = []
        types = []

        objects = []

        for i in range(n_objects):
            if len(objects) == 0:
                if balanced:
                    self.env.beacon_probability = 1

                self.env.reset()

                if balanced:
                    objects = [self.env.agent_ship] + self.env.planets[:1] + self.env.beacons
                else:
                    objects = [self.env.agent_ship] + self.env.planets + self.env.beacons

            objekt = objects.pop()

            seqs.append(self.object_to_seq(objekt, start=start, stop=stop, skip=skip))
            targets.append(torch.FloatTensor([0]) if isinstance(objekt, Planet) else torch.FloatTensor([1]))
            types.append(torch.Tensor([0]) if isinstance(objekt, Planet) else (torch.Tensor([1]) if isinstance(objekt, Ship) else torch.Tensor([2])))

        if as_tensor:
            seqs = torch.stack(seqs)
            targets = torch.stack(targets)
            types = torch.stack(types)

        return seqs, targets, types


def bugtest():
    line = ExperimentLine(name="max_action_None-fuel_0-max_imag_4-5", path=('storage', 'home', 'memless'))
    ship = line.env.agent_ship
    planet = line.env.planets[0]

    print("episode: {}".format(line.exps[0].agent.i_episode))

    batch = line.get_batch(19, balanced=True)
    print(batch[1])
    print(batch[0].shape, batch[1].shape)

    print(line.object_to_seq(ship).shape)
    print(line.object_to_seq(planet).shape)


def lstm_test(n=100):
    line = ExperimentLine(name="max_action_None-fuel_0-max_imag_4-5", path=('storage', 'home', 'memless'))

    lstm = torch.nn.LSTM(51, 100, batch_first=True)
    linear = torch.nn.Linear(100, 1)

    for i in range(n):
        line.env.reset()

        objects = [line.env.agent_ship] + line.env.planets
        seqs = [line.object_to_seq(objekt) for objekt in objects]
        targets = [torch.FloatTensor([1]), [torch.FloatTensor(0) for __ in range(len(line.env.planets))]]

        for (seq, target) in zip(seqs, targets):
            hidden = (torch.randn(1, 1, 100), torch.randn(1, 1, 100))
            for thing in seq:
                out, hidden = lstm(thing.view(1, 1, -1), hidden)
                estimate = torch.sigmoid(linear(hidden[0]))
                print(estimate)
                loss = torch.nn.functional.mse_loss(estimate.squeeze().squeeze(), target)
                print(loss)

# bugtest()
