from manager import Manager
from imaginator import Imaginator
from memory import Memory
import numpy as np


def test_forward_manager():
    test = Manager()

    ship_vector = np.zeros(5)
    planet_vectors = [np.zeros(5) for _ in range(5)]
    history_vector = np.zeros(100)

    for _ in range(12):
        print(test(ship_vector, planet_vectors, history_vector))


def test_forward_imaginator():
    test = Imaginator()

    ship_vector = np.zeros(5)
    planet_vectors = [np.zeros(5) for _ in range(5)]
    action_vector = np.zeros(2)

    print(test(ship_vector, planet_vectors, action_vector))
    print(test(ship_vector, planet_vectors))


def test_forward_memory():
    test = Memory()

    ship_vector = np.zeros(5)
    planet_vectors = [np.zeros(5) for _ in range(5)]
    state_vector = np.concatenate([ship_vector] + planet_vectors)
    action_vector = np.zeros(2)

    for i_episode in range(2):
        test.reset_state()

        for i_action in range(3):
            for i_imagination in range(2):
                history_embedding = test(np.array([0, 1, 0]), state_vector, state_vector, action_vector, state_vector, 0.8, i_action, i_imagination)
                print("Ep {}, j={}, k={}: {}".format(i_episode, i_action, i_imagination, history_embedding))


test_forward_manager()
test_forward_imaginator()
test_forward_memory()
