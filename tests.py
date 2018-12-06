from manager import Manager
from imaginator import Imaginator
import numpy as np


def test_forward_manager():
    test = Manager()

    ship_vector = np.zeros(5)
    planet_vectors = [np.zeros(5) for _ in range(5)]
    history_vector = np.zeros(100)

    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))
    print(test(ship_vector, planet_vectors, history_vector))


def test_forward_imaginator():
    test = Imaginator()

    ship_vector = np.zeros(5)
    planet_vectors = [np.zeros(5) for _ in range(5)]
    action_vector = np.zeros(2)

    print(test(ship_vector, planet_vectors, action_vector))
    print(test(ship_vector, planet_vectors))


test_forward_manager()
test_forward_imaginator()
