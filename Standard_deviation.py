import math

import numpy as np


def generate_random_vector(n, lower_bound=0, upper_bound=10000):
    """
    Generates a 1D array of random integers.
    """
    rng = np.random.default_rng()  # The generator instance
    return rng.integers(lower_bound, upper_bound, size=n)


# Example: Generate a set with 15 elements
n = 15
random_data = generate_random_vector(n)
print(f"Generated Vector: {random_data}")


def sd(random_data, average):
    addition = 0
    total = 0
    square = 0
    for element in random_data:
        addition = element-average
        square = math.pow(addition, 2)
        total += square

    stddev = (math.sqrt(total / len(random_data)) )
    return stddev


def average(liste):
    cont = 0
    addition = 0
    for element in liste:

        addition += element
        cont += 1

    average = addition / cont
    return average


ave = average(random_data)
print(f"Average is {ave}")

std = sd(random_data, ave)
print(f"Standard deviation is {std}")

print(np.std(random_data))
