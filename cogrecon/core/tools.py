import numpy as np
import itertools
import random

# TODO: Documentation needs an audit/overhaul


# This function is for testing. It generates a set of "correct" and "incorrect" points such that the correct points are
# randomly placed between (0,0) and (1,1) in R2. Then it generates "incorrect" points which are offset randomly up
# to 10% in the positive direction, shuffled, and where one point is completely random.
def generate_random_test_points(number_of_points=5, dimension=2, shuffle_points=True, noise=(1.0 / 20.0),
                                num_rerandomed_points=1):
    correct_points = [[random.random() for _ in range(dimension)] for _ in range(0, number_of_points)]
    offsets = [[(random.random()) * noise for _ in range(dimension)] for _ in range(0, number_of_points)]
    input_points = np.array(correct_points) + np.array(offsets)

    if shuffle_points:
        perms = list(itertools.permutations(input_points))
        input_points = perms[random.randint(0, len(perms) - 1)]

    if num_rerandomed_points > len(input_points):  # Bound the number of randomized points to the number of points
        num_rerandomed_points = len(input_points)
    # Generate a random sampling of point indicies
    indicies = random.sample(range(0, len(input_points)), num_rerandomed_points)
    # Loop through each index and re-randomize each element in the point
    for index in indicies:
        for idx in range(len(input_points[index])):
            input_points[index][idx] = random.random()

    return np.array(correct_points).tolist(), np.array(input_points).tolist()


# This function performs simple vector linear interpolation on two equal length number lists
def lerp(start, finish, t):
    assert len(start) == len(finish), "lerp requires equal length lists as inputs."
    # noinspection PyTypeChecker
    assert 0.0 <= t <= 1.0, "lerp t must be between 0.0 and 1.0 inclusively."
    return [(bx - ax) * t + ax for ax, bx in zip(start, finish)]


def mask_points(points, keep_indicies):
    return np.array([points[idx] for idx in keep_indicies])


def collapse_unique_components(components_list):
    return map(list,
               set(frozenset(i) for i in map(set, [element for sublist in components_list for element in sublist])))


def validate_type(obj, t, name, source):
    assert isinstance(obj, t), "{2} expects type {3} for {1}, found {0}".format(type(obj), name, source, t.__name__)
