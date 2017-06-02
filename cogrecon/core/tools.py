import numpy as np
import itertools
import random

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from math import factorial

from .full_pipeline import minimization_function

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


# noinspection PyUnusedLocal
def brute_force_find_minimal_mapping(p0, p1):
    min_score = np.inf
    min_score_idx = -1
    min_permutation = p1
    idx = 0
    for perm in itertools.permutations(p1):
        score = minimization_function(perm, p0)
        if score < min_score:
            min_score = score
            min_score_idx = idx
            min_permutation = perm
            if score == 0:
                break
        idx += 1
    raise DeprecationWarning
    # return min_score, min_score_idx, min_permutation


# From https://stackoverflow.com/questions/12146910/finding-the-lexicographic-index-of-a-permutation-of-a-given-array
def lexicographic_index(p):
    """
    Return the lexicographic index of the permutation `p` among all
    permutations of its elements. `p` must be a sequence and all elements
    of `p` must be distinct.

    # >>> lexicographic_index('dacb')
    19
    # >>> from itertools import permutations
    # >>> all(lexicographic_index(p) == i
    ...     for i, p in enumerate(permutations('abcde')))
    True
    """
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


# From https://stackoverflow.com/questions/39016821/minimize-total-distance-between-two-sets-of-points-in-python
def find_minimal_mapping(p0, p1):
    C = cdist(p0, p1)

    _, assignment = linear_sum_assignment(C)
    print(assignment)
    p1_reordered = [p1[idx] for idx in assignment]
    return minimization_function(p0, p1_reordered), lexicographic_index(assignment), p1_reordered
