import numpy as np
import itertools
import random
import sys

import scipy.spatial.distance as distance
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from math import factorial

from .cogrecon_globals import default_dimensions

# TODO: Documentation needs an audit/overhaul


# This function defines the misplacement metric which is used for minimization (in de-anonymization).
# It is also used to calculate the original misplacement metric.
def sum_of_distance(list1, list2):
    """

    :param list1:
    :param list2:
    :return:
    """

    return sum(np.diag(distance.cdist(list1, list2)))


# This function is for testing. It generates a set of "correct" and "incorrect" points such that the correct points are
# randomly placed between (0,0) and (1,1) in R2. Then it generates "incorrect" points which are offset randomly up
# to 10% in the positive direction, shuffled, and where one point is completely random.
def generate_random_test_points(number_of_points=5, dimension=default_dimensions,
                                shuffle_points=True, noise=(1.0 / 20.0),
                                num_rerandomed_points=1):
    """

    :param number_of_points:
    :param dimension:
    :param shuffle_points:
    :param noise:
    :param num_rerandomed_points:
    :return:
    """
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
    """

    :param start:
    :param finish:
    :param t:
    :return:
    """
    assert len(start) == len(finish), "lerp requires equal length lists as inputs."
    # noinspection PyTypeChecker
    assert 0.0 <= t <= 1.0, "lerp t must be between 0.0 and 1.0 inclusively."
    return [(bx - ax) * t + ax for ax, bx in zip(start, finish)]


def mask_points(points, keep_indicies):
    """

    :param points:
    :param keep_indicies:
    :return:
    """
    return np.array([points[idx] for idx in keep_indicies])


def mask_dimensions(data3d, num_dims, remove_dims):
    """

    :param data3d:
    :param num_dims:
    :param remove_dims:
    :return:
    """
    return np.transpose([np.transpose(data3d, axes=(2, 1, 0))[i] for i in range(num_dims) if i not in remove_dims],
                        axes=(2, 1, 0))


def collapse_unique_components(components_list):
    # Filter NoneType from list as it is not orderable
    """

    :param components_list:
    :return:
    """
    filtered_list = [x for x in components_list if x is not None]
    # Generate the unique list
    unique_list = np.unique(filtered_list).tolist()
    # Add in a single None value if the filter found a NoneType to maintain a record of its existence
    if len(filtered_list) < len(components_list):
        unique_list += [None]
    return unique_list


def validate_type(obj, t, name, source):
    """

    :param obj:
    :param t:
    :param name:
    :param source:
    """
    assert isinstance(obj, t), "{2} expects type {3} for {1}, found {0}".format(type(obj), name, source, t.__name__)


# noinspection PyUnusedLocal
def brute_force_find_minimal_mapping(p0, p1):
    """

    :param p0:
    :param p1:
    """
    min_score = np.inf
    min_score_idx = -1
    min_permutation = p1
    idx = 0
    for perm in itertools.permutations(p1):
        score = sum_of_distance(perm, p0)
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
    :param p:
    :return:
    """
    result = 0
    for j in range(len(p)):
        k = sum(1 for i in p[j + 1:] if i < p[j])
        result += k * factorial(len(p) - j - 1)
    return result


# From https://stackoverflow.com/questions/39016821/minimize-total-distance-between-two-sets-of-points-in-python
def find_minimal_mapping(p0, p1):
    """

    :param p0:
    :param p1:
    :return:
    """
    C = cdist(p0, p1)

    _, assignment = linear_sum_assignment(C)

    p1_reordered = [p1[idx] for idx in assignment]
    return sum_of_distance(p0, p1_reordered), lexicographic_index(assignment), p1_reordered


# TODO: There's a problem with this function - it's always returning the identity order ([0, 1, 2, ..., n])
# noinspection PyUnusedLocal
def greedy_find_minimal_mapping(p0, p1, order):
    """

    :param p0:
    :param p1:
    :param order:
    :return:
    """
    assert len(p0) == len(p1) and len(p0) == len(order), "greedy_find_minimal_mapping requires all list to be equal " \
                                                         "in length "
    assert sorted(list(set(order))) == list(range(len(order))), "greedy_find_minimal_mapping order should contain " \
                                                                "unique values between 0 and n-1, inclusively, " \
                                                                "where n is the number of items "
    # Sort by order
    indices = list(range(len(order)))
    indices.sort(key=order.__getitem__)
    sorted_p0 = list(map(p0.__getitem__, indices))
    sorted_p1 = list(map(p1.__getitem__, indices))

    map_order = []
    for idx, p in enumerate(sorted_p1):
        min_dist = sys.float_info.max
        min_idx = -1
        for idxx, pp in enumerate(sorted_p0):
            dist = distance.euclidean(p, pp)
            if dist < min_dist:
                min_dist = dist
                min_idx = idx
        map_order.append(min_idx)
        del sorted_p1[min_idx]

    p1_reordered = [p1[idx] for idx in map_order]

    return sum_of_distance(p0, p1_reordered), lexicographic_index(map_order), p1_reordered
