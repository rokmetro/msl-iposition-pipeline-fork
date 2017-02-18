import argparse
import itertools
import logging
# noinspection PyUnresolvedReferences
import sys
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import warnings

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum
from numpy import *
from scipy.spatial import distance

from similarity_transform import similarity_transform


# TODO: Documentation needs an audit/overhaul
# TODO: Addition transformation/de-anonymization methods(see https://en.wikipedia.org/wiki/Point_set_registration)
# TODO: Debug geometric transform issue causing it to often produce a transform which lowers accuracy (~36% of the time)
# TODO: Additional testing needed to confirm that trial_by_trial_accuracy didn't break anything

class PipelineFlags(Enum):
    Unknown = 0
    Simple = 0
    Deanonymize = 1
    GlobalTransformation = 2
    All = 3

    def __or__(self, other):
        return PipelineFlags(self.value | other.value)

    def __eq__(self, other):
        return (self.value & other.value) != 0


# This function is for testing. It generates a set of "correct" and "incorrect" points such that the correct points are
# randomly placed between (0,0) and (1,1) in R2. Then it generates "incorrect" points which are offset randomly up
# to 10% in the positive direction, shuffled, and where one point is completely random.
def generate_random_test_points(number_of_points=5, dimension=2):
    correct_points = [[random.random() for _ in range(dimension)] for _ in range(0, number_of_points)]
    offsets = [[(random.random()) / 20.0 for _ in range(dimension)] for _ in range(0, number_of_points)]
    input_points = array(correct_points) + array(offsets)

    perms = list(itertools.permutations(input_points))
    input_points = perms[random.randint(0, len(perms) - 1)]
    index = random.randint(0, len(input_points))
    for idx in range(len(input_points[index])):
        input_points[index][idx] = random.random()

    return correct_points, input_points


# This function defines the misplacement metric which is used for minimization (in de-anonymization).
# It is also used to calculate the original misplacement metric.
def minimization_function(list1, list2):
    return sum(diag(distance.cdist(list1, list2)))


# This function performs simple vector linear interpolation on two equal length number lists
def lerp(start, finish, t):
    assert len(start) == len(finish), "lerp requires equal length lists as inputs."
    assert 0.0 <= t <= 1.0, "lerp t must be between 0.0 and 1.0 inclusively."
    return [(bx - ax) * t + ax for ax, bx in zip(start, finish)]


def accuracy(actual_points, data_points, z_value=1.96, output_threshold=False, trial_by_trial_accuracy=True):
    if z_value is None:
            logging.error('a z_value was not found for accuracy, using z=1.96')
            z_value = 1.96
    if trial_by_trial_accuracy:
        dist_accuracy_map = []
        exclusion_thresholds = []
        for actual_trial, data_trial in zip(actual_points, data_points):
            dists = diag(distance.cdist(data_trial, actual_trial))
            mu = mean(dists)
            sd = std(dists)
            ste = sd / sqrt(len(dists))
            ci = ste * z_value
            exclusion_threshold = ci + mu
            exclusion_thresholds.append(exclusion_threshold)
            dist_accuracy_map.append([x < exclusion_threshold for x in dists])
    else:
        collapsed_actual_points = array(actual_points).reshape(-1, len(actual_points[0][0]))
        data_points = [list(x) for x in data_points]
        collapsed_data_points = array(data_points).reshape(-1, len(data_points[0][0]))
        dists = diag(distance.cdist(collapsed_actual_points, collapsed_data_points))
        mu = mean(dists)
        sd = std(dists)
        ste = sd / sqrt(len(dists))
        ci = ste * z_value
        exclusion_threshold = ci + mu
        exclusion_thresholds = [exclusion_threshold] * len(actual_points)
        dist_accuracy_map = []
        for actual_trial, data_trial in zip(actual_points, data_points):
            dists = diag(distance.cdist(data_trial, actual_trial))
            dist_accuracy_map.append([x < exclusion_threshold for x in dists])
    if output_threshold:
        return dist_accuracy_map, exclusion_thresholds
    else:
        return dist_accuracy_map


def axis_swap(actual_points, data_points, actual_labels=None, data_labels=None, generate_pairs=False):
    if not actual_labels:
        actual_labels = range(len(actual_points))
    if not data_labels:
        data_labels = range(len(data_points))
    axis_swaps = 0
    comparisons = 0
    axis_swap_pairs = []
    for idx in range(0, len(actual_points)):
        for idx2 in range(idx + 1, len(actual_points)):
            comparisons += 1
            if all(array(map(sign, array(actual_points[idx]) - array(actual_points[idx2]))) !=
                   array(map(sign, array(data_points[idx]) - array(data_points[idx2])))):
                axis_swaps += 1
                axis_swap_pairs.append([actual_labels[idx], data_labels[idx2]])
    axis_swaps = float(axis_swaps) / float(comparisons)
    if generate_pairs:
        return axis_swaps, axis_swap_pairs
    else:
        return axis_swaps


def edge_resizing(actual_points, data_points):
    actual_edges = []
    data_edges = []
    for idx1 in range(len(actual_points)):
        for idx2 in range(idx1, len(actual_points)):
            actual_edges.append(distance.euclidean(actual_points[idx1], actual_points[idx2]))
            data_edges.append(distance.euclidean(data_points[idx1], data_points[idx2]))
    resizing = mean(abs(array(actual_edges) - array(data_edges)))

    return resizing


def edge_distortion(actual_points, data_points):
    edge_distortions_count = 0
    comparisons = 0
    for idx in range(0, len(actual_points)):
        for idx2 in range(idx + 1, len(actual_points)):
            comparisons += 1
            edge_distortions_count += (list(array(map(sign, array(actual_points[idx]) - array(actual_points[idx2]))) ==
                                            array(map(sign, array(data_points[idx]) - array(data_points[idx2]))))) \
                .count(False)
    distortions = float(edge_distortions_count) / float(comparisons)

    return distortions


def mask_points(points, keep_indicies):
    return array([points[idx] for idx in keep_indicies])


def geometric_transform(actual_points, data_points, z_value=1.96, debug_labels=None, trial_by_trial_accuracy=True):
    # Determine if the points meet the specified accuracy threshold
    dist_accuracy_map, dist_threshold = accuracy(actual_points, data_points,
                                                 z_value=z_value,
                                                 output_threshold=True, trial_by_trial_accuracy=trial_by_trial_accuracy)
    result = []
    for idx, (a, d, dam, dt) in enumerate(zip(actual_points, data_points, dist_accuracy_map, dist_threshold)):
        result.append(trial_geometric_transform(a, d, dam, dt, debug_labels=debug_labels+[idx]))

    return transpose(result)


def trial_geometric_transform(actual_points, data_points, dist_accuracy_map, dist_threshold, debug_labels=None):
    # Determine which points should be included in the transformation step and generate the point sets
    valid_points_indicies = [x for (x, y) in zip(range(len(actual_points)), dist_accuracy_map) if y]
    from_points = mask_points(data_points, valid_points_indicies)
    to_points = mask_points(actual_points, valid_points_indicies)

    # noinspection PyTypeChecker
    # Number of "inaccurate" points after deanonymizing is number of False in this list
    num_geometric_transform_points_excluded = dist_accuracy_map.count(False)
    # noinspection PyTypeChecker
    num_geometric_transform_points_used = dist_accuracy_map.count(True)

    translation_magnitude = nan
    rotation_theta = nan
    scaling = nan
    transformation_auto_exclusion = True
    translation = [nan, nan]
    transformed_coordinates = array(data_points, copy=True)
    # Confirm there are enough points to perform the transformation
    # (it is meaningless to perform with 0 or 1 points)
    if num_geometric_transform_points_used <= 1:
        logging.warning(str(debug_labels) + " : " + ('Not enough points were found to be accurate enough to '
                                                     'create a geometric transform. It will be skipped.'))
    else:
        try:
            # Perform the transformation via Umeyama's method
            rotation_matrix, scaling, translation = similarity_transform(from_points, to_points)
            translation = list(translation)
            # Compute the rotation factor
            theta_matrix = [map(arccos, x) for x in rotation_matrix]
            theta_matrix = [map(abs, x) for x in theta_matrix]
            rotation_theta = mean(theta_matrix)  # Rotation angle
            translation_magnitude = linalg.norm(translation)  # Translation magnitude (direction is in 'translation')
            # Apply the linear transformation to the data coordinates to cancel out global errors if possible
            transformed_coordinates = [(array(x) + array(translation)).dot(rotation_matrix) * scaling
                                       for x in data_points]
            transformation_auto_exclusion = False
            new_error = minimization_function(transformed_coordinates, actual_points)
            old_error = minimization_function(data_points, actual_points)
            if new_error > old_error:  # Exclude rotation from transform
                rotation_theta = nan
                logging.info(str(debug_labels) + " : " +
                             ('The transformation function did not reduce the error, removing rotation and retying' +
                              ' (old_error={0}, new_error={1}).').format(old_error,
                                                                         new_error))
                transformed_coordinates = [(array(x) + array(translation)) * scaling for x in data_points]
                new_error = minimization_function(transformed_coordinates, actual_points)
                old_error = minimization_function(data_points, actual_points)
                if new_error > old_error:  # Completely exclude transform
                    transformation_auto_exclusion = True
                    rotation_theta = nan
                    scaling = nan
                    translation_magnitude = nan
                    translation = [nan, nan]
                    transformed_coordinates = array(data_points, copy=True)
                    logging.warning(str(debug_labels) + " : " +
                                    ('The transformation function did not reduce the error, removing transform ' +
                                     '(old_error={0}, new_error={1}).').format(old_error,
                                                                               new_error))

        except ValueError:
            transformed_coordinates = array(data_points, copy=True)
            logging.error(('Finding transformation failed , ' +
                           'from_points={0}, to_points={1}.').format(from_points, to_points))
    return (translation, translation_magnitude, scaling, rotation_theta, transformation_auto_exclusion,
            num_geometric_transform_points_excluded, transformed_coordinates, dist_threshold)


def swaps(actual_points, data_points, actual_labels, data_labels, z_value=1.96, trial_by_trial_accuracy=True):
    dist_accuracy_map, dist_threshold = accuracy(actual_points, data_points,
                                                 z_value=z_value,
                                                 output_threshold=True, trial_by_trial_accuracy=trial_by_trial_accuracy)
    result = []
    for a, d, al, dl, dam, dt in zip(actual_points, data_points, actual_labels, data_labels,
                                     dist_accuracy_map, dist_threshold):
        result.append(trial_swaps(a, d, al, dl, dam, dt))

    return transpose(result)


def trial_swaps(actual_points, data_points, actual_labels, data_labels, dist_accuracy_map, dist_threshold):
    assert unique(actual_labels).shape == array(actual_labels).shape, \
        "swaps actual_labels are not unique: {0}".format(actual_labels)
    assert unique(data_labels).shape == array(data_labels).shape, \
        "swaps data_labelsare not unique: {0}".format(data_labels)
    assert all(sort(actual_labels) == sort(data_labels)), \
        "swaps actual_labels and data_labels are not unequal: actual, {0}; data, {1}".format(actual_labels, data_labels)

    accurate_points_labels = [label for (label, is_accurate) in zip(actual_labels, dist_accuracy_map) if is_accurate]

    deanonymized_graph = nx.Graph()
    deanonymized_graph.add_nodes_from(actual_labels)
    deanonymized_graph.add_edges_from(zip(actual_labels, data_labels))
    components = sorted(nx.connected_components(deanonymized_graph), key=len, reverse=True)

    accurate_placements = 0
    inaccurate_placements = 0
    true_swaps = 0
    true_swap_distances = []
    true_swap_expected_distances = []
    partial_swaps = 0
    partial_swap_distances = []
    partial_swap_expected_distances = []
    cycle_swaps = 0
    cycle_swap_distances = []
    cycle_swap_expected_distances = []
    partial_cycle_swaps = 0
    partial_cycle_swap_distances = []
    partial_cycle_swap_expected_distances = []
    for component in components:
        component = list(component)
        if len(component) == 1:
            if all([node in accurate_points_labels for node in component]):
                accurate_placements += 1
            else:
                inaccurate_placements += 1
        elif len(component) == 2:
            swap_actual_idx0 = list(actual_labels).index(component[0])
            swap_actual_idx1 = list(actual_labels).index(component[1])
            swap_data_idx0 = list(data_labels).index(component[0])
            swap_data_idx1 = list(data_labels).index(component[1])
            dist_actual = distance.euclidean(actual_points[swap_actual_idx0], actual_points[swap_actual_idx1])
            dist_data = distance.euclidean(data_points[swap_data_idx0], data_points[swap_data_idx1])
            if all([node in accurate_points_labels for node in component]):
                true_swaps += 1
                true_swap_distances.append(dist_data)
                true_swap_expected_distances.append(dist_actual)
            else:
                partial_swaps += 1
                partial_swap_distances.append(dist_data)
                partial_swap_expected_distances.append(dist_actual)
        elif len(component) > 2:
            swap_actual_idxs = [list(actual_labels).index(c) for c in component]
            swap_actual_idxs_combinations = [(a, b) for (a, b) in
                                             list(itertools.product(swap_actual_idxs, swap_actual_idxs)) if a != b]
            swap_data_idxs = [list(actual_labels).index(c) for c in component]
            swap_data_idxs_combinations = [(a, b) for (a, b) in
                                           list(itertools.product(swap_data_idxs, swap_data_idxs)) if a != b]
            dists_actual = [distance.euclidean(actual_points[a], actual_points[b]) for (a, b) in
                            swap_actual_idxs_combinations]
            dists_data = [distance.euclidean(data_points[a], data_points[b]) for (a, b) in
                          swap_data_idxs_combinations]
            if all([node in accurate_points_labels for node in component]):
                cycle_swaps += 1
                cycle_swap_distances.append(mean(dists_data))
                cycle_swap_expected_distances.append(mean(dists_actual))
            else:
                partial_cycle_swaps += 1
                partial_cycle_swap_distances.append(mean(dists_data))
                partial_cycle_swap_expected_distances.append(mean(dists_actual))

    misassignment = 0
    accurate_misassignment = 0
    inaccurate_misassignment = 0
    for actual_label, data_label, acc in zip(actual_labels, data_labels, dist_accuracy_map):
        if actual_label != data_label:
            misassignment += 1
            if acc:
                accurate_misassignment += 1
            else:
                inaccurate_misassignment += 1

    with warnings.catch_warnings():  # Ignore empty mean warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
                components, misassignment, accurate_misassignment, inaccurate_misassignment, dist_threshold,
                nanmean(true_swap_distances), nanmean(true_swap_expected_distances),
                nanmean(partial_swap_distances), nanmean(partial_swap_expected_distances),
                nanmean(cycle_swap_distances), nanmean(cycle_swap_expected_distances),
                nanmean(partial_cycle_swap_distances), nanmean(partial_cycle_swap_expected_distances))


def deanonymize(actual_points, data_points):
    min_coordinates = []
    min_score = []
    min_score_position = []
    raw_deanonymized_misplacement = []
    for actual_trial, data_trial in zip(actual_points, data_points):
        perms = list(itertools.permutations(data_trial))
        scores = [minimization_function(x, actual_trial) for x in perms]
        min_score_pos = scores.index(min(scores))
        min_score_position.append(min_score_pos)
        min_permutation = perms[min_score_pos]
        min_s = min(scores)
        min_score.append(min_s)
        min_coordinates.append(min_permutation)
        raw_deanonymized_misplacement.append(min_s / len(actual_trial))
    return min_coordinates, min_score, min_score_position, raw_deanonymized_misplacement


# animation length in seconds
# animation ticks in frames
def visualization(actual_points, data_points, min_points, transformed_points, output_list,
                  z_value=1.96,
                  animation_duration=2, animation_ticks=20, debug_labels=""):
    for l, o in zip(get_header_labels(), output_list):
        print(l + ": " + str(o))

    if len(actual_points[0]) != 2:
        logging.error("the visualization method expects 2D points, found {0}D".format(len(actual_points[0])))
        return

    # Generate a figure with 3 scatter plots (actual points, data points, and transformed points)
    fig, ax = plt.subplots()
    plt.title(str(debug_labels))
    ax.set_aspect('equal')
    labels = range(len(actual_points))
    x = [float(v) for v in list(transpose(transformed_points)[0])]
    y = [float(v) for v in list(transpose(transformed_points)[1])]
    ax.scatter(x, y, c='b', alpha=0.5)
    scat = ax.scatter(x, y, c='b', animated=True)
    ax.scatter(transpose(actual_points)[0], transpose(actual_points)[1], c='g', s=50)
    ax.scatter(transpose(data_points)[0], transpose(data_points)[1], c='r', s=50)
    # Label the stationary points (actual and data)
    for idx, xy in enumerate(zip(transpose(actual_points)[0], transpose(actual_points)[1])):
        ax.annotate(labels[idx], xy=xy, textcoords='data', fontsize=20)
    for idx, xy in enumerate(zip(transpose(data_points)[0], transpose(data_points)[1])):
        ax.annotate(labels[idx], xy=xy, textcoords='data', fontsize=20)
    # Generate a set of interpolated points to animate the transformation
    lerp_data = [[lerp(p1, p2, t) for p1, p2 in zip(min_points, transformed_points)] for t in
                 linspace(0.0, 1.0, animation_ticks)]

    accuracies, threshold = accuracy([actual_points], [transformed_points],
                                     z_value=z_value, output_threshold=True, trial_by_trial_accuracy=True)
    accuracies = accuracies[0]
    threshold = threshold[0]
    for acc, x, y in zip(accuracies, transpose(transformed_points)[0], transpose(transformed_points)[1]):
        color = 'r'
        if acc:
            color = 'g'
        ax.add_patch(plt.Circle((x, y), threshold, alpha=0.3, color=color))

    accuracies, threshold = accuracy([actual_points], [min_points],
                                     z_value=z_value, output_threshold=True, trial_by_trial_accuracy=True)
    accuracies = accuracies[0]
    threshold = threshold[0]

    for acc, x, y in zip(accuracies, transpose(min_points)[0], transpose(min_points)[1]):
        ax.add_patch(plt.Circle((x, y), threshold, alpha=0.1, color='b'))

    # An update function which will set the animated scatter plot to the next interpolated points
    def update(i):
        scat.set_offsets(lerp_data[i % animation_ticks])
        return scat,

    # Begin the animation/plot
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, update, interval=(float(animation_duration) / float(animation_ticks)) * 1000,
                                   blit=True)
    plt.show()


# This function reads a data file and shapes the data into the appropriate expected shape (usually (Nt, Ni, 2) where
# Nt is the number of trials (rows) and Ni is the number of items (columns / 2), and 2 is the number of dimensions.
def get_coordinates_from_file(path, expected_shape):
    with open(path) as tsv:
        coordinates = zip(*([float(element) for element in line.strip().split('\t')] for line in tsv))
        coordinates = transpose(coordinates)
    if expected_shape is not None:
        coordinates = reshape(array(coordinates), expected_shape)
        assert array(coordinates).shape == expected_shape, \
            "shape {0} does not equal expectation {1}".format(array(coordinates).shape, expected_shape)
    return coordinates


# This function grabs the first 3 characters of the filename which are assumed to be the participant id
def get_id_from_file_prefix(path, prefix_length=3):
    return os.path.basename(path)[0:prefix_length]


# This function is the main pipeline for the new processing methods. When run alone, it just returns the values
# for a single trial. With visualize=True it will display the results. debug_labels is used to help specify
# which participant/trial is being observed when running from an external process (it is appended to the debug info).
# The coordinates are expected to be equal in length of the for (Nt, Ni, 2) where Nt is the number of trials and Ni is
# the number of items.
def full_pipeline(actual_coordinates, data_coordinates, visualize=False, debug_labels=None,
                  accuracy_z_value=1.96, flags=PipelineFlags.All, trial_by_trial_accuracy=True):
    # If only a single trial worth of points is input, flex the data so it's the right dimension
    if len(array(actual_coordinates).shape) == 2:
        actual_coordinates = [actual_coordinates]
    if len(array(data_coordinates).shape) == 2:
        data_coordinates = [data_coordinates]

    num_trials = len(actual_coordinates)

    straight_misplacements = []
    axis_swaps = []
    axis_swap_pairs = []
    edge_resize = []
    edge_distort = []
    for (actual_trial, data_trial) in zip(actual_coordinates, data_coordinates):
        # First calculate the two primary original metrics, misplacement and axis swaps - this has been validated
        # against the previous script via an MRE data set of 20 individuals
        straight_misplacements.append(minimization_function(actual_trial, data_trial) / len(actual_trial))
        axis_swaps_element, axis_swap_pairs_element = axis_swap(actual_trial, data_trial, generate_pairs=True)
        axis_swaps.append(axis_swaps_element)
        axis_swap_pairs.append(axis_swap_pairs_element)
        edge_resize.append(edge_resizing(actual_trial, data_trial))
        edge_distort.append(edge_distortion(actual_trial, data_trial))

    pre_process_accuracies, pre_process_threshold = accuracy(actual_coordinates, data_coordinates,
                                                             z_value=accuracy_z_value,
                                                             output_threshold=True,
                                                             trial_by_trial_accuracy=trial_by_trial_accuracy)

    # De-anonymization via Global Minimization of Misplacement
    # Try all permutations of the data coordinates to find an ordering which is globally minimal in misplacement
    if flags == PipelineFlags.Deanonymize:
        min_coordinates, min_score, min_score_position, raw_deanonymized_misplacement = \
            deanonymize(actual_coordinates,
                        data_coordinates)
        deanon_accuracies, deanon_threshold = accuracy(actual_coordinates, data_coordinates,
                                                       z_value=accuracy_z_value,
                                                       output_threshold=True,
                                                       trial_by_trial_accuracy=trial_by_trial_accuracy)
    else:
        min_coordinates = array(data_coordinates, copy=True)
        deanon_threshold = [nan] * num_trials
        deanon_accuracies = [] * num_trials
        raw_deanonymized_misplacement = [nan] * num_trials
        min_score_position = [0] * num_trials

    if flags == PipelineFlags.GlobalTransformation:
        (translation, translation_magnitude, scaling, rotation_theta,
         transformation_auto_exclusion, num_geometric_transform_points_excluded,
         transformed_coordinates, geo_dist_threshold) = \
            geometric_transform(actual_coordinates, min_coordinates, z_value=accuracy_z_value,
                                debug_labels=debug_labels, trial_by_trial_accuracy=trial_by_trial_accuracy)
    else:
        translation = [[nan, nan]] * num_trials
        transformed_coordinates = array(min_coordinates, copy=True)
        transformation_auto_exclusion = [nan] * num_trials
        num_geometric_transform_points_excluded = [nan] * num_trials
        rotation_theta = [nan] * num_trials
        scaling = [nan] * num_trials
        translation_magnitude = [nan] * num_trials
        geo_dist_threshold = [nan] * num_trials

    # Determine if the points meet the specified accuracy threshold
    # noinspection PyTypeChecker
    deanonymized_labels = [list(itertools.permutations(range(0, len(actual_coordinates[0]))))[position] for position in
                           min_score_position]
    actual_labels = [range(len(actual_trial)) for actual_trial in
                     actual_coordinates]
    (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
     components, misassignment, accurate_misassignment, inaccurate_misassignment, swap_dist_threshold,
     true_swap_distances, true_swap_expected_distances,
     partial_swap_distances, partial_swap_expected_distances,
     cycle_swap_distances, cycle_swap_expected_distances,
     partial_cycle_swap_distances, partial_cycle_swap_expected_distances) = \
        swaps(actual_coordinates, transformed_coordinates, actual_labels, deanonymized_labels,
              z_value=accuracy_z_value, trial_by_trial_accuracy=trial_by_trial_accuracy)

    output = transpose(
             [straight_misplacements,
              axis_swaps,
              edge_resize,
              edge_distort,
              axis_swap_pairs,
              [list(x).count(True) for x in pre_process_accuracies],
              [list(x).count(False) for x in pre_process_accuracies],
              pre_process_threshold,
              [list(x).count(True) for x in deanon_accuracies],
              [list(x).count(False) for x in deanon_accuracies],
              deanon_threshold,
              raw_deanonymized_misplacement,
              transformation_auto_exclusion,
              num_geometric_transform_points_excluded,
              rotation_theta,
              scaling,
              translation_magnitude,
              translation,
              geo_dist_threshold,
              [len(x) for x in components],
              accurate_placements,
              inaccurate_placements,
              true_swaps,
              partial_swaps,
              cycle_swaps,
              partial_cycle_swaps,
              misassignment,
              accurate_misassignment,
              inaccurate_misassignment,
              swap_dist_threshold,
              true_swap_distances,
              true_swap_expected_distances,
              partial_swap_distances,
              partial_swap_expected_distances,
              cycle_swap_distances,
              cycle_swap_expected_distances,
              partial_cycle_swap_distances,
              partial_cycle_swap_expected_distances,
              [map(list, x) for x in components]
              ])

    # If requested, visualize the data
    if visualize:
        for idx, (actual_trial, data_trial, min_trial, transformed_trial, output_trial) in \
                enumerate(zip(actual_coordinates, data_coordinates, min_coordinates, transformed_coordinates, output)):
            visualization(actual_trial, data_trial, min_trial, transformed_trial, output_trial,
                          z_value=accuracy_z_value, debug_labels=debug_labels + [idx])

    return output


# This function is responsible for returning the names of the values returned in full_pipeline
def get_header_labels():
    return ["Original Misplacement", "Original Swap", "Original Edge Resizing", "Original Edge Distortion",    # 0
            "Axis Swap Pairs", "Pre-Processed Accurate Placements", "Pre-Processed Inaccurate Placements",     # 1
            "Pre-Processed Accuracy Threshold", "Deanonymized Accurate Placements",                            # 2
            "Deanonymized Inaccurate Placements", "Deanonymized Accuracy Threshold",                           # 3
            "Raw Deanonymized Misplacement", "Transformation Auto-Exclusion",                                  # 4
            "Number of Points Excluded From Geometric Transform", "Rotation Theta", "Scaling",                 # 5
            "Translation Magnitude",                                                                           # 6
            "Translation", "Geometric Distance Threshold",                                                     # 7
            "Number of Components", "Accurate Placements", "Inaccurate Placements", "True Swaps",              # 8
            "Partial Swaps", "Cycle Swaps", "Partial Cycle Swaps", "Misassignment", "Accurate Misassignment",  # 9
            "Inaccurate Misassignment", "Swap Distance Threshold",                                             # 10
            "True Swap Data Distance", "True Swap Actual Distance", "Partial Swap Data Distance",              # 11
            "Partial Swap Actual Distance", "Cycle Swap Data Distance", "Cycle Swap Actual Distance",          # 12
            "Partial Cycle Swap Data Distance", "Partial Cycle Swap Actual Distance",                          # 13
            "Unique Components"]                                                                               # 14


def collapse_unique_components(components_list):
    return map(list,
               set(frozenset(i) for i in map(set, [element for sublist in components_list for element in sublist])))


# (lambda x: list(array(x).flatten())) for append
def get_aggregation_functions():
    return [nanmean, nanmean, nanmean, nanmean,                                                                # 0
            collapse_unique_components, nanmean, nanmean,                                                      # 1
            nanmean, nanmean,                                                                                  # 2
            nanmean, nanmean,                                                                                  # 3
            nanmean, nansum,                                                                                   # 4
            nansum, nanmean, nanmean,                                                                          # 5
            nanmean,                                                                                           # 6
            (lambda xs: [nanmean(x) for x in transpose(xs)]), nanmean,  # Mean of vectors                      # 7
            nanmean, nanmean, nanmean, nanmean,                                                                # 8
            nanmean, nanmean, nanmean, nanmean, nanmean,                                                       # 9
            nanmean, nanmean,                                                                                  # 10
            nanmean, nanmean, nanmean,                                                                         # 11
            nanmean, nanmean, nanmean,                                                                         # 12
            nanmean, nanmean,                                                                                  # 13
            collapse_unique_components]                                                                        # 14


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    parser.add_argument('actual_coordinates', type=str, help='the path to the file containing the actual coordinates')
    parser.add_argument('data_coordinates', type=str, help='the path to the file containing the data coordinates')
    parser.add_argument('num_trials', type=int, help='the number of trials in the file')
    parser.add_argument('num_items', type=int, help='the number of items to be analyzed')
    parser.add_argument('line_number', type=int, help='the line number to be processed (starting with 0) - typically '
                                                      'the trial number minus 1.')
    parser.add_argument('--pipeline_mode', type=int, help='the mode in which the pipeline should process; \n\t0 for '
                                                          'just accuracy+swaps, \n\t1 for '
                                                          'accuracy+deanonymization+swaps, \n\t2 for accuracy+global '
                                                          'transformations+swaps, \n\t3 for '
                                                          'accuracy+deanonymization+global transformations+swaps \n'
                                                          '(default is 3)', default=3)
    parser.add_argument('--accuracy_z_value', type=float, help='the z value to be used for accuracy exclusion ('
                                                               'default is 1.96), corresponding to 95% confidence; if ',
                        default=1.96)
    parser.add_argument('--dimension', type=int, help='the dimensionality of the data (default is 2)', default=2)

    if len(sys.argv) > 1:
        args = parser.parse_args()
        actual = get_coordinates_from_file(args.actual_coordinates, (args.num_trials, args.num_items, args.dimension))
        data = get_coordinates_from_file(args.data_coordinates, (args.num_trials, args.num_items, args.dimension))
        full_pipeline(actual[args.line_number], data[args.line_number],
                      accuracy_z_value=args.accuracy_z_value,
                      flags=PipelineFlags(args.pipeline_mode), visualize=True,
                      debug_labels=[get_id_from_file_prefix(args.data_coordinates), args.line_number])
        exit()

    logging.info("No arguments found - assuming running in test mode.")

    # Test code
    # a, b = generate_random_test_points(dimension=3)
    # full_pipeline(a, b, visualize=True)

    root_dir = r"Z:\Kevin\iPosition\Hillary\MRE\\"
    actual = get_coordinates_from_file(root_dir + r"actual_coordinates.txt", (15, 5, 2))
    data101 = get_coordinates_from_file(root_dir + r"101\101position_data_coordinates.txt", (15, 5, 2))
    data104 = get_coordinates_from_file(root_dir + r"104\104position_data_coordinates.txt", (15, 5, 2))
    data105 = get_coordinates_from_file(root_dir + r"105\105position_data_coordinates.txt", (15, 5, 2))
    data112 = get_coordinates_from_file(root_dir + r"112\112position_data_coordinates.txt", (15, 5, 2))
    data113 = get_coordinates_from_file(root_dir + r"113\113position_data_coordinates.txt", (15, 5, 2))
    data114 = get_coordinates_from_file(root_dir + r"114\114position_data_coordinates.txt", (15, 5, 2))
    data118 = get_coordinates_from_file(root_dir + r"118\118position_data_coordinates.txt", (15, 5, 2))
    data119 = get_coordinates_from_file(root_dir + r"119\119position_data_coordinates.txt", (15, 5, 2))
    data120 = get_coordinates_from_file(root_dir + r"120\120position_data_coordinates.txt", (15, 5, 2))

    # Cycle Agree
    full_pipeline(actual[10], data101[10], visualize=True, debug_labels=["101", 10])
    full_pipeline(actual[12], data104[12], visualize=True, debug_labels=["104", 12])
    full_pipeline(actual[2], data105[2], visualize=True, debug_labels=["105", 2])
    full_pipeline(actual[6], data112[6], visualize=True, debug_labels=["112", 6])

    # Old Swap, New Cycle (only truly debatable one in my opinion)
    full_pipeline(actual[2], data104[2], visualize=True, debug_labels=["104", 2])

    # New Single Swap
    full_pipeline(actual[0], data101[0], visualize=True, debug_labels=["101", 0])
    full_pipeline(actual[12], data114[12], visualize=True, debug_labels=["114", 12])
    full_pipeline(actual[10], data118[10], visualize=True, debug_labels=["118", 10])
    full_pipeline(actual[10], data119[10], visualize=True, debug_labels=["119", 10])
    full_pipeline(actual[14], data120[14], visualize=True, debug_labels=["120", 14])

    # False Alarms (one or more old swap where new disagrees)
    full_pipeline(actual[11], data101[11], visualize=True, debug_labels=["101", 11])  # 3 too many
    full_pipeline(actual[10], data104[10], visualize=True, debug_labels=["104", 10])  # 1 too many
    full_pipeline(actual[2], data113[2], visualize=True, debug_labels=["113", 2])  # 2 too many
    full_pipeline(actual[12], data120[12], visualize=True, debug_labels=["120", 12])  # 2 too many
