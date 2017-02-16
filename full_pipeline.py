import argparse
import itertools
import logging
# noinspection PyUnresolvedReferences
import sys
# noinspection PyUnresolvedReferences
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
from enum import Enum
from numpy import *
from scipy.spatial import distance

from similarity_transform import similarity_transform

# TODO: Documentation needs an audit/overhaul
# TODO: Metrics which output *which* item is invovled in a swap would be useful to compare between metric types
# TODO: Addition transformation/de-anonymization methods(see https://en.wikipedia.org/wiki/Point_set_registration)
# TODO: Debug geometric transform issue causing it to often produce a transform which lowers accuracy (~36% of the time)


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


def accuracy(actual_points, data_points, z_value=1.96, output_threshold=False):
    dists = diag(distance.cdist(data_points, actual_points))
    mu = mean(dists)
    sd = std(dists)
    ste = sd / sqrt(len(dists))
    ci = ste * z_value
    exclusion_threshold = ci + mu
    dist_accuracy_map = [x < exclusion_threshold for x in dists]
    if output_threshold:
        return dist_accuracy_map, exclusion_threshold
    else:
        return dist_accuracy_map


def axis_swap(actual_points, data_points):
    axis_swaps = 0
    comparisons = 0
    for idx in range(0, len(actual_points)):
        for idx2 in range(idx + 1, len(actual_points)):
            comparisons += 1
            if all(array(map(sign, array(actual_points[idx]) - array(actual_points[idx2]))) !=
                   array(map(sign, array(data_points[idx]) - array(data_points[idx2])))):
                axis_swaps += 1
    axis_swaps = float(axis_swaps) / float(comparisons)
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
                                            array(map(sign, array(data_points[idx]) - array(data_points[idx2])))))\
                .count(False)
    distortions = float(edge_distortions_count) / float(comparisons)

    return distortions


def mask_points(points, keep_indicies):
    return array([points[idx] for idx in keep_indicies])


def geometric_transform(actual_points, data_points, z_value=1.96, debug_labels=None):
    # Determine if the points meet the specified accuracy threshold
    dist_accuracy_map = accuracy(actual_points, data_points, z_value=z_value)

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
                    transformed_coordinates = array(data_points, copy=True)
                    logging.warning(str(debug_labels) + " : " +
                                    ('The transformation function did not reduce the error, removing transform ' +
                                    '(old_error={0}, new_error={1}).').format(old_error,
                                                                              new_error))

        except ValueError:
            transformed_coordinates = array(data_points, copy=True)
            logging.error(('Finding transformation failed , ' +
                           'from_points={0}, to_points={1}.').format(from_points, to_points))
    return (translation_magnitude, scaling, rotation_theta, transformation_auto_exclusion,
            num_geometric_transform_points_excluded, transformed_coordinates)


def swaps(actual_points, data_points, actual_labels, data_labels, z_value=1.96):
    dist_accuracy_map = accuracy(actual_points, data_points,
                                 z_value=z_value)
    accurate_points_labels = [label for (label, is_accurate) in zip(actual_labels, dist_accuracy_map) if is_accurate]

    deanonymized_graph = nx.Graph()
    deanonymized_graph.add_nodes_from(actual_labels)
    deanonymized_graph.add_edges_from(zip(actual_labels, data_labels))
    components = sorted(nx.connected_components(deanonymized_graph), key=len, reverse=True)

    accurate_placements = 0
    inaccurate_placements = 0
    true_swaps = 0
    partial_swaps = 0
    cycle_swaps = 0
    partial_cycle_swaps = 0
    for component in components:
        if len(component) == 1:
            if all([node in accurate_points_labels for node in component]):
                accurate_placements += 1
            else:
                inaccurate_placements += 1
        elif len(component) == 2:
            if all([node in accurate_points_labels for node in component]):
                true_swaps += 1
            else:
                partial_swaps += 1
        elif len(component) > 2:
            if all([node in accurate_points_labels for node in component]):
                cycle_swaps += 1
            else:
                partial_cycle_swaps += 1

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

    return (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
            components, misassignment, accurate_misassignment, inaccurate_misassignment)


def deanonymize(actual_points, data_points):
    perms = list(itertools.permutations(data_points))
    scores = [minimization_function(x, actual_points) for x in perms]
    min_score_position = scores.index(min(scores))
    min_permutation = perms[min_score_position]
    min_score = min(scores)
    min_coordinates = min_permutation
    return min_coordinates, min_score, min_score_position


# animation length in seconds
# animation ticks in frames
def visualization(actual_points, data_points, min_points, transformed_points, output_list,
                  animation_duration=2, animation_ticks=20):

    for l, o in zip(get_header_labels(), output_list):
        print(l + ": " + str(o))

    if len(actual_points[0]) != 2:
        logging.error("the visualization method expects 2D points, found {0}D".format(len(actual_points[0])))
        return

    # Generate a figure with 3 scatter plots (actual points, data points, and transformed points)
    fig, ax = plt.subplots()
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

    accuracies, threshold = accuracy(actual_points, transformed_points, output_threshold=True)

    for acc, x, y in zip(accuracies, transpose(transformed_points)[0], transpose(transformed_points)[1]):
        color = 'r'
        if acc:
            color = 'g'
        ax.add_patch(plt.Circle((x, y), threshold, alpha=0.3, color=color))

    accuracies, threshold = accuracy(actual_points, min_points, output_threshold=True)

    for acc, x, y in zip(accuracies, transpose(min_points)[0], transpose(min_points)[1]):
        ax.add_patch(plt.Circle((x, y), threshold, alpha=0.1, color='b'))

    # An update function which will set the animated scatter plot to the next interpolated points
    def update(i):
        scat.set_offsets(lerp_data[i % animation_ticks])
        return scat,

    # Begin the animation/plot
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, update, interval=(float(animation_duration)/float(animation_ticks))*1000,
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
                  accuracy_z_value=1.96, flags=PipelineFlags.All):
    # First calculate the two primary original metrics, misplacement and axis swaps - this has been validated against
    # the previous script via an MRE data set of 20 individuals
    straight_misplacements = minimization_function(actual_coordinates, data_coordinates) / len(actual_coordinates)
    axis_swaps = axis_swap(actual_coordinates, data_coordinates)
    edge_resize = edge_resizing(actual_coordinates, data_coordinates)
    edge_distort = edge_distortion(actual_coordinates, data_coordinates)

    # De-anonymization via Global Minimization of Misplacement
    # Try all permutations of the data coordinates to find an ordering which is globally minimal in misplacement
    if flags == PipelineFlags.Deanonymize:
        min_coordinates, min_score, min_score_position = deanonymize(actual_coordinates, data_coordinates)
        # Compute the new misplacement value for the raw, deanonymized values
        raw_deanonymized_misplacement = min_score / len(actual_coordinates)  # Standard misplacement
    else:
        min_coordinates = array(data_coordinates, copy=True)
        raw_deanonymized_misplacement = nan
        min_score_position = 0

    if flags == PipelineFlags.GlobalTransformation:
        (translation_magnitude, scaling, rotation_theta, transformation_auto_exclusion,
         num_geometric_transform_points_excluded, transformed_coordinates) = \
            geometric_transform(actual_coordinates, min_coordinates, accuracy_z_value, debug_labels=debug_labels)
    else:
        transformed_coordinates = array(min_coordinates, copy=True)
        transformation_auto_exclusion = nan
        num_geometric_transform_points_excluded = nan
        rotation_theta = nan
        scaling = nan
        translation_magnitude = nan

    # Determine if the points meet the specified accuracy threshold
    deanonymized_labels = list(itertools.permutations(range(0, len(actual_coordinates))))[min_score_position]
    actual_labels = range(len(actual_coordinates))
    (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
     components, misassignment, accurate_misassignment, inaccurate_misassignment) = \
        swaps(actual_coordinates, transformed_coordinates, actual_labels, deanonymized_labels, z_value=accuracy_z_value)

    output = [straight_misplacements,
              axis_swaps,
              edge_resize,
              edge_distort,
              raw_deanonymized_misplacement,
              transformation_auto_exclusion,
              num_geometric_transform_points_excluded,
              rotation_theta,
              scaling,
              translation_magnitude,
              len(components),
              accurate_placements,
              inaccurate_placements,
              true_swaps,
              partial_swaps,
              cycle_swaps,
              partial_cycle_swaps,
              misassignment,
              accurate_misassignment,
              inaccurate_misassignment
              ]

    # If requested, visualize the data
    if visualize:
        visualization(actual_coordinates, data_coordinates, min_coordinates, transformed_coordinates, output)

    return output


# This function is responsible for returning the names of the values returned in full_pipeline
def get_header_labels():
    return ["Original Misplacement", "Original Swap", "Original Edge Resizing", "Original Edge Distortion",
            "Raw Deanonymized Misplacement", "Transformation Auto-Exclusion",
            "Number of Points Excluded From Geometric Transform", "Rotation Theta", "Scaling", "Translation Magnitude",
            "Number of Components", "Accurate Placements", "Inaccurate Placements", "True Swaps",
            "Partial Swaps", "Cycle Swaps", "Partial Cycle Swaps", "Misassignment", "Accurate Misassignment",
            "Inaccurate Misassignment"]


def get_aggregation_functions():
    return [nanmean, nanmean, nanmean, nanmean,
            nanmean, nansum,
            nansum, nanmean, nanmean, nanmean,
            nanmean, nanmean, nanmean, nanmean,
            nanmean, nanmean, nanmean, nanmean,
            nanmean, nanmean]

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
                                                          'accuracy+deanonymization+global transformations+swaps \n('
                                                          'default is 3)', default=3)
    parser.add_argument('--accuracy_z_value', type=float, help='the z value to be used for accuracy exclusion ('
                                                               'default is 1.96, corresponding to 95% confidence',
                        default=1.96)
    parser.add_argument('--dimension', type=int, help='the dimensionality of the data (default is 2)', default=2)

    if len(sys.argv) > 1:
        args = parser.parse_args()
        actual = get_coordinates_from_file(args.actual_coordinates, (args.num_trials, args.num_items, args.dimension))
        data = get_coordinates_from_file(args.data_coordinates, (args.num_trials, args.num_items, args.dimension))
        full_pipeline(actual[args.line_number], data[args.line_number],
                      accuracy_z_value=args.accuracy_z_value, flags=PipelineFlags(args.pipeline_mode), visualize=True)
        exit()

    logging.info("No arguments found - assuming running in test mode.")

    # Test code
    a, b = generate_random_test_points(dimension=3)
    full_pipeline(a, b, visualize=True)
