import argparse
import logging
# noinspection PyUnresolvedReferences
import sys
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import warnings
import itertools
import networkx as nx
from enum import Enum
import numpy as np
from scipy.spatial import distance

from similarity_transform import similarity_transform
import io
import tools
import visualization as vis
from data import TrialData, ParticipantData, AnalysisConfiguration


# TODO: Documentation needs an audit/overhaul
# TODO: Addition transformation/de-anonymization methods(see https://en.wikipedia.org/wiki/Point_set_registration)
# TODO: Additional testing needed to confirm that trial_by_trial_accuracy didn't break anything

class PipelineFlags(Enum):
    Unknown = 0
    Simple = 0
    Deanonymize = 1
    GlobalTransformation = 2
    All = 3
    value = 0

    def __or__(self, other):
        return PipelineFlags(self.value | other.value)

    def __eq__(self, other):
        return (self.value & other.value) != 0


# This function defines the misplacement metric which is used for minimization (in de-anonymization).
# It is also used to calculate the original misplacement metric.
def minimization_function(list1, list2):
    return sum(np.diag(distance.cdist(list1, list2)))


def accuracy(participant_data, analysis_configuration):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "edge_resizing")
    tools.validate_type(analysis_configuration, type(AnalysisConfiguration),
                        "analysis_configuration", "trial_geometric_transform")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points
    z_value = analysis_configuration.z_value
    trial_by_trial_accuracy = analysis_configuration.trial_by_trial_accuracy
    manual_threshold = analysis_configuration.manual_threshold

    if z_value is None:
        logging.error('a z_value was not found for accuracy, using z=1.96')
        z_value = 1.96
    if trial_by_trial_accuracy:
        dist_accuracy_map = []
        exclusion_thresholds = []
        for actual_trial, data_trial in zip(actual_points, data_points):
            if manual_threshold is not None:
                dists = np.diag(distance.cdist(data_trial, actual_trial))
                dist_accuracy_map.append([xd < manual_threshold for xd in dists])
                exclusion_thresholds.append(manual_threshold)
            else:
                dists = np.diag(distance.cdist(data_trial, actual_trial))
                mu = np.mean(dists)
                sd = np.std(dists)
                ste = sd / np.sqrt(len(dists))
                ci = ste * z_value
                exclusion_threshold = ci + mu
                exclusion_thresholds.append(exclusion_threshold)
                dist_accuracy_map.append([xd < exclusion_threshold for xd in dists])
    else:
        dist_accuracy_map = []
        if manual_threshold is not None:
            for actual_trial, data_trial in zip(actual_points, data_points):
                dists = np.diag(distance.cdist(data_trial, actual_trial))
                dist_accuracy_map.append([xd < manual_threshold for xd in dists])
            exclusion_thresholds = [manual_threshold] * len(actual_points)
        else:
            collapsed_actual_points = np.array(actual_points).reshape(-1, len(actual_points[0][0]))
            data_points = [list(xd) for xd in data_points]
            collapsed_data_points = np.array(data_points).reshape(-1, len(data_points[0][0]))
            dists = np.diag(distance.cdist(collapsed_actual_points, collapsed_data_points))
            mu = np.mean(dists)
            sd = np.std(dists)
            ste = sd / np.sqrt(len(dists))
            ci = ste * z_value
            exclusion_threshold = ci + mu
            exclusion_thresholds = [exclusion_threshold] * len(actual_points)

            for actual_trial, data_trial in zip(actual_points, data_points):
                # noinspection PyTypeChecker
                dists = np.diag(distance.cdist(data_trial, actual_trial))
                dist_accuracy_map.append([xd < exclusion_threshold for xd in dists])

    participant_data.distance_accuracy_map = dist_accuracy_map
    participant_data.distance_threshold = exclusion_thresholds

    return participant_data


def axis_swap(participant_data):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "axis_swap")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points
    actual_labels = participant_data.actual_labels
    data_labels = participant_data.data_labels

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
            if all(np.array(map(np.sign, np.array(actual_points[idx]) - np.array(actual_points[idx2]))) !=
                   np.array(map(np.sign, np.array(data_points[idx]) - np.array(data_points[idx2])))):
                axis_swaps += 1
                axis_swap_pairs.append([actual_labels[idx], data_labels[idx2]])
    axis_swaps = float(axis_swaps) / float(comparisons)
    return axis_swaps, axis_swap_pairs


def edge_resizing(participant_data):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "edge_resizing")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points

    actual_edges = []
    data_edges = []
    for idx1 in range(len(actual_points)):
        for idx2 in range(idx1, len(actual_points)):
            actual_edges.append(distance.euclidean(actual_points[idx1], actual_points[idx2]))
            data_edges.append(distance.euclidean(data_points[idx1], data_points[idx2]))
    # noinspection PyTypeChecker
    resizing = np.mean(abs(np.array(actual_edges) - np.array(data_edges)))

    return resizing


def edge_distortion(participant_data):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "edge_distortion")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points

    edge_distortions_count = 0
    comparisons = 0
    for idx in range(0, len(actual_points)):
        for idx2 in range(idx + 1, len(actual_points)):
            comparisons += 1
            actual_signs = np.array(list((map(np.sign, np.array(actual_points[idx]) - np.array(actual_points[idx2])))))
            data_signs = np.array(list(map(np.sign, np.array(data_points[idx]) - np.array(data_points[idx2]))))
            equality_list = list(actual_signs == data_signs)
            edge_distortions_count += equality_list.count(False)

    distortions = float(edge_distortions_count) / float(comparisons)

    return distortions


# noinspection PyDefaultArgument
def geometric_transform(participant_data, analysis_configuration):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "geometric_transform")
    tools.validate_type(analysis_configuration, type(AnalysisConfiguration),
                        "analysis_configuration", "geometric_transform")

    # Determine if the points meet the specified accuracy threshold
    participant_data = accuracy(participant_data, analysis_configuration)
    result = []
    for trial in participant_data.trials:
        # noinspection PyTypeChecker
        result.append(trial_geometric_transform(trial, analysis_configuration))

    return np.transpose(result)


# noinspection PyDefaultArgument
def trial_geometric_transform(trial_data, analysis_configuration):
    tools.validate_type(trial_data, type(TrialData), "trial_data", "trial_geometric_transform")
    tools.validate_type(analysis_configuration, type(AnalysisConfiguration),
                        "analysis_configuration", "trial_geometric_transform")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points
    dist_accuracy_map = trial_data.dist_accuracy_map
    dist_threshold = trial_data.dist_threshold
    debug_labels = analysis_configuration.debug_labels

    # Determine which points should be included in the transformation step and generate the point sets
    valid_points_indicies = [x for (x, y) in zip(range(len(actual_points)), dist_accuracy_map) if y]
    from_points = tools.mask_points(data_points, valid_points_indicies)
    to_points = tools.mask_points(actual_points, valid_points_indicies)

    # noinspection PyTypeChecker
    # Number of "inaccurate" points after deanonymizing is number of False in this list
    num_geometric_transform_points_excluded = dist_accuracy_map.count(False)
    # noinspection PyTypeChecker
    num_geometric_transform_points_used = dist_accuracy_map.count(True)

    translation_magnitude = np.nan
    rotation_theta = np.nan
    scaling = np.nan
    transformation_auto_exclusion = True
    translation = [np.nan, np.nan]
    transformed_coordinates = np.array(data_points, copy=True)
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
            theta_matrix = [map(np.arccos, x) for x in rotation_matrix]
            theta_matrix = [map(abs, x) for x in theta_matrix]
            rotation_theta = np.mean([list(x) for x in theta_matrix])  # Rotation angle
            translation_magnitude = np.linalg.norm(translation)  # Translation magnitude (direction is in 'translation')
            # Apply the linear transformation to the data coordinates to cancel out global errors if possible
            transformed_coordinates = [(np.array(x) + np.array(translation)).dot(rotation_matrix) * scaling
                                       for x in data_points]
            transformation_auto_exclusion = False
            new_error = minimization_function(transformed_coordinates, actual_points)
            old_error = minimization_function(data_points, actual_points)
            if new_error > old_error:  # Exclude rotation from transform
                rotation_theta = np.nan
                logging.info(str(debug_labels) + " : " +
                             ('The transformation function did not reduce the error, removing rotation and retying' +
                              ' (old_error={0}, new_error={1}).').format(old_error,
                                                                         new_error))
                transformed_coordinates = [(np.array(x) + np.array(translation)) * scaling for x in data_points]
                new_error = minimization_function(transformed_coordinates, actual_points)
                old_error = minimization_function(data_points, actual_points)
                if new_error > old_error:  # Completely exclude transform
                    transformation_auto_exclusion = True
                    rotation_theta = np.nan
                    scaling = np.nan
                    translation_magnitude = np.nan
                    translation = [np.nan, np.nan]
                    transformed_coordinates = np.array(data_points, copy=True)
                    logging.warning(str(debug_labels) + " : " +
                                    ('The transformation function did not reduce the error, removing transform ' +
                                     '(old_error={0}, new_error={1}).').format(old_error,
                                                                               new_error))

        except ValueError:
            transformed_coordinates = np.array(data_points, copy=True)
            logging.error(('Finding transformation failed , ' +
                           'from_points={0}, to_points={1}.').format(from_points, to_points))
    return (translation, translation_magnitude, scaling, rotation_theta, transformation_auto_exclusion,
            num_geometric_transform_points_excluded, transformed_coordinates, dist_threshold)


def swaps(participant_data, analysis_configuration):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "swaps")
    tools.validate_type(analysis_configuration, type(AnalysisConfiguration), "analysis_configuration", "swaps")

    participant_data = accuracy(participant_data, analysis_configuration)

    result = []
    for trial in participant_data.trials:
        result.append(trial_swaps(trial))

    return np.transpose(result)


def trial_swaps(trial_data):
    # TODO: include this error checking in the ParticipantData object
    # assert np.unique(actual_labels).shape == np.array(actual_labels).shape, \
    #     "swaps actual_labels are not unique: {0}".format(actual_labels)
    # assert np.unique(data_labels).shape == np.array(data_labels).shape, \
    #     "swaps data_labelsare not unique: {0}".format(data_labels)
    # assert all(np.sort(actual_labels) == np.sort(data_labels)), \
    #     ("swaps actual_labels and data_labels are " +
    #      "not unequal: actual, {0}; data, {1}").format(actual_labels, data_labels)

    tools.validate_type(trial_data, type(TrialData), "trial_data", "trial_swaps")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points
    actual_labels = trial_data.actual_labels
    data_labels = trial_data.data_labels
    dist_accuracy_map = trial_data.distance_accuracy_map
    dist_threshold = trial_data.distance_threshold

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
        # noinspection PyTypeChecker
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
                cycle_swap_distances.append(np.mean(dists_data))
                cycle_swap_expected_distances.append(np.mean(dists_actual))
            else:
                partial_cycle_swaps += 1
                partial_cycle_swap_distances.append(np.mean(dists_data))
                partial_cycle_swap_expected_distances.append(np.mean(dists_actual))

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
                np.nanmean(true_swap_distances), np.nanmean(true_swap_expected_distances),
                np.nanmean(partial_swap_distances), np.nanmean(partial_swap_expected_distances),
                np.nanmean(cycle_swap_distances), np.nanmean(cycle_swap_expected_distances),
                np.nanmean(partial_cycle_swap_distances), np.nanmean(partial_cycle_swap_expected_distances))


def deanonymize(participant_data):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "deanonymize")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points

    min_coordinates = []
    min_scores = []
    min_score_positions = []
    raw_deanonymized_misplacements = []
    for actual_trial, data_trial in zip(actual_points, data_points):
        min_score = np.inf
        min_score_idx = -1
        min_permutation = data_trial
        idx = 0
        for perm in itertools.permutations(data_trial):
            score = minimization_function(perm, actual_trial)
            if score < min_score:
                min_score = score
                min_score_idx = idx
                min_permutation = perm
                if score == 0:
                    break
            idx += 1
        min_score_positions.append(min_score_idx)
        min_scores.append(min_score)
        min_coordinates.append(min_permutation)
        raw_deanonymized_misplacements.append(min_score / len(actual_trial))
    return min_coordinates, min_scores, min_score_positions, raw_deanonymized_misplacements


# This function is the main pipeline for the new processing methods. When run alone, it just returns the values
# for a single trial. With visualize=True it will display the results. debug_labels is used to help specify
# which participant/trial is being observed when running from an external process (it is appended to the debug info).
# The coordinates are expected to be equal in length of the for (Nt, Ni, 2) where Nt is the number of trials and Ni is
# the number of items.
# noinspection PyDefaultArgument
def full_pipeline(participant_data, analysis_configuration, visualize=False):
    tools.validate_type(participant_data, type(ParticipantData), "participant_data", "full_pipeline")
    tools.validate_type(analysis_configuration, type(AnalysisConfiguration), "analysis_configuration", "full_pipeline")

    actual_coordinates = participant_data.actual_points
    data_coordinates = participant_data.data_points
    min_coordinates = participant_data.data_points  # For visualization
    transformed_coordinates = participant_data.data_points  # For visualization

    debug_labels = analysis_configuration.debug_labels
    accuracy_z_value = analysis_configuration.z_value
    flags = analysis_configuration.flags

    num_trials = len(participant_data.trials)

    straight_misplacements = []
    axis_swaps = []
    axis_swap_pairs = []
    edge_resize = []
    edge_distort = []
    for (actual_trial, data_trial) in zip(actual_coordinates, data_coordinates):
        # First calculate the two primary original metrics, misplacement and axis swaps - this has been validated
        # against the previous script via an MRE data set of 20 individuals
        straight_misplacements.append(minimization_function(actual_trial, data_trial) / len(actual_trial))
        axis_swaps_element, axis_swap_pairs_element = axis_swap(participant_data)
        axis_swaps.append(axis_swaps_element)
        axis_swap_pairs.append(axis_swap_pairs_element)
        edge_resize.append(edge_resizing(participant_data))
        edge_distort.append(edge_distortion(participant_data))

    pre_processed_accuracy = accuracy(participant_data, analysis_configuration)
    pre_process_accuracies = pre_processed_accuracy.distance_accuracy_map
    pre_process_threshold = pre_processed_accuracy .distance_threshold

    # De-anonymization via Global Minimization of Misplacement
    # Try all permutations of the data coordinates to find an ordering which is globally minimal in misplacement
    if flags == PipelineFlags.Deanonymize:
        min_coordinates, min_score, min_score_position, raw_deanonymized_misplacement = deanonymize(participant_data)
        participant_data.data_points = min_coordinates
        deanon_processed_accuracy = accuracy(participant_data, analysis_configuration)
        deanon_accuracies = deanon_processed_accuracy.distance_accuracy_map
        deanon_threshold = deanon_processed_accuracy.distance_threshold
        # noinspection PyTypeChecker
        deanonymized_labels = [list(itertools.permutations(
            range(0, len(participant_data.actual_points[0]))))[position] for position in min_score_position]
        participant_data.data_labels = deanonymized_labels
        actual_labels = [range(len(actual_trial)) for actual_trial in
                         actual_coordinates]
        participant_data.actual_labels = actual_labels
    else:
        deanon_threshold = [np.nan] * num_trials
        deanon_accuracies = [] * num_trials
        raw_deanonymized_misplacement = [np.nan] * num_trials
        deanonymized_labels = [list(itertools.permutations(
            range(0, len(participant_data.actual_points[0]))))[position] for position in [0] * num_trials]
        participant_data.data_labels = deanonymized_labels
        actual_labels = [range(len(actual_trial)) for actual_trial in
                         actual_coordinates]
        participant_data.actual_labels = actual_labels

    if flags == PipelineFlags.GlobalTransformation:
        (translation, translation_magnitude, scaling, rotation_theta,
         transformation_auto_exclusion, num_geometric_transform_points_excluded,
         transformed_coordinates, geo_dist_threshold) = geometric_transform(participant_data, analysis_configuration)
        participant_data.data_points = transformed_coordinates
    else:
        translation = [[np.nan, np.nan]] * num_trials
        transformation_auto_exclusion = [np.nan] * num_trials
        num_geometric_transform_points_excluded = [np.nan] * num_trials
        rotation_theta = [np.nan] * num_trials
        scaling = [np.nan] * num_trials
        translation_magnitude = [np.nan] * num_trials
        geo_dist_threshold = [np.nan] * num_trials

    # Determine if the points meet the specified accuracy threshold

    (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
     components, misassignment, accurate_misassignment, inaccurate_misassignment, swap_dist_threshold,
     true_swap_distances, true_swap_expected_distances,
     partial_swap_distances, partial_swap_expected_distances,
     cycle_swap_distances, cycle_swap_expected_distances,
     partial_cycle_swap_distances, partial_cycle_swap_expected_distances) = swaps(participant_data,
                                                                                  analysis_configuration)

    output = np.transpose(
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
         [list(map(list, x)) for x in components]
         ])

    # If requested, visualize the data
    if visualize:
        for idx, (actual_trial, data_trial, min_trial, transformed_trial, output_trial) in \
                enumerate(zip(actual_coordinates, data_coordinates, min_coordinates, transformed_coordinates, output)):
            # noinspection PyTypeChecker
            vis.visualization(actual_trial, data_trial, min_trial, transformed_trial, output_trial,
                              z_value=accuracy_z_value, debug_labels=debug_labels + [idx])

    return output


# This function is responsible for returning the names of the values returned in full_pipeline
def get_header_labels():
    return ["Original Misplacement", "Original Swap", "Original Edge Resizing", "Original Edge Distortion",  # 0
            "Axis Swap Pairs", "Pre-Processed Accurate Placements", "Pre-Processed Inaccurate Placements",  # 1
            "Pre-Processed Accuracy Threshold", "Deanonymized Accurate Placements",  # 2
            "Deanonymized Inaccurate Placements", "Deanonymized Accuracy Threshold",  # 3
            "Raw Deanonymized Misplacement", "Transformation Auto-Exclusion",  # 4
            "Number of Points Excluded From Geometric Transform", "Rotation Theta", "Scaling",  # 5
            "Translation Magnitude",  # 6
            "Translation", "Geometric Distance Threshold",  # 7
            "Number of Components", "Accurate Placements", "Inaccurate Placements", "True Swaps",  # 8
            "Partial Swaps", "Cycle Swaps", "Partial Cycle Swaps", "Misassignment", "Accurate Misassignment",  # 9
            "Inaccurate Misassignment", "Swap Distance Threshold",  # 10
            "True Swap Data Distance", "True Swap Actual Distance", "Partial Swap Data Distance",  # 11
            "Partial Swap Actual Distance", "Cycle Swap Data Distance", "Cycle Swap Actual Distance",  # 12
            "Partial Cycle Swap Data Distance", "Partial Cycle Swap Actual Distance",  # 13
            "Unique Components"]  # 14


# (lambda x: list(array(x).flatten())) for append
def get_aggregation_functions():
    return [np.nanmean, np.nanmean, np.nanmean, np.nanmean,  # 0
            tools.collapse_unique_components, np.nanmean, np.nanmean,  # 1
            np.nanmean, np.nanmean,  # 2
            np.nanmean, np.nanmean,  # 3
            np.nanmean, np.nansum,  # 4
            np.nansum, np.nanmean, np.nanmean,  # 5
            np.nanmean,  # 6
            (lambda xs: [np.nanmean(x) for x in np.transpose(xs)]), np.nanmean,  # Mean of vectors # 7
            np.nanmean, np.nanmean, np.nanmean, np.nanmean,  # 8
            np.nanmean, np.nanmean, np.nanmean, np.nanmean, np.nanmean,  # 9
            np.nanmean, np.nanmean,  # 10
            np.nanmean, np.nanmean, np.nanmean,  # 11
            np.nanmean, np.nanmean, np.nanmean,  # 12
            np.nanmean, np.nanmean,  # 13
            tools.collapse_unique_components]  # 14


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
        actual = io.get_coordinates_from_file(args.actual_coordinates,
                                              (args.num_trials, args.num_items, args.dimension))
        data = io.get_coordinates_from_file(args.data_coordinates, (args.num_trials, args.num_items, args.dimension))
        _analysis_configuration = AnalysisConfiguration(z_value=args.accuracy_z_value,
                                                        flags=PipelineFlags(args.pipeline_mode),
                                                        debug_labels=[io.get_id_from_file_prefix(args.data_coordinates),
                                                                      args.line_number])
        _participant_data = ParticipantData([TrialData(_a, _d) for _a, _d in zip([actual[args.line_number]],
                                                                                 [data[args.line_number]])])
        full_pipeline(_participant_data, _analysis_configuration, visualize=True)
        exit()

    logging.info("No arguments found - assuming running in test mode.")

    # Test code
    '''
    a, b = generate_random_test_points(dimension=2, noise=0.2)
    full_pipeline(a, b, visualize=True)
    exit()
    '''
    root_dir = r"Z:\Kevin\iPosition\Hillary\MRE\\"
    actual = io.get_coordinates_from_file(root_dir + r"actual_coordinates.txt", (15, 5, 2))
    data101 = io.get_coordinates_from_file(root_dir + r"101\101position_data_coordinates.txt", (15, 5, 2))
    data104 = io.get_coordinates_from_file(root_dir + r"104\104position_data_coordinates.txt", (15, 5, 2))
    data105 = io.get_coordinates_from_file(root_dir + r"105\105position_data_coordinates.txt", (15, 5, 2))
    data112 = io.get_coordinates_from_file(root_dir + r"112\112position_data_coordinates.txt", (15, 5, 2))
    data113 = io.get_coordinates_from_file(root_dir + r"113\113position_data_coordinates.txt", (15, 5, 2))
    data114 = io.get_coordinates_from_file(root_dir + r"114\114position_data_coordinates.txt", (15, 5, 2))
    data118 = io.get_coordinates_from_file(root_dir + r"118\118position_data_coordinates.txt", (15, 5, 2))
    data119 = io.get_coordinates_from_file(root_dir + r"119\119position_data_coordinates.txt", (15, 5, 2))
    data120 = io.get_coordinates_from_file(root_dir + r"120\120position_data_coordinates.txt", (15, 5, 2))

    # Cycle Agree
    full_pipeline(ParticipantData([TrialData(actual[10], data101[10])]),
                  AnalysisConfiguration(debug_labels=["101", 10]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[12], data104[12])]),
                  AnalysisConfiguration(debug_labels=["104", 12]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[2], data105[2])]),
                  AnalysisConfiguration(debug_labels=["105", 2]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[6], data112[6])]),
                  AnalysisConfiguration(debug_labels=["112", 6]), visualize=True)

    # Old Swap, New Cycle (only truly debatable one in my opinion)
    full_pipeline(ParticipantData([TrialData(actual[2], data104[2])]),
                  AnalysisConfiguration(debug_labels=["104", 2]), visualize=True)

    # New Single Swap
    full_pipeline(ParticipantData([TrialData(actual[0], data101[0])]),
                  AnalysisConfiguration(debug_labels=["101", 0]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[12], data114[12])]),
                  AnalysisConfiguration(debug_labels=["114", 12]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[10], data118[10])]),
                  AnalysisConfiguration(debug_labels=["118", 10]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[10], data119[10])]),
                  AnalysisConfiguration(debug_labels=["119", 10]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[14], data120[14])]),
                  AnalysisConfiguration(debug_labels=["120", 14]), visualize=True)

    # False Alarms (one or more old swap where new disagrees)
    full_pipeline(ParticipantData([TrialData(actual[11], data101[11])]),
                  AnalysisConfiguration(debug_labels=["101", 11]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[10], data104[10])]),
                  AnalysisConfiguration(debug_labels=["104", 10]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[2], data113[2])]),
                  AnalysisConfiguration(debug_labels=["113", 2]), visualize=True)
    full_pipeline(ParticipantData([TrialData(actual[12], data120[12])]),
                  AnalysisConfiguration(debug_labels=["120", 12]), visualize=True)
