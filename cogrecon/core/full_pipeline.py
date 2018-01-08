import copy
import itertools
import logging
import warnings

import networkx as nx
import numpy as np
from scipy.spatial import distance

from .visualization.vis_iposition import visualization
from .cogrecon_globals import default_z_value
from .data_structures import TrialData, ParticipantData, AnalysisConfiguration, PipelineFlags
from .similarity_transform import similarity_transform
from .tools import validate_type, mask_points, collapse_unique_components, \
    find_minimal_mapping, greedy_find_minimal_mapping, sum_of_distance, permutation_from_index


def accuracy(participant_data, analysis_configuration, use_manual_threshold=False):
    """
    This function computes the accuracy of each item in a ParticipantData object, optionally using a manual threshold.

    :param participant_data: the ParticipantData object
    :param analysis_configuration: the AnalysisConfiguration object
    :param use_manual_threshold: if True, the manual threshold from the AnalysisConfiguration object will be used
                                 instead of an automatically computed threshold

    :return: the input ParticipantData object with the distance_accuracy_map and distance_threshold attributes populated
    """
    validate_type(participant_data, ParticipantData, "participant_data", "accuracy")
    validate_type(analysis_configuration, AnalysisConfiguration, "analysis_configuration", "accuracy")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points
    z_value = analysis_configuration.z_value
    trial_by_trial_accuracy = analysis_configuration.trial_by_trial_accuracy
    if use_manual_threshold:
        manual_threshold = analysis_configuration.manual_threshold
    else:
        manual_threshold = None

    if z_value is None:
        logging.error('a z_value was not found for accuracy, using z={0}'.format(default_z_value))
        z_value = default_z_value
    if trial_by_trial_accuracy:
        dist_accuracy_map = []
        exclusion_thresholds = []
        for actual_trial, data_trial in zip(actual_points, data_points):
            if manual_threshold is not None:
                if len(data_trial) == 0 or len(actual_trial) == 0:
                    dists = []
                else:
                    dists = np.diag(distance.cdist(data_trial, actual_trial))
                dist_accuracy_map.append([xd < manual_threshold for xd in dists])
                exclusion_thresholds.append(manual_threshold)
            else:
                if len(data_trial) == 0 or len(actual_trial) == 0:
                    dists = []
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
                if len(data_trial) == 0 or len(actual_trial) == 0:
                    dists = []
                else:
                    dists = np.diag(distance.cdist(data_trial, actual_trial))
                dist_accuracy_map.append([xd < manual_threshold for xd in dists])
            exclusion_thresholds = [manual_threshold] * len(actual_points)
        else:
            collapsed_actual_points = np.array([x for sublist in actual_points for x in sublist])
            # collapsed_actual_points = np.reshape(np.array(actual_points), (-1, len(actual_points[0][0])))
            data_points = [list(xd) for xd in data_points]
            # collapsed_data_points = np.array(data_points).reshape(-1, len(data_points[0][0]))
            collapsed_data_points = np.array([x for sublist in data_points for x in sublist])
            if len(collapsed_actual_points) == 0 or len(collapsed_data_points) == 0:
                dists = []
            else:
                dists = np.diag(distance.cdist(collapsed_actual_points, collapsed_data_points))
            mu = np.mean(dists)
            sd = np.std(dists)
            ste = sd / np.sqrt(len(dists))
            ci = ste * z_value
            exclusion_threshold = ci + mu
            exclusion_thresholds = [exclusion_threshold] * len(actual_points)

            for actual_trial, data_trial in zip(actual_points, data_points):
                if len(data_trial) == 0 or len(actual_trial) == 0:
                    dists = []
                else:
                    # noinspection PyTypeChecker
                    dists = np.diag(distance.cdist(data_trial, actual_trial))
                dist_accuracy_map.append([bool(xd < exclusion_threshold) for xd in dists])

    participant_data.distance_accuracy_map = dist_accuracy_map
    participant_data.distance_threshold = exclusion_thresholds

    return participant_data


def trial_axis_swap(trial_data):
    """
    This function calculates the original axis swap metrics on a TrialData object.

    :param trial_data: a TrialData object
    :return: the axis swap value
    """
    validate_type(trial_data, TrialData, "trial_data", "trial_axis_swap")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points
    actual_labels = trial_data.actual_labels
    data_labels = trial_data.data_labels

    if len(data_points) < 2:
        return 0, []

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


def trial_edge_resizing(trial_data):
    """
    This function calculates the original edge resizing metrics on a TrialData object.

    :param trial_data: a TrialData object
    :return: the edge resizing value
    """
    validate_type(trial_data, TrialData, "trial_data", "trial_edge_resizing")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points

    actual_edges = []
    data_edges = []
    for idx1 in range(len(actual_points)):
        for idx2 in range(idx1, len(actual_points)):
            actual_edges.append(distance.euclidean(actual_points[idx1], actual_points[idx2]))
            data_edges.append(distance.euclidean(data_points[idx1], data_points[idx2]))
    # noinspection PyTypeChecker
    resizing = np.mean(abs(np.array(actual_edges) - np.array(data_edges)))

    return resizing


def trial_edge_distortion(trial_data):
    """
    This function calculates the original edge distortion metrics on a TrialData object.

    :param trial_data: a TrialData object

    :return: the edge distortion value
    """
    validate_type(trial_data, TrialData, "trial_data", "trial_edge_distortion")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points

    if len(data_points) < 2:
        return 0, []

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
    """
    This function performs the geometric transform computation on an entire ParticipantData object.

    :param participant_data: the ParticipantData object
    :param analysis_configuration: the AnalysisConfiguration object

    :return: a list of the geometric transform results from trial_geometric_transform for each trial in the
             ParticipantData object
    """
    validate_type(participant_data, ParticipantData, "participant_data", "geometric_transform")
    validate_type(analysis_configuration, AnalysisConfiguration, "analysis_configuration", "geometric_transform")

    # Determine if the points meet the specified accuracy threshold
    participant_data = accuracy(participant_data, analysis_configuration)
    result = []
    for trial in participant_data.trials:
        # noinspection PyTypeChecker
        result.append(trial_geometric_transform(trial, analysis_configuration))

    return np.transpose(result)


# noinspection PyDefaultArgument
def trial_geometric_transform(trial_data, analysis_configuration):
    # TODO: Addition transformation/de-anonymization methods(see https://en.wikipedia.org/wiki/Point_set_registration)
    """
    This function calculates the geometric transform metrics on a TrialData object.

    :param trial_data: a TrialData object
    :param analysis_configuration: the AnalysisConfiguration object

    :return: a tuple containing: translation, translation_magnitude, scaling, rotation_theta,
             transformation_auto_exclusion, num_geometric_transform_points_excluded, transformed_coordinates,
             dist_threshold
    """
    validate_type(trial_data, TrialData, "trial_data", "trial_geometric_transform")
    validate_type(analysis_configuration, AnalysisConfiguration, "analysis_configuration", "trial_geometric_transform")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points
    dist_accuracy_map = trial_data.distance_accuracy_map
    dist_threshold = trial_data.distance_threshold
    debug_labels = analysis_configuration.debug_labels

    # Determine which points should be included in the transformation step and generate the point sets
    valid_points_indicies = [x for (x, y) in zip(range(len(actual_points)), dist_accuracy_map) if y]
    from_points = mask_points(data_points, valid_points_indicies)
    to_points = mask_points(actual_points, valid_points_indicies)

    # noinspection PyTypeChecker
    # Number of "inaccurate" points after deanonymizing is number of False in this list
    num_geometric_transform_points_excluded = dist_accuracy_map.count(False)
    # noinspection PyTypeChecker
    num_geometric_transform_points_used = dist_accuracy_map.count(True)

    translation_magnitude = np.nan
    rotation_theta = np.nan
    scaling = np.nan
    transformation_auto_exclusion = True
    translation = [np.nan] * len(actual_points[0])
    transformed_coordinates = np.array(data_points, copy=True).tolist()
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
            transformed_coordinates = np.array([(np.array(x) + np.array(translation)).dot(rotation_matrix) * scaling
                                               for x in data_points]).tolist()
            transformation_auto_exclusion = False
            new_error = sum_of_distance(transformed_coordinates, actual_points)
            old_error = sum_of_distance(data_points, actual_points)
            if new_error > old_error:  # Exclude rotation from transform
                rotation_theta = np.nan
                logging.info(str(debug_labels) + " : " +
                             ('The transformation function did not reduce the error, removing rotation and retying' +
                              ' (old_error={0}, new_error={1}).').format(old_error,
                                                                         new_error))
                transformed_coordinates = np.array([(np.array(x) + np.array(translation)) * scaling
                                                    for x in data_points]).tolist()
                new_error = sum_of_distance(transformed_coordinates, actual_points)
                old_error = sum_of_distance(data_points, actual_points)
                if new_error > old_error:  # Completely exclude transform
                    transformation_auto_exclusion = True
                    rotation_theta = np.nan
                    scaling = np.nan
                    translation_magnitude = np.nan
                    translation = [np.nan] * len(actual_points[0])
                    transformed_coordinates = np.array(data_points, copy=True).tolist()
                    logging.warning(str(debug_labels) + " : " +
                                    ('The transformation function did not reduce the error, removing transform ' +
                                     '(old_error={0}, new_error={1}).').format(old_error,
                                                                               new_error))

        except ValueError:
            transformed_coordinates = np.array(data_points, copy=True).tolist()
            logging.error(('Finding transformation failed , ' +
                           'from_points={0}, to_points={1}.').format(from_points, to_points))
    return (translation, translation_magnitude, scaling, rotation_theta, transformation_auto_exclusion,
            num_geometric_transform_points_excluded, transformed_coordinates, dist_threshold)


def swaps(participant_data, analysis_configuration):
    """
    This function computes the swap metrics across all trials in a ParticipantData object.

    :param participant_data: the ParticipantData object
    :param analysis_configuration: the AnalysisConfiguration object

    :return: a list of results from the trial_swaps function for all trials in ParticipantData in order
    """
    validate_type(participant_data, ParticipantData, "participant_data", "swaps")
    validate_type(analysis_configuration, AnalysisConfiguration, "analysis_configuration", "swaps")

    participant_data = accuracy(participant_data, analysis_configuration, use_manual_threshold=True)

    result = []
    for trial in participant_data.trials:
        result.append(list(trial_swaps(trial)))

    return np.transpose(np.array(result, dtype=object)).tolist()


def trial_swaps(trial_data):
    """
    This function calculates the swap metrics on a particular TrialData object.

    :param trial_data: a TrialData object
    :return: a tuple containing: accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps,
             partial_cycle_swaps, components, misassignment, accurate_misassignment, inaccurate_misassignment,
             dist_threshold, mean true_swap_distances, mean true_swap_expected_distances,
             mean partial_swap_distances, mean partial_swap_expected_distances,
             mean cycle_swap_distances, mean cycle_swap_expected_distances,
             mean partial_cycle_swap_distances, mean partial_cycle_swap_expected_distances
    """
    validate_type(trial_data, TrialData, "trial_data", "trial_swaps")

    actual_points = trial_data.actual_points
    data_points = trial_data.data_points
    actual_labels = trial_data.actual_labels
    data_labels = trial_data.data_labels
    dist_accuracy_map = trial_data.distance_accuracy_map
    dist_threshold = trial_data.distance_threshold

    accurate_points_labels = [l for (l, is_accurate) in zip(actual_labels, dist_accuracy_map) if is_accurate]

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

    accurate_misassignment_pairs = []
    inaccurate_misassignment_pairs = []
    misassignment = 0
    accurate_misassignment = 0
    inaccurate_misassignment = 0
    for actual_label, data_label, acc in zip(actual_labels, data_labels, dist_accuracy_map):
        if actual_label != data_label:
            misassignment += 1
            if acc:
                accurate_misassignment_pairs.append([actual_label, data_label])
                accurate_misassignment += 1
            else:
                inaccurate_misassignment_pairs.append([actual_label, data_label])
                inaccurate_misassignment += 1

    with warnings.catch_warnings():  # Ignore empty mean warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
                components, misassignment, accurate_misassignment, inaccurate_misassignment, dist_threshold,
                np.nanmean(true_swap_distances), np.nanmean(true_swap_expected_distances),
                np.nanmean(partial_swap_distances), np.nanmean(partial_swap_expected_distances),
                np.nanmean(cycle_swap_distances), np.nanmean(cycle_swap_expected_distances),
                np.nanmean(partial_cycle_swap_distances), np.nanmean(partial_cycle_swap_expected_distances),
                accurate_misassignment_pairs, inaccurate_misassignment_pairs)


def deanonymize(participant_data, analysis_configuration):
    """
    This function performs the deanonymization routine on ParticipantData.

    :param participant_data: the ParticipantData object
    :param analysis_configuration: the AnalysisConfiguration object

    :return: the minimal mapping coordinates, minimal distance scores, minimal score lexicographical positions, and
             misplacement values for each trial
    """
    validate_type(participant_data, ParticipantData, "participant_data", "deanonymize")
    validate_type(analysis_configuration, AnalysisConfiguration, "analysis_configuration", "deanonymize")

    actual_points = participant_data.actual_points
    data_points = participant_data.data_points
    data_order = participant_data.data_order

    min_coordinates = []
    min_scores = []
    min_score_positions = []
    raw_deanonymized_misplacements = []

    for actual_trial, data_trial, order_trial in zip(actual_points, data_points, data_order):

        if analysis_configuration.greedy_order_deanonymization:
            # First we shrink the order to unique ordered ints between 0 and len(order_trial) rather than their
            # potentially non-contiguous, ordered ints
            indices = list(range(len(order_trial)))
            indices.sort(key=order_trial.__getitem__)
            shrunk_order = [0 for _ in range(len(order_trial))]
            for i, n in zip(indices, range(len(indices))):
                shrunk_order[i] = n
            min_score, min_score_idx, min_permutation = greedy_find_minimal_mapping(actual_trial,
                                                                                    data_trial, shrunk_order)
        else:
            min_score, min_score_idx, min_permutation = find_minimal_mapping(actual_trial, data_trial)

        min_score_positions.append(min_score_idx)
        min_scores.append(min_score)
        min_coordinates.append(list(min_permutation))
        raw_deanonymized_misplacements.append(min_score / len(actual_trial))
    return min_coordinates, min_scores, min_score_positions, raw_deanonymized_misplacements


# noinspection PyDefaultArgument
def full_pipeline(participant_data, analysis_configuration, visualize=False, visualization_extent=None, fig_size=None):
    """
    This function is the main pipeline for the new processing methods. When run alone, it just returns the values
    for a single trial. With visualize=True it will display the results. debug_labels is used to help specify
    which participant/trial is being observed when running from an external process (it is appended to the debug info).
    The coordinates are expected to be equal in length of the for (Nt, Ni, 2) where Nt is the number of trials and Ni is
    the number of items.


    :param participant_data: the ParticipantData object to process
    :param analysis_configuration: the AnalysisConfiguration object to determine the pipeline configuration
    :param visualize: if True, the output will be visualized
    :param visualization_extent: the data bounds to be used for visualization
    :param fig_size: the size of the output figure in visualization (in inches)

    :return: an output list of all metrics produces in the pipeline whose headers are defined by get_header_labels
    """
    validate_type(participant_data, ParticipantData, "participant_data", "full_pipeline")
    validate_type(analysis_configuration, AnalysisConfiguration, "analysis_configuration", "full_pipeline")

    participant_data.trials = [t for idx, t in enumerate(participant_data.trials)
                               if not (len(t.actual_points) == 0 or len(t.data_points) == 0)]

    if len(participant_data.trials) == 0:
        return None

    original_participant_data = copy.deepcopy(participant_data)

    if analysis_configuration.category_independence:
        if visualize:
            logging.warning('Visualization is not supported for category data at the moment.')
        categories = participant_data.category_labels
        unique_categories = np.unique(categories).tolist()
        _category_participant_data, _unknown_category_participant_data = \
            ParticipantData.category_split_participant(participant_data, unique_categories)
        _category_analysis_configuration = copy.deepcopy(analysis_configuration)
        _category_analysis_configuration.category_independence = False
        _category_analysis_configuration.is_category = True

        cat_outputs = []
        for cat, cat_label in zip(_category_participant_data, unique_categories):
            _category_analysis_configuration.category_label = cat_label
            cat_output = full_pipeline(cat, _category_analysis_configuration)
            if cat_output is not None:
                cat_outputs.append(cat_output)

        _category_analysis_configuration.category_label = 'unknown'
        unk_output = full_pipeline(_unknown_category_participant_data, _category_analysis_configuration)
        if unk_output is not None:
            cat_outputs.append(unk_output)

        return cat_outputs

    actual_coordinates = participant_data.actual_points
    min_coordinates = participant_data.data_points  # For visualization
    transformed_coordinates = participant_data.data_points  # For visualization

    flags = analysis_configuration.flags

    num_trials = len(participant_data.trials)
    num_items = len(participant_data.trials[0].actual_points)
    num_dimensions = len(participant_data.trials[0].actual_points[0])

    straight_misplacements = []
    axis_swaps = []
    axis_swap_pairs = []
    edge_resize = []
    edge_distort = []
    for trial in participant_data.trials:
        # First calculate the two primary original metrics, misplacement and axis swaps - this has been validated
        # against the previous script via an MRE data set of 20 individuals
        if len(trial.data_points) == 0 or len(trial.actual_points) == 0:
            avg_misplacement = -1
        else:
            avg_misplacement = sum_of_distance(trial.actual_points, trial.data_points) / len(trial.actual_points)
        straight_misplacements.append(avg_misplacement)
        axis_swaps_element, axis_swap_pairs_element = trial_axis_swap(trial)
        axis_swaps.append(axis_swaps_element)
        axis_swap_pairs.append(axis_swap_pairs_element)
        edge_resize.append(trial_edge_resizing(trial))
        edge_distort.append(trial_edge_distortion(trial))

    pre_processed_accuracy = accuracy(participant_data, analysis_configuration)
    pre_process_accuracies = pre_processed_accuracy.distance_accuracy_map
    pre_process_threshold = pre_processed_accuracy .distance_threshold

    # De-anonymization via Global Minimization of Misplacement
    # Try all permutations of the data coordinates to find an ordering which is globally minimal in misplacement
    if flags == PipelineFlags.Deanonymize:
        min_coordinates, min_score, min_score_position, \
            raw_deanonymized_misplacement = deanonymize(participant_data, analysis_configuration)
        participant_data.data_points = min_coordinates
        deanon_processed_accuracy = accuracy(participant_data, analysis_configuration)
        deanon_accuracies = deanon_processed_accuracy.distance_accuracy_map
        deanon_threshold = deanon_processed_accuracy.distance_threshold
        # noinspection PyTypeChecker
        deanonymized_labels = [permutation_from_index(
            list(range(0, len(participant_data.actual_points[idx]))), position)
            for idx, position in enumerate(min_score_position)]
        participant_data.data_labels = deanonymized_labels
        actual_labels = [range(len(actual_trial)) for actual_trial in
                         actual_coordinates]
        participant_data.actual_labels = actual_labels
    else:
        deanon_threshold = [np.nan] * num_trials
        deanon_accuracies = [[]] * num_trials
        raw_deanonymized_misplacement = [np.nan] * num_trials
        deanonymized_labels = [permutation_from_index(list(range(0, len(participant_data.actual_points[0]))), position)
                               for position in [0] * num_trials]
        deanonymized_labels = [list(_x) for _x in deanonymized_labels]
        participant_data.data_labels = deanonymized_labels
        actual_labels = [range(len(actual_trial)) for actual_trial in
                         actual_coordinates]
        participant_data.actual_labels = actual_labels

    post_deanonymized_misplacement = []
    for trial in participant_data.trials:
        post_deanonymized_misplacement.append(
            sum_of_distance(trial.actual_points, trial.data_points) / len(trial.actual_points))

    if flags == PipelineFlags.GlobalTransformation:
        (translation, translation_magnitude, scaling, rotation_theta,
         transformation_auto_exclusion, num_geometric_transform_points_excluded,
         transformed_coordinates, geo_dist_threshold) = geometric_transform(participant_data, analysis_configuration)
        participant_data.data_points = transformed_coordinates
    else:
        translation = [[np.nan] * num_dimensions] * num_trials
        transformation_auto_exclusion = [np.nan] * num_trials
        num_geometric_transform_points_excluded = [np.nan] * num_trials
        rotation_theta = [np.nan] * num_trials
        scaling = [np.nan] * num_trials
        translation_magnitude = [np.nan] * num_trials
        geo_dist_threshold = [np.nan] * num_trials

    post_transform_misplacement = []
    for trial in participant_data.trials:
        post_transform_misplacement.append(sum_of_distance(trial.actual_points,
                                                           trial.data_points) / len(trial.actual_points))

    # Determine if the points meet the specified accuracy threshold

    (accurate_placements, inaccurate_placements, true_swaps, partial_swaps, cycle_swaps, partial_cycle_swaps,
     components, misassignment, accurate_misassignment, inaccurate_misassignment, swap_dist_threshold,
     true_swap_distances, true_swap_expected_distances,
     partial_swap_distances, partial_swap_expected_distances,
     cycle_swap_distances, cycle_swap_expected_distances,
     partial_cycle_swap_distances, partial_cycle_swap_expected_distances,
     accurate_misassignment_pairs, inaccurate_miassignment_pairs) = swaps(participant_data,
                                                                          analysis_configuration)

    # TODO: This is a patch fix, but it doesn't address the problem. For some reason, edge_distort occasionally returns
    # (0, []). I have no idea where it's coming from, but replacing it with nan is a stopgap as that metric isn't used.
    if isinstance(edge_distort, list) and isinstance(edge_distort[0], list) and (0, []) in edge_distort:
        edge_distort = [[_x if _x is not (0, []) else np.nan for _x in sublist] for sublist in edge_distort]

    # This is a bit of a temporary fix for the fact that the rest of the package can work in arbitrary dimensions, but
    # a common use case is to do 1D/2D translation measurement. I'm not prepared to make a completely variable number
    # of columns based on dimensionality...
    if len(translation[0]) == 1:
        x_translation = [x[0] for x in translation]
        y_translation = [np.nan for _ in translation]
    elif len(translation[0]) == 2:
        try:
            x_translation = [x for x, y in translation]
            y_translation = [y for x, y in translation]
        except ValueError:
            x_translation = [np.nan for _ in translation]
            y_translation = [np.nan for _ in translation]
    else:
        x_translation = [np.nan for _ in translation]
        y_translation = [np.nan for _ in translation]

    output = \
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
         post_deanonymized_misplacement,
         transformation_auto_exclusion,
         num_geometric_transform_points_excluded,
         rotation_theta,
         scaling,
         translation_magnitude,
         translation,
         x_translation,
         y_translation,
         geo_dist_threshold,
         post_transform_misplacement,
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
         [list(map(list, x)) for x in components],
         [analysis_configuration.is_category] * len(participant_data.trials),
         [analysis_configuration.category_label] * len(participant_data.trials),
         [list(map(list, x)) for x in accurate_misassignment_pairs],
         [list(map(list, x)) for x in inaccurate_miassignment_pairs]
         ]

    output = np.transpose(np.array(output, dtype=object))
    # If requested, visualize the data
    if visualize:
        for idx, (trial, min_trial, transformed_trial, output_trial, pre_process_threshold_trial,
                  swap_dist_threshold_trial, pre_processed_accuracy_trial, post_distance_accuracy_map_trial) in \
                enumerate(zip(original_participant_data.trials, min_coordinates, transformed_coordinates, output,
                              pre_process_threshold, swap_dist_threshold, pre_processed_accuracy.distance_accuracy_map,
                              participant_data.distance_accuracy_map)):
            # noinspection PyTypeChecker
            visualization(trial, analysis_configuration, min_trial, transformed_trial, output_trial,
                          pre_process_threshold_trial, swap_dist_threshold_trial,
                          pre_processed_accuracy_trial, post_distance_accuracy_map_trial,
                          extent=visualization_extent, fig_size=fig_size)

    return output


def get_header_labels():
    """
    This function is responsible for returning the names of the values returned in full_pipeline

    :return: a list of header names for each output column
    """
    return ["Original Misplacement", "Original Swap", "Original Edge Resizing", "Original Edge Distortion",  # 0
            "Axis Swap Pairs", "Pre-Processed Accurate Placements", "Pre-Processed Inaccurate Placements",  # 1
            "Pre-Processed Accuracy Threshold", "Deanonymized Accurate Placements",  # 2
            "Deanonymized Inaccurate Placements", "Deanonymized Accuracy Threshold",  # 3
            "Raw Deanonymized Misplacement", "Post-Deanonymized Misplacement", "Transformation Auto-Exclusion",  # 4
            "Number of Points Excluded From Geometric Transform", "Rotation Theta", "Scaling",  # 5
            "Translation Magnitude", "Translation", "TranslationX", "TranslationY",  # 6
            "Geometric Distance Threshold", "Post-Transform Misplacement",  # 7
            "Number of Components", "Accurate Single-Item Placements", "Inaccurate Single-Item Placements",  # 8
            "True Swaps",  # 9
            "Partial Swaps", "Cycle Swaps", "Partial Cycle Swaps", "Misassignment", "Accurate Misassignment",  # 10
            "Inaccurate Misassignment", "Swap Distance Threshold",  # 11
            "True Swap Data Distance", "True Swap Actual Distance", "Partial Swap Data Distance",  # 12
            "Partial Swap Actual Distance", "Cycle Swap Data Distance", "Cycle Swap Actual Distance",  # 13
            "Partial Cycle Swap Data Distance", "Partial Cycle Swap Actual Distance",  # 14
            "Unique Components", "Contains Category Data", "Category Label",  # 15
            "Accurate Misassignment Pairs", "Inaccurate Misassignment Pairs"]  # 16


# (lambda x: list(array(x).flatten())) for append
def get_aggregation_functions():
    """
    This function gets a list of functions which should be used when collapsing trial data for each column in the
    output.

    :return: a list of functions to be applied to cross-trial data to collapse it appropriately
    """
    return [np.nanmean, np.nanmean, np.nanmean, np.nanmean,  # 0
            collapse_unique_components, np.nanmean, np.nanmean,  # 1
            np.nanmean, np.nanmean,  # 2
            np.nanmean, np.nanmean,  # 3
            np.nanmean, np.nanmean, np.nansum,  # 4
            np.nansum, np.nanmean, np.nanmean,  # 5
            np.nanmean, (lambda xs: [np.nanmean(x) for x in np.transpose(xs)]), np.nanmean, np.nanmean,  # 6; vector mu
            np.nanmean, np.nanmean,  # 7
            np.nanmean, np.nanmean, np.nanmean,  # 8
            np.nanmean,  # 9
            np.nanmean, np.nanmean, np.nanmean, np.nanmean, np.nanmean,  # 10
            np.nanmean, np.nanmean,  # 11
            np.nanmean, np.nanmean, np.nanmean,  # 12
            np.nanmean, np.nanmean, np.nanmean,  # 13
            np.nanmean, np.nanmean,  # 14
            collapse_unique_components, any, collapse_unique_components,  # 15
            collapse_unique_components, collapse_unique_components]  # 16
