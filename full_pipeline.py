import matplotlib.pyplot as plt
import itertools
from scipy.spatial import distance
from numpy import *
from similarity_transform import similarity_transform
import logging
import matplotlib.animation as animation
import networkx as nx
from enum import Enum


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
def generate_random_test_points(number_of_points=5):
    correct_points = [[random.random(), random.random()] for _ in range(0, number_of_points)]
    offsets = [[(random.random()) / 20.0, (random.random()) / 20.0] for _ in range(0, number_of_points)]
    input_points = [list(first + second for first, second in zip(ta, tb)) for ta, tb in zip(correct_points, offsets)]

    perms = list(itertools.permutations(input_points))
    input_points = perms[random.randint(0, len(perms) - 1)]
    index = random.randint(0, len(input_points))
    input_points[index][0] = random.random()
    input_points[index][1] = random.random()

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


# TODO: For the entire function, it should be made to be reconfigurable, skipping steps as is appropriate
# This function is the main pipeline for the new processing methods. When run alone, it just returns the values
# for a single trial. With visualize=True it will display the results. debug_labels is used to help specify
# which participant/trial is being observed when running from an external process (it is appended to the debug info).
# The coordinates are expected to be equal in length of the for (Nt, Ni, 2) where Nt is the number of trials and Ni is
# the number of items.
def full_pipeline(actual_coordinates, data_coordinates, visualize=False, debug_labels=None,
                  geometric_transform_exclusion_z_value=1.96, flags=PipelineFlags.All):
    # First calculate the two primary original metrics, misplacement and axis swaps - this has been validated against
    # the previous script via an MRE data set of 20 individuals
    straight_misplacements = minimization_function(actual_coordinates, data_coordinates) / len(actual_coordinates)
    axis_swaps = 0
    comparisons = 0
    for idx in range(0, len(actual_coordinates)):
        for idx2 in range(idx + 1, len(actual_coordinates)):
            comparisons += 1
            if all(array(map(sign, array(actual_coordinates[idx]) - array(actual_coordinates[idx2]))) !=
                   array(map(sign, array(data_coordinates[idx]) - array(data_coordinates[idx2])))):
                axis_swaps += 1
    axis_swaps = float(axis_swaps) / float(comparisons)

    # Generate some arbitrary, ordered labels
    labels = range(0, len(actual_coordinates))

    # De-anonymization via Global Minimization of Misplacement
    # Try all permutations of the data coordinates to find an ordering which is globally minimal in misplacement
    if flags == PipelineFlags.Deanonymize:
        perms = list(itertools.permutations(data_coordinates))
        scores = [minimization_function(x, actual_coordinates) for x in perms]
        min_score_position = scores.index(min(scores))
        min_permutation = perms[min_score_position]
        min_coordinates = min_permutation

        # Compute the new misplacement value for the raw, deanonymized values
        raw_deanonymized_misplacement = min(scores) / len(actual_coordinates)  # Standard misplacement
    else:
        min_coordinates = data_coordinates
        raw_deanonymized_misplacement = nan
        min_score_position = 0

    if flags == PipelineFlags.GlobalTransformation:
        # Determine if the points meet the specified accuracy threshold
        dists = diag(distance.cdist(min_coordinates, actual_coordinates))
        mu = mean(dists)
        sd = std(dists)
        ste = sd / sqrt(len(dists))
        ci = ste * geometric_transform_exclusion_z_value
        exclusion_threshold = ci + mu
        dist_accuracy_map = [x < exclusion_threshold for x in dists]

        # Determine which points should be included in the transformation step and generate the point sets
        valid_points_labels = [x for (x, y) in zip(labels, dist_accuracy_map) if y]
        from_points = array([min_coordinates[idx] for idx in valid_points_labels])
        to_points = array([actual_coordinates[idx] for idx in valid_points_labels])

        # noinspection PyTypeChecker
        # Number of "inaccurate" points after deanonymizing is number of False in this list
        num_geometric_transform_points_excluded = dist_accuracy_map.count(False)
        # noinspection PyTypeChecker
        num_geometric_transform_points_used = dist_accuracy_map.count(True)

        rotation_theta = nan
        # rotation_matrix = nan
        scaling = nan
        # translation = nan
        # transformed_coordinates = nan
        translation_magnitude = nan
        transformation_auto_exclusion = False
        # Confirm there are enough points to perform the transformation
        # (it is meaningless to perform with 0 or 1 points)
        if num_geometric_transform_points_used <= 1:
            transformation_auto_exclusion = True
            transformed_coordinates = min_coordinates
            logging.warning(str(debug_labels) + " : " + ('Not enough points were found to be accurate enough to '
                                                         'create a geometric transform. It will be skipped.'))
        else:
            try:
                # Perform the transformation via Umeyama's method
                rotation_matrix, scaling, translation = similarity_transform(from_points, to_points)
                # Compute the rotation factor
                theta_matrix = [[arccos(rotation_matrix[0][0]), arcsin(rotation_matrix[0][1])],
                                [arcsin(rotation_matrix[0][1]), arccos(rotation_matrix[1][1])]]
                theta_matrix = [map(abs, x) for x in theta_matrix]
                rotation_theta = mean(theta_matrix)  # Rotation angle
                translation_magnitude = linalg.norm(
                    translation)  # Translation magnitude (direction is in 'translation' variable)
                # Apply the linear transformation to the data coordinates to cancel out global errors if possible
                transformed_coordinates = [array([ax + bx for (ax, bx) in
                                                  zip(x, translation)]).dot(rotation_matrix) * scaling
                                           for x in data_coordinates]
                if minimization_function(transformed_coordinates, actual_coordinates) > \
                        minimization_function(min_coordinates, actual_coordinates):
                    transformation_auto_exclusion = True
                    transformed_coordinates = min_coordinates
                    logging.warning(str(debug_labels) + " : " + 'The transformation function did not reduce the error.')

            except ValueError:
                transformation_auto_exclusion = True
                transformed_coordinates = min_coordinates
                logging.error(('Finding transformation failed due to colinearility, ' +
                               'from_points={0}, to_points={1}.').format(from_points, to_points))

        logging.debug(str(debug_labels) + " : " + ('Misplacement: {0}, Points Used for Geometric Transform: {1}, ' +
                                                   'Theta={2}, Scaling={3}, Translation Magnitude={4}' +
                                                   '').format(raw_deanonymized_misplacement,
                                                              num_geometric_transform_points_excluded,
                                                              rotation_theta, scaling, translation_magnitude))
    else:
        transformed_coordinates = min_coordinates
        transformation_auto_exclusion = nan
        num_geometric_transform_points_excluded = nan
        rotation_theta = nan
        scaling = nan
        translation_magnitude = nan

    # Determine if the points meet the specified accuracy threshold
    dists = diag(distance.cdist(transformed_coordinates, actual_coordinates))
    mu = mean(dists)
    sd = std(dists)
    ste = sd / sqrt(len(dists))
    ci = ste * geometric_transform_exclusion_z_value
    exclusion_threshold = ci + mu
    dist_accuracy_map = [dist < exclusion_threshold for dist in dists]
    accurate_points_labels = [label for (label, is_accurate) in zip(labels, dist_accuracy_map) if is_accurate]

    actual_labels = range(0, len(actual_coordinates))
    deanonymized_labels = list(itertools.permutations(range(0, len(actual_coordinates))))[min_score_position]

    deanonymized_graph = nx.Graph()
    deanonymized_graph.add_nodes_from(actual_labels)
    deanonymized_graph.add_edges_from(zip(actual_labels, deanonymized_labels))
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

    output = [straight_misplacements,
              axis_swaps,
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
              partial_cycle_swaps
              ]

    # If requested, visualize the data
    if visualize:
        # Generate a figure with 3 scatter plots (actual points, data points, and transformed points)
        fig, ax = plt.subplots()
        x = [float(v) for v in list(transpose(transformed_coordinates)[0])]
        y = [float(v) for v in list(transpose(transformed_coordinates)[1])]
        ax.scatter(x, y, c='b', alpha=0.5)
        scat = ax.scatter(x, y, c='b', animated=True)
        ax.scatter(transpose(actual_coordinates)[0], transpose(actual_coordinates)[1], c='g')
        ax.scatter(transpose(data_coordinates)[0], transpose(data_coordinates)[1], c='r')
        # Label the stationary points (actual and data)
        for idx, xy in enumerate(zip(transpose(actual_coordinates)[0], transpose(actual_coordinates)[1])):
            ax.annotate(labels[idx], xy=xy, textcoords='data')
        for idx, xy in enumerate(zip(transpose(data_coordinates)[0], transpose(data_coordinates)[1])):
            ax.annotate(labels[idx], xy=xy, textcoords='data')
        # Generate a set of interpolated points to animate the transformation
        data = [[lerp(p1, p2, t) for p1, p2 in zip(min_coordinates, transformed_coordinates)] for t in
                linspace(0.0, 1.0, 20)]

        # An update function which will set the animated scatter plot to the next interpolated points
        def update(i):
            scat.set_offsets(data[i % 20])
            return scat,

        for l, o in zip(get_header_labels(), output):
            print(l + ": " + str(o))

        # Begin the animation/plot
        # noinspection PyUnusedLocal
        anim = animation.FuncAnimation(fig, update, interval=100, blit=True)
        plt.show()

    return output


# This function is responsible for returning the names of the values returned in full_pipeline
def get_header_labels():
    return ["Original Misplacement", "Original Swap", "Raw Deanonymized Misplacement", "Transformation Auto-Exclusion",
            "Number of Points Excluded From Geometric Transform", "Rotation Theta", "Scaling", "Translation Magnitude",
            "Number of Components", "Accurate Placements", "Inaccurate Placements", "True Swaps",
            "Partial Swaps", "Cycle Swaps", "Partial Cycle Swaps"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test code
    a, b = generate_random_test_points()

    full_pipeline(a, b, visualize=True)

    full_pipeline(a, b, visualize=True, flags=PipelineFlags.Simple)
