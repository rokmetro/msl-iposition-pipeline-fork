import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import full_pipeline as pipe
import tools as tools
from data_structures import TrialData, ParticipantData, AnalysisConfiguration


# animation length in seconds
# animation ticks in frames
# noinspection PyDefaultArgument
def visualization(actual_points, data_points, min_points, transformed_points, output_list,
                  z_value=1.96,
                  animation_duration=2, animation_ticks=20, debug_labels=['']):
    for l, o in zip(pipe.get_header_labels(), output_list):
        print(l + ": " + str(o))

    if len(actual_points[0]) != 2:
        logging.error("the visualization method expects 2D points, found {0}D".format(len(actual_points[0])))
        return

    # Generate a figure with 3 scatter plots (actual points, data points, and transformed points)
    fig, ax = plt.subplots()
    plt.title(str(debug_labels))
    ax.set_aspect('equal')
    labels = range(len(actual_points))
    x = [float(v) for v in list(np.transpose(transformed_points)[0])]
    y = [float(v) for v in list(np.transpose(transformed_points)[1])]
    ax.scatter(x, y, c='b', alpha=0.5)
    scat = ax.scatter(x, y, c='b', animated=True)
    ax.scatter(np.transpose(actual_points)[0], np.transpose(actual_points)[1], c='g', s=50)
    ax.scatter(np.transpose(data_points)[0], np.transpose(data_points)[1], c='r', s=50)
    # Label the stationary points (actual and data)
    for idx, xy in enumerate(zip(np.transpose(actual_points)[0], np.transpose(actual_points)[1])):
        ax.annotate(labels[idx], xy=xy, textcoords='data', fontsize=20)
    for idx, xy in enumerate(zip(np.transpose(data_points)[0], np.transpose(data_points)[1])):
        ax.annotate(labels[idx], xy=xy, textcoords='data', fontsize=20)
    # Generate a set of interpolated points to animate the transformation
    lerp_data = [[tools.lerp(p1, p2, t) for p1, p2 in zip(min_points, transformed_points)] for t in
                 np.linspace(0.0, 1.0, animation_ticks)]

    _analysis_configuration = AnalysisConfiguration(z_value=z_value, trial_by_trial_accuracy=True)
    _participant_data = ParticipantData([TrialData(actual_points, transformed_points)])
    participant_data = pipe.accuracy(_participant_data, _analysis_configuration)
    accuracies = participant_data.distance_accuracy_map
    print(participant_data.distance_accuracy_map)
    threshold = participant_data.distance_threshold
    accuracies = accuracies[0]
    threshold = threshold[0]
    for acc, x, y in zip(accuracies, np.transpose(transformed_points)[0], np.transpose(transformed_points)[1]):
        color = 'r'
        if acc:
            color = 'g'
        ax.add_patch(plt.Circle((x, y), threshold, alpha=0.3, color=color))

    _participant_data = ParticipantData([TrialData(actual_points, min_points)])
    participant_data = pipe.accuracy(_participant_data, _analysis_configuration)
    accuracies = participant_data.distance_accuracy_map
    threshold = participant_data.distance_threshold
    accuracies = accuracies[0]
    threshold = threshold[0]

    for acc, x, y in zip(accuracies, np.transpose(min_points)[0], np.transpose(min_points)[1]):
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
