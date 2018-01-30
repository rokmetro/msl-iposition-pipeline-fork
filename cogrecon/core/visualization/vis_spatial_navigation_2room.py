import os
import logging

import numpy as np

from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if __name__ == '__main__':
    from cogrecon.core.data_flexing.spatial_navigation.spatial_navigation_parser import read_raw_file, \
        read_summary_file, get_simple_path_from_raw_iterations, get_simple_orientation_path_from_raw_iterations, \
        compress
else:
    from ..data_flexing.spatial_navigation.spatial_navigation_parser import read_raw_file, read_summary_file, \
        get_simple_path_from_raw_iterations, get_simple_orientation_path_from_raw_iterations, compress


def get_direction_patch_location(center, radius, angle):
    """
    Returns a point whose location is determined by a location, radius and angle.

    :param center: the center/anchor point from which the new point is determined
    :param radius: the distance from the center at which the new point is located
    :param angle: the angle at which the new point is located

    :return: a new point
    """
    return center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)


def update_line(num, avatar, avatar_direction_marker, fig, pos, orient, line, r):
    """
    This helper function animates the visualization.

    :param avatar: the character marker
    :param avatar_direction_marker: the character direction marker
    :param fig: the figure to which plotting is performed
    :param num: the iteration number
    :param pos: the position data
    :param orient: the orientation data
    :param line: the path line to be updated
    :param r: the radius for the direction marker

    :return: the updated line information associated with 'line' input
    """
    jump = 10
    num *= jump
    pos_data = pos[..., num]
    ori_data = orient[num]
    avatar.center = pos_data
    avatar_direction_marker.center = get_direction_patch_location(pos_data, r, ori_data)
    fig.canvas.draw()
    line.set_data(pos[..., :num])
    return line,


def visualize(raw_file, summary_file,
              background_img_path='./media/spatial_navigation/background.png',
              frame_interval_ms=1):
    """
    This function provides basic visualization for the 2-Room Spatial Navigation task given a raw and summary input
    file.

    :param frame_interval_ms: the updated interval of the animation in ms
    :param background_img_path: the path to the image to be displayed in the background
    :param raw_file: the full path to the raw data file
    :param summary_file: the full path to the summary data file
    """

    avatar_size = 1
    direction_marker_propotion_size = 0.25
    padding_size = 0.3

    bounds = (-20, 20, -40, 40)

    # Set data
    logging.info('Reading raw file {0}...'.format(raw_file))
    raw_iterations, raw_events = read_raw_file(raw_file)
    logging.info('Reading summary file {0}...'.format(summary_file))
    summary_events = read_summary_file(summary_file)
    logging.info('Processing raw path into simple path...')
    position_data = get_simple_path_from_raw_iterations(raw_iterations)
    logging.info('Plotting {0} points.'.format(len(position_data)))
    logging.info('Processing raw path into simple orientation...')
    orientation_data = get_simple_orientation_path_from_raw_iterations(raw_iterations)
    logging.info('Compressing data points...')
    position_data, orientation_data = compress(position_data, orientation_data)
    logging.info('Plotting {0} compressed points.'.format(len(position_data)))
    position_data = np.transpose(position_data)
    orientation_data = np.transpose(orientation_data)
    data_length = len(raw_iterations)

    logging.info('Generating figures...')
    # Set up figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Set up markers and lines
    l, = plt.plot([], [], 'r-')
    avatar = plt.Circle((0, 0), avatar_size, fc='y')
    avatar_direction_marker_true_size = direction_marker_propotion_size * avatar_size
    r_adj = avatar_size - avatar_direction_marker_true_size
    avatar_direction_marker = plt.Circle((0, 0), avatar_direction_marker_true_size, fc='b')
    ax.add_patch(avatar)
    ax.add_patch(avatar_direction_marker)

    for event in summary_events:
        col = 'k'
        if event['eventType'] == 'placed':
            col = 'b'
        elif event['eventType'] == 'picked':
            col = 'r'
        elif event['eventType'] == 'identified':
            col = 'y'
        elif event['eventType'] == 'deidentified':
            col = 'g'
        try:
            ax.add_patch(plt.Circle((event['location'][0], event['location'][2]), 1, fc=col, alpha=0.25,
                                    label=event['objectName']))
        except KeyError:
            continue

    # Set up plot bounds
    padded_bounds = np.array(bounds) * (padding_size + 1.)

    plt.xlim(padded_bounds[0], padded_bounds[1])
    plt.ylim(padded_bounds[2], padded_bounds[3])

    # Set up plot labels
    plt.title('Holodeck Spatial Navigation Animation')

    logging.info('Loading background image...')

    # Show Background Image
    img = imread(os.path.abspath(background_img_path))
    plt.imshow(img, zorder=0, extent=bounds)

    logging.info('Generating animation...')

    # Animate Line
    # noinspection PyUnusedLocal
    anim = animation.FuncAnimation(fig, update_line, data_length, fargs=(avatar, avatar_direction_marker, fig,
                                                                         position_data, orientation_data, l, r_adj),
                                   interval=frame_interval_ms, blit=True)

    logging.info('Showing figure...')

    try:
        plt.show()
    except AttributeError:
        pass

    return anim


if __name__ == "__main__":
    directory = r'Z:\Kelsey\2017 Summer RetLu\Virtual_Navigation_Task\v5_2\NavigationTask_Data\Logged_Data' \
                r'\2RoomTestAnonymous\124\\'
    raw_filepath = directory + 'RawLog_Sub124_Trial1_13_15_57_30-05-2017.csv'
    summary_filepath = directory + 'SummaryLog_Sub124_Trial1_13_15_57_30-05-2017.csv'
    visualize(os.path.join(directory, raw_filepath), os.path.join(directory, summary_filepath))
