from time_travel_task_binary_reader import find_data_files_in_directory, get_item_details, read_binary_file, \
    parse_test_items, get_filename_meta_data, compute_accuracy, get_exploration_metrics, \
    get_click_locations_and_indicies, get_items_solutions
from itertools import chain, izip
import numpy as np
import os
import easygui
import logging
import scipy.optimize


# example search_directory: r'C:\Users\Kevin\Desktop\Work\Time Travel Task\v2'
def summarize_test_data(search_directory=None, file_regex="\d\d\d_\d_2_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat",
                        output_path='time_travel_task_test_summary.csv', last_pilot_id=20, verbose=True):
    """
    This function will produce an output file which contains the test summary data for all data files in a
    directory, searching recursively based on a regular expression.

    :param search_directory: the directory to recursively search for files matching file_regex to process
    :param file_regex: the regular expression for files to search for in search_directory, recursively
    :param output_path: the output path to save the data (CSV format)
    :param last_pilot_id: for the 'pilot?' column, IDs less than or equal to last_pilot_id will be marked True
    :param verbose: If True, participant information will be logged to logging.info as parsing progresses.

    :rtype: bool
    :return: True if files were successfully processed, False otherwise.
    """
    if search_directory is None:
        search_directory = easygui.diropenbox()
    if search_directory is '':
        logging.info('No directory selected, returning.')
        return False
    if not os.path.exists(search_directory):
        raise IOError('Specified search directory was not found.')

    files = find_data_files_in_directory(search_directory, file_regex=file_regex)

    if len(files) == 0:
        logging.info('No files found. returning.')
        return False

    event_state_labels, item_number_label, item_label_filename, cols = get_item_details()

    item_labels = list(chain.from_iterable(izip([x + '_x' for x in item_number_label],
                                                [x + '_z' for x in item_number_label],
                                                [x + '_time' for x in item_number_label],
                                                [x + '_type' for x in item_number_label])))

    fp = open(output_path, 'wb')

    header = 'subID,trial,inverse,datetime,placed_items,complete?,pilot?,space_misplacement,time_misplacement,' \
             'space_time_misplacement,event_type_correct_count,context_crossing_dist_exclude_wrong_color_pairs,' \
             'context_noncrossing_dist_exclude_wrong_color_pairs,context_crossing_dist_pairs,' \
             'context_noncrossing_dist_pairs,space_context_crossing_dist_exclude_wrong_color_pairs,' \
             'space_context_noncrossing_dist_exclude_wrong_color_pairs,space_context_crossing_dist_pairs,' \
             'space_context_noncrossing_dist_pairs,' + ','.join(item_labels)

    fp.write(header + '\r\n')

    for path in files:
        iterations = read_binary_file(path)
        reconstruction_items, order = parse_test_items(iterations, cols,
                                                       item_number_label, event_state_labels)

        meta = get_filename_meta_data(os.path.basename(path))
        line_start = meta['subID'] + ',' + meta['trial'] + ',' + meta['inverse'] + ',' + str(meta['datetime'])
        items = [','.join([str(x) for x in item['pos']]) + ',' + str(item['direction']) if item is not None
                 else 'nan,nan,nan,nan' for item in reconstruction_items]
        line_start += ',' + str(len(items) - items.count('nan,nan,nan,nan'))
        complete = items.count('nan,nan,nan,nan') == 0
        line_start += ',' + str(complete)
        line_start += ',' + str(int(meta['subID']) <= last_pilot_id)
        if complete:
            space_m, time_m, space_time_m, event, ccexc, cncexc, cc, cnc, sccexc, scncexc, scc, scnc = \
                compute_accuracy(meta, reconstruction_items)
        else:
            space_m, time_m, space_time_m, event, ccexc, cncexc, cc, cnc, sccexc, scncexc, scc, scnc = \
                (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
        line_start += ',' + str(space_m) + ',' + str(time_m) + ',' + str(space_time_m) + ',' + str(event)
        line_start += ',' + str(ccexc) + ',' + str(cncexc) + ',' + str(cc) + ',' + str(cnc)
        line_start += ',' + str(sccexc) + ',' + str(scncexc) + ',' + str(scc) + ',' + str(scnc)
        line = line_start + ',' + (','.join(items))
        if verbose:
            logging.info(line_start)
        fp.write(line + '\r\n')
        fp.flush()

    fp.close()

    return True


def summarize_navigation_data(search_directory=None, file_regex="\d\d\d_\d_1_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat",
                              output_path='time_travel_task_navigation_summary.csv', last_pilot_id=20, verbose=True,
                              temporal_boundary_regions=None,
                              fd_indicies_time=None,
                              fd_indicies_space=None,
                              fd_indicies_spacetime=None):
    """
    This function will produce an output file which contains the navigation summary data for all data files in a
    directory, searching recursively based on a regular expression.

    :type fd_indicies_time: object
    :param fd_indicies_time: If None, the FD scale window will be calculated for every path independently. Otherwise, a
    list of indicies is expected which specify the scale indicies across the range
    [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0,
    8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]. Currently, this range is not able to be redefined via the top
    level interface.
    :param fd_indicies_space: If None, the FD scale window will be calculated for every path independently. Otherwise, a
    list of indicies is expected which specify the scale indicies across the range
    [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0,
    8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]. Currently, this range is not able to be redefined via the top
    level interface.
    :param fd_indicies_spacetime: If None, the FD scale window will be calculated for every path independently.
    Otherwise, a list of indicies is expected which specify the scale indicies across the range
    [0.0009765625, 0.001953125, 0.00390625, 0.0078125, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0,
    8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0]. Currently, this range is not able to be redefined via the top
    level interface.
    :param search_directory: the directory to recursively search for files matching file_regex to process
    :param file_regex: the regular expression for files to search for in search_directory, recursively
    :param output_path: the output path to save the data (CSV format)
    :param last_pilot_id: for the 'pilot?' column, IDs less than or equal to last_pilot_id will be marked True
    :param verbose: If True, participant information will be logged to logging.info as parsing progresses.
    :param temporal_boundary_regions: the (Nr, Nd, 2) representation of the boundary regions in the environment where
    Nr is the number of regions (note that regions can intersect and do not have to cover the whole space - boundaries
    are defined as when the contained regions for a point change in any way), Nd is the number of dimensions, and 2
    are the min and max of the box region in each dimension for each region. The default is
    [[[-100, 15]], [[15, 30]], [[30, 45]], [[45, 100]]].

    :rtype: bool
    :return: True if files were successfully processed, False otherwise.
    """
    if search_directory is None:
        search_directory = easygui.diropenbox()
    if search_directory is '':
        logging.info('No directory selected, returning.')
        return False
    if not os.path.exists(search_directory):
        raise IOError('Specified search directory was not found.')

    if temporal_boundary_regions is None:
        temporal_boundary_regions = [[[-100, 15]], [[15, 30]], [[30, 45]], [[45, 100]]]

    files = find_data_files_in_directory(search_directory, file_regex=file_regex)

    if len(files) == 0:
        logging.info('No files found. returning.')
        return False

    fp = open(output_path, 'wb')

    header = 'subID,trial,inverse,datetime,pilot?,total_time,space_travelled,time_travelled,space_time_travelled,' + \
             'context_boundary_crossings,fd_time,fd_space,fd_spacetime,' + \
             'lacunarity_time,lacunarity_space,lacunarity_spacetime,click_order'

    fp.write(header + '\r\n')

    for path in files:
        iterations = read_binary_file(path)
        meta = get_filename_meta_data(os.path.basename(path))

        total_time, space_travelled, time_travelled, space_time_travelled = get_exploration_metrics(iterations)

        timeline = [[i['time_val']] for i in iterations]
        spaceline = [[i['x'], i['z']] for i in iterations]
        spacetimeline = [[i['x'], i['z'], i['time_val']] for i in iterations]

        boundary_crossings = count_boundary_crossings(timeline, temporal_boundary_regions)

        fd_t, lac_t = calculate_fd_and_lacunarity(timeline, indicies=fd_indicies_time)
        fd_s, lac_s = calculate_fd_and_lacunarity(spaceline, indicies=fd_indicies_space)
        fd_st, lac_st = calculate_fd_and_lacunarity(spacetimeline, indicies=fd_indicies_spacetime)

        clicks = get_click_locations_and_indicies(iterations, list(range(0, 10)), meta)

        line_start = meta['subID'] + ',' + meta['trial'] + ',' + meta['inverse'] + ',' + str(meta['datetime'])
        line_start += ',' + str(int(meta['subID']) <= last_pilot_id)
        line = line_start + ',' + str(total_time) + ',' + str(space_travelled) + ',' \
                          + str(time_travelled) + ',' + str(space_time_travelled) + ',' \
                          + str(boundary_crossings) + ',' + str(fd_t) + ',' + str(fd_s) + ',' + str(fd_st) + ',' \
                          + str(lac_t) + ',' + str(lac_s) + ',' + str(lac_st) + ',' + '"' \
                          + str(list(clicks['index'])) + '"'

        if verbose:
            logging.info(line_start)

        fp.write(line + '\r\n')
        fp.flush()

    fp.close()

    return True


def get_regions_of_points(point, regions):
    in_regions = []
    for region in regions:
        in_regions.append(all([r[0] < p <= r[1] for p, r in zip(point, region)]))
    return in_regions


def count_boundary_crossings(path_points, regions):
    count = 0
    prev_regions = get_regions_of_points(path_points[0], regions)
    for p in path_points:
        current_regions = get_regions_of_points(p, regions)
        count += prev_regions != current_regions
        prev_regions = current_regions
    return count


def count_boxes(data, scale):
    boxed_path = np.floor(np.divide(data, scale))
    unique = np.unique(boxed_path, axis=0)
    filled_boxes = len(unique)
    return filled_boxes


def calculate_fd_and_lacunarity(data, indicies=None):
    scale_range = 20
    r = np.array([2.0 ** (scale_range / 2) / (2.0 ** i) for i in range(scale_range, 0, -1)])  # Powers of 2 around 0
    N = [count_boxes(data, ri) for ri in r]
    Nlog = np.log(N)
    ste = np.std(Nlog) / np.sqrt(len(data))
    if indicies is None:
        indicies = [idx for idx, n in enumerate(Nlog) if (not n <= (min(Nlog) + ste) and not n >= (max(Nlog) - ste))]
    N = np.take(N, indicies)
    r = np.take(r, indicies)

    def linear_function(x, A, Df):
        return Df * x + A

    popt, pcov = scipy.optimize.curve_fit(linear_function, np.log(1. / r), np.log(N))
    lacunarity, fd = popt

    return fd, lacunarity
