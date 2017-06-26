from time_travel_task_binary_reader import find_data_files_in_directory, get_item_details, read_binary_file, \
    parse_test_items, get_filename_meta_data, compute_accuracy, get_exploration_metrics
from itertools import chain, izip
import numpy as np
import os
import easygui
import logging


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
        billboard_item_labels, reconstruction_items = parse_test_items(iterations, cols,
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


def summarize_navigation_data(search_directory=None, file_regex="\d\d\d_\d_2_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat",
                              output_path='time_travel_task_test_summary.csv', last_pilot_id=20, verbose=True):
    """
    This function will produce an output file which contains the navigation summary data for all data files in a
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

    fp = open(output_path, 'wb')

    header = 'subID,trial,inverse,datetime,pilot?,total_time,space_travelled,time_travelled,space_time_travelled'

    fp.write(header + '\r\n')

    for path in files:
        iterations = read_binary_file(path)
        total_time, space_travelled, time_travelled, space_time_travelled = get_exploration_metrics(iterations)
        meta = get_filename_meta_data(os.path.basename(path))
        line_start = meta['subID'] + ',' + meta['trial'] + ',' + meta['inverse'] + ',' + str(meta['datetime'])
        line_start += ',' + str(int(meta['subID']) <= last_pilot_id)
        line = line_start + ',' + str(total_time) + ',' + str(space_travelled) + ',' \
                          + str(time_travelled) + ',' + str(space_time_travelled)
        if verbose:
            logging.info(line_start)
        fp.write(line + '\r\n')
        fp.flush()

    fp.close()

    return True
