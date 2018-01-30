import os
import csv
import logging
import datetime
import argparse

import numpy as np

from scipy.spatial import distance

import spatial_navigation_parser as log_parser

if __name__ == "__main__":
    from cogrecon.core.cogrecon_globals import data_coordinates_file_suffix, actual_coordinates_file_suffix, category_file_suffix, \
        order_file_suffix
else:
    from ...cogrecon_globals import data_coordinates_file_suffix, actual_coordinates_file_suffix, category_file_suffix, \
        order_file_suffix


def generate_intermediate_files_command_line():
    # Parse inputs
    parser = argparse.ArgumentParser(
        description='This script will take a folder containing subject data for the Holodeck Navigation Task and '
                    'generate CSV files containing the meta-data of interest as requested by the script options. The '
                    'default is for all data to be generated in a folder adjacent to this script tagged with the '
                    'current data and time. To speed processing, this can be restricted to particular subsets of the '
                    'meta-data. If any one option is given to specify a subset of processing, the script will, '
                    'by default, exclude subsets of data not included as options.')
    parser.add_argument('path', help='The full input path to the folder containing the data to be parsed.')
    parser.add_argument('--full_study_path', dest='full_study_path', action='store_true',
                        help='Generates study_path.csv containing relevant study path variables.')
    parser.set_defaults(full_study_path=False)
    parser.add_argument('--full_study_look', dest='full_study_look', action='store_true',
                        help='Generates study_look.csv containing relevant study look variables.')
    parser.set_defaults(full_study_look=False)
    parser.add_argument('--full_test_path', dest='full_test_path', action='store_true',
                        help='Generates test_path.csv containing relevant test path variables.')
    parser.set_defaults(full_test_path=False)
    parser.add_argument('--full_test_look', dest='full_test_look', action='store_true',
                        help='Generates test_look.csv containing relevant test look variables.')
    parser.set_defaults(full_test_look=False)
    parser.add_argument('--full_practice_path', dest='full_practice_path', action='store_true',
                        help='Generates practice_path.csv containing relevant practice path variables.')
    parser.set_defaults(full_practice_path=False)
    parser.add_argument('--full_practice_look', dest='full_practice_look', action='store_true',
                        help='Generates practice_look.csv containing relevant practice look variables.')
    parser.set_defaults(full_practice_look=False)
    parser.add_argument('--full_test_2d', dest='full_test_2d', action='store_true',
                        help='Generates 2d_test.csv containing relevant 2D test results.')
    parser.set_defaults(full_test_2d=False)
    parser.add_argument('--full_test_vr', dest='full_test_vr', action='store_true',
                        help='Generates vr_test.csv containing relevant 2D test results.')
    parser.set_defaults(full_test_vr=False)

    parser.add_argument('--min_num_trials', default=4, type=int,
                        help='Minimum number of valid, complete trials necessary to include subject in output ('
                             'default=1).')

    parser.add_argument('--log_level', default=20, type=int,
                        help='Logging level of the application (default=20/INFO). ' +
                             'See https://docs.python.org/2/library/logging.html#levels for more info.')
    parser.set_defaults(log_level=20)

    args = parser.parse_args()
    generate_intermediate_files(args.path, args.full_study_path, args.full_study_look, args.full_test_path,
                                args.full_test_look, args.full_practice_path, args.full_practice_look,
                                args.full_test_2d, args.full_test_vr, args.log_level, args.min_num_trials)


def generate_intermediate_files(search_path, full_study_path=False, full_study_look=False, full_test_path=False,
                                full_test_look=False, full_practice_path=False, full_practice_look=False,
                                full_test_2d=False, full_test_vr=False, min_num_trials=4):
    """
    This script will take a folder containing subject data for the Holodeck Navigation Task and generate CSV files
    containing the meta-data of interest as requested by the script options. The default is for all data to be
    generated in a folder adjacent to this script tagged with the current data and time. To speed processing, this can
    be restricted to particular subsets of the meta-data. If any one option is given to specify a subset of processing,
    the script will by default, exclude subsets of data not included as options.

    :param search_path: the path in which to search for files recursively
    :param full_study_path: if True, study paths will be processed
    :param full_study_look: if True, study looking will be processed
    :param full_test_path: if True, test paths will be processed
    :param full_test_look: if True, test looking will be processed
    :param full_practice_path: if True, practice paths will be processed
    :param full_practice_look: if True, practice looking will be processed
    :param full_test_2d: if True, test 2D will be processed
    :param full_test_vr: if True, test VR will be processed
    :param log_level: the logging.level to output
    :param min_num_trials: the minimum number of trials required for included data
    """

    # Handle case where no optional arguments excluding processing are provided in which case all optional
    # args are assumed to be true (for convenience)
    if not (full_study_path or full_study_look or full_test_path or
            full_test_look or full_practice_path or full_practice_look or
            full_test_2d or full_test_vr):
        full_study_path = True
        full_study_look = True
        full_test_path = True
        full_test_look = True
        full_practice_path = True
        full_practice_look = True
        full_test_2d = True
        full_test_vr = True
        logging.info("No command line arguments found. Defaulting all true.")

    logging.info("Done parsing command line arguments.")

    # Populate list of files, recursively
    files = []
    for walk_root, walk_dirs, walk_files in os.walk(search_path):
        for f in walk_files:
            files.append(os.path.join(walk_root, f))

    # Check if there aren't any files and early stop if there aren't
    if not files:
        logging.error("No files found in directory. Closing without creation of output files.")
        return

    logging.info("Found %d files. Attempting to catalog filenames by Individual, Trial, and Phase" % len(files))

    # Stores filenames for individuals in a data structure for easy handling
    individuals, excluded, non_matching = log_parser.catalog_files(files,
                                                                   min_num_trials)

    logging.info(("Done cataloging files. %d individuals found which conform to the trial minimum (%d). " +
                  "%d files not matching any expected filename format. %d files excluded " +
                  "on input criteria.")
                 % (len(individuals), min_num_trials,
                    len(non_matching), len(excluded)))

    # In debug mode, print excluded files
    for filename in excluded:
        logging.debug("%s was excluded." % filename)

    # Create the output directory
    output_directory = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.mkdir(output_directory)
        logging.info("Output directory (%s) created." % output_directory)
    except OSError as e:
        if e.errno != 17:
            raise
        else:
            logging.info("Output directory (%s) already exists. Continuing..." % output_directory)

    logging.info("Creating output files.")

    # Generate variable references for the csv writers
    full_study_path_writer = None
    full_study_look_writer = None
    full_test_path_writer = None
    full_test_look_writer = None
    full_practice_path_writer = None
    full_practice_look_writer = None
    full_test_2d_writer = None
    full_test_vr_writer = None

    output_file_pointers = []
    output_file_pointer = None

    # Create the appropriate output files for the options requested and write their headers
    if full_study_path:
        full_study_path_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "study_path.csv",
                                        ["subject_id",
                                         "trial_number", "time",
                                         "x", "y",
                                         "z", "room_by_order",
                                         "room_by_color",
                                         "items_clicked",
                                         "distance_from_last_point",
                                         "time_since_last_point",
                                         "most_recent_item_interaction"])
    output_file_pointers.append(output_file_pointer)
    if full_study_look:
        full_study_look_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "study_look.csv",
                                        ["subject_id",
                                         "trial_number", "time",
                                         "x", "y",
                                         "z", "w", "euler_x",
                                         "euler_y", "euler_z",
                                         "room_by_order",
                                         "room_by_color",
                                         "items_clicked",
                                         "distance_from_last_point",
                                         "time_since_last_point",
                                         "most_recent_item_interaction"])
    output_file_pointers.append(output_file_pointer)
    if full_test_path:
        full_test_path_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "test_path.csv",
                                        ["subject_id",
                                         "trial_number", "time", "x",
                                         "y",
                                         "z", "room_by_order",
                                         "room_by_color",
                                         "items_clicked",
                                         "distance_from_last_point",
                                         "time_since_last_point",
                                         "most_recent_item_interaction"])
    output_file_pointers.append(output_file_pointer)
    if full_test_look:
        full_test_look_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "test_look.csv",
                                        ["subject_id",
                                         "trial_number", "time", "x",
                                         "y",
                                         "z", "w", "euler_x",
                                         "euler_y", "euler_z",
                                         "room_by_order",
                                         "room_by_color",
                                         "items_clicked",
                                         "distance_from_last_point",
                                         "time_since_last_point",
                                         "most_recent_item_interaction"])
    output_file_pointers.append(output_file_pointer)
    if full_practice_path:
        full_practice_path_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "practice_path.csv",
                                        ["subject_id",
                                         "trial_number", "time",
                                         "x",
                                         "y",
                                         "z", "room_by_order",
                                         "room_by_color",
                                         "items_clicked",
                                         "distance_from_last_point",
                                         "time_since_last_point",
                                         "most_recent_item_interaction"])
    output_file_pointers.append(output_file_pointer)
    if full_practice_look:
        full_practice_look_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "practice_look.csv",
                                        ["subject_id",
                                         "trial_number", "time",
                                         "x",
                                         "y",
                                         "z", "w", "euler_x",
                                         "euler_y", "euler_z",
                                         "room_by_order",
                                         "room_by_color",
                                         "items_clicked",
                                         "distance_from_last_point",
                                         "time_since_last_point",
                                         "most_recent_item_interaction"])
    output_file_pointers.append(output_file_pointer)
    if full_test_2d:
        full_test_2d_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "2d_test.csv",
                                        ["subject_id", "trial_number",
                                         "item_id",
                                         "x_placed", "y_placed",
                                         "x_expected", "y_expected",
                                         "order_clicked_study",
                                         "expected_room_by_order",
                                         "expected_room_by_color",
                                         "actual_room_by_order",
                                         "actual_room_by_color"])
    output_file_pointers.append(output_file_pointer)
    if full_test_vr:
        full_test_vr_writer, output_file_pointer = \
            log_parser.make_output_file(output_directory,
                                        "vr_test.csv",
                                        ["subject_id", "trial_number",
                                         "item_id",
                                         "x_placed", "y_placed",
                                         "x_expected", "y_expected",
                                         "order_clicked_study",
                                         "expected_room_by_order",
                                         "expected_room_by_color",
                                         "actual_room_by_order",
                                         "actual_room_by_color",
                                         "number_of_replacements",
                                         "time_placed"])
    output_file_pointers.append(output_file_pointer)

    logging.info("Parsing input files.")

    # Parse each individual and contribute their data to the appropriate file if possible
    count = 1
    for individual in individuals:
        logging.info("Parsing Individual %s (%d/%d)." % (individual.subject_id, count, len(individuals)))
        count += 1
        trial_count = 1
        for trial in individual.trials:
            logging.info("Parsing Trial %d (%d/%d)." % (trial.num, trial_count, len(individual.trials)))
            trial_count += 1
            if full_study_path:
                log_parser.parse_file_and_write(trial.study_path, individual.subject_id, trial.num,
                                                log_parser.FileType.path_file,
                                                full_study_path_writer,
                                                trial.study_summary)
            if full_study_look:
                log_parser.parse_file_and_write(trial.study_look, individual.subject_id, trial.num,
                                                log_parser.FileType.look_file,
                                                full_study_look_writer,
                                                trial.study_summary)
            if full_test_path:
                log_parser.parse_file_and_write(trial.test_path, individual.subject_id, trial.num,
                                                log_parser.FileType.path_file,
                                                full_test_path_writer,
                                                trial.test_summary)
            if full_test_look:
                log_parser.parse_file_and_write(trial.test_look, individual.subject_id, trial.num,
                                                log_parser.FileType.look_file,
                                                full_test_look_writer,
                                                trial.test_summary)
            if full_practice_path:
                log_parser.parse_file_and_write(trial.practice_path, individual.subject_id, trial.num,
                                                log_parser.FileType.path_file,
                                                full_practice_path_writer,
                                                trial.practice_summary)
            if full_practice_look:
                log_parser.parse_file_and_write(trial.practice_look, individual.subject_id, trial.num,
                                                log_parser.FileType.look_file,
                                                full_practice_look_writer,
                                                trial.practice_summary)
            if full_test_2d:
                log_parser.parse_file_and_write(trial.test_2d, individual.subject_id, trial.num,
                                                log_parser.FileType.test_file_2d,
                                                full_test_2d_writer,
                                                trial.study_summary)
            if full_test_vr:
                log_parser.parse_file_and_write(trial.test_vr, individual.subject_id, trial.num,
                                                log_parser.FileType.test_file_vr,
                                                full_test_vr_writer,
                                                trial.study_summary)

    logging.info("Done parsing input files.")

    logging.info("Closing output files.")

    # Close all writers if they were opened
    for pointer in output_file_pointers:
        log_parser.close_writer(pointer)

    logging.info('Parsing complete.')


def generate_buffered_points_and_clicks(data_dict, subject_id, trial_num):
    """
    This function generates positions, times, clicks, room_by_color and lates items from a data dictionary.

    :param data_dict: the data dictionary from which to extract data
    :param subject_id: the subject id to extract
    :param trial_num: the trial number to extract

    :return: a tuple containing the x, y, t, click, room_by_color, and latest_item lists for all iterations in the
             data dictionary
    """
    internal_id = subject_id + trial_num
    dat = data_dict[internal_id]
    xr = []
    zr = []
    time = []
    click = []
    room_by_color = []
    latest_item = []
    for l in dat:
        xr.append(float(l[3]))
        # zr.append(float(l[4]))
        zr.append(float(l[5]))
        time.append(float(l[10]) / (np.power(10, 7)))
        click.append(float(l[8]))
        latest_item.append(l[11])
        room_by_color.append(l[7])

    return xr, zr, time, click, room_by_color, latest_item


def generate_item_specific_efficiency(data_dict, subject_id, trial_num):
    """
    This function generates an error list by item showing the efficiency in getting to particular items.

    :param data_dict: the data dictionary from which to extract data
    :param subject_id: the subject id to extract
    :param trial_num: the trial number to extract

    :return: a list of dictionaries containing 'item' and 'error' keys
    """
    xrs, zrs, times, clicks, room_by_colors, latest_items = generate_buffered_points_and_clicks(data_dict,
                                                                                                subject_id,
                                                                                                trial_num)
    search_data = enumerate(zip(xrs, zrs, times, clicks, room_by_colors, latest_items))
    next(search_data)

    targets = []
    for index, (x, y, time, click, room_by_color, latest_item) in search_data:
        if click != clicks[index - 1]:
            targets.append((x, y, time, click, room_by_color, latest_item))

    search_data = enumerate(zip(xrs, zrs, times, clicks, room_by_colors, latest_items))
    next(search_data)
    current_target_index = 0
    error_sum = 0
    error_list_by_item = []
    partial_error_sum_by_item = 0
    error_list = [0]
    cumulative_error_list = [0]
    for index, (x, y, time, click, room_by_color, latest_item) in search_data:
        if current_target_index >= len(targets):  # If we're out of targets, stop computing error
            break
        # Generate the vector coordinates of previous location, current location, and target location
        (x, y) = (float(x), float(y))
        (prev_x, prev_y) = (float(xrs[index - 1]), float(zrs[index - 1]))
        (target_x, target_y) = (float(targets[current_target_index][0]), float(targets[current_target_index][1]))
        # Generate vectors which point from the previous location to the current and target location.
        real_vector = (x - prev_x, y - prev_y)
        ideal_vector = (target_x - prev_x, target_y - prev_y)
        # Get the magnitude of both the real and target vectors
        real_vector_magnitude = np.linalg.norm(real_vector)
        ideal_vector_magnitude = np.linalg.norm(ideal_vector)
        # Normalize both vectors for comparison, catching case where vectors are (0, 0)
        if real_vector_magnitude == 0:
            normed_real_vector = (0, 0)
        else:
            normed_real_vector = np.multiply(np.divide(real_vector, real_vector_magnitude), time)
        if ideal_vector_magnitude == 0:
            normed_ideal_vector = (0, 0)
        else:
            normed_ideal_vector = np.multiply(np.divide(ideal_vector, ideal_vector_magnitude), time)
        # Generate the magnitude of the difference between these vectors (this variable changes non-linearly as
        # the real vector is rotated relative to the target, but it's consistently non-linear so rescaling via the sine
        # isn't really necessary)
        difference_magnitude = np.linalg.norm(np.subtract(normed_ideal_vector, normed_real_vector))
        # The final error contribution is the magnitude of the spatial component and the temporal component as a vector
        # This penalizes individuals for standing still according to how much time they stood still.
        if difference_magnitude == 0:
            difference_magnitude = time
        error = np.abs(difference_magnitude)
        error_sum += error
        partial_error_sum_by_item += error
        error_list.append(error)
        cumulative_error_list.append(error_sum)
        if click != clicks[index - 1]:
            current_target_index += 1
            error_list_by_item.append({'item': latest_item, 'error': partial_error_sum_by_item})
            partial_error_sum_by_item = 0

    # total_error = np.sum(error_list)
    # validation_sum = 0
    # for element in error_list_by_item:
    #    print "Item: {0}, Error: {1}".format(element['item'], str(element['error']))
    #    validation_sum += element['error']
    # print "Total Error: " + str(total_error) + ", Validation Error: " + str(validation_sum)
    return error_list_by_item


def generate_study_path_efficiency_command_line():
    """
    This function can be called from a python script to wrap the generate_study_path_efficiency function in command
    line arguments which will be parsed and passed to the command.
    """
    # Parse inputs
    parser = argparse.ArgumentParser(
        description='This script will process a *_path.csv file (first input arg) and generate basic exploration '
                    'metrics (distance, time, distance/time) for the various segments of data (a segment is any '
                    'section of data points which share a subject_id, trial_number, room_by_order/room_by_color). The '
                    'data will be saved in the output_path parameter (second input arg) in csv format.')
    parser.add_argument('path', help='The full input path to the *_path.csv file generated by ' +
                                     'Holodeck_GenerateIntermediateFiles.py that you wish to process.')

    parser.add_argument('--log_level', default=20, type=int,
                        help='Logging level of the application (default=20/INFO). ' +
                             'See https://docs.python.org/2/library/logging.html#levels for more info.')
    parser.set_defaults(log_level=20)

    args = parser.parse_args()

    generate_study_path_efficiency(args.path, args.point_speed)


def generate_study_path_efficiency(input_file):

    """
    This function will generate an efficiency_by_item.csv output file which lists the item-by-item efficiency for a log
    file. The input file is assumed to be an intermediate file from generate_intermediate_files.

    :param input_file: an input file to be processed
    """

    logging.info("Done parsing command line arguments.")

    fp_reader = open(input_file, 'rb')
    reader = csv.reader(fp_reader)
    # noinspection PyUnusedLocal
    header = reader.next()

    data_dict = dict()

    for line in reader:
        segment_id = line[0] + line[1]
        if segment_id in data_dict:
            data_dict[segment_id].append(line)
        else:
            data_dict[segment_id] = [line]

    logging.info("Done parsing input files into segments. %d segments found. Closing input file." % len(data_dict))

    fp_reader.close()

    if len(data_dict) == 0:
        logging.error("Error: No segments found in input file. Are you sure this is a properly formatted input file?")
        exit()

    keys = data_dict.keys()
    identifiers = []
    for k in keys:
        identifiers.append([k[0:3], k[3]])

    lines = []
    for _id in identifiers:
        items = generate_item_specific_efficiency(data_dict, _id[0], _id[1])
        for i in items:
            lines.append(','.join([str(_id[0]), str(_id[1]), str(i['item']), str(i['error'])]) + '\r\n')

    header = 'subject_id,trial_number,item_id,efficiency\r\n'

    f = open('efficiency_by_item.csv', 'wb')
    f.write(header)

    for line in lines:
        f.write(line)

    f.close()

    '''

    app = QtGui.QApplication([])
    win = pg.GraphicsWindow(title="Basic plotting examples")
    win.resize(1000, 600)
    win.setWindowTitle('pyqtgraph example: Plotting')
    p = win.addPlot(title='Path')
    p2 = win.addPlot(title='Error')

    img = QtGui.QImage('flat_no_items.jpg')
    img = img.convertToFormat(QtGui.QImage.Format_ARGB32_Premultiplied)
    imgArray = pg.imageToArray(img, copy=True)
    img = pg.ImageItem(imgArray)
    img.setPxMode(False)
    img.setRect(QtCore.QRectF(-12.5, -2.5, 80, 80))
    img.x = -12.5
    img.y = -2.5
    p.addItem(img)

    (x, y) = (xrs, zrs)
    ptr = 1
    buf_x = [x[0]]
    buf_y = [y[0]]
    buf_error = [cumulative_error_list[0]]
    done = False


    def update():
        global p, x, y, ptr, buf_x, buf_y, args, done
        p.clear()
        p2.clear()

        if not done:
            for i in range(0, args.point_speed):
                if ptr >= len(x):
                    done = True
                    break
                buf_x.append(x[ptr])
                buf_y.append(y[ptr])
                if ptr >= len(error_list):
                    buf_error.append(0)
                else:
                    buf_error.append(error_list[ptr])
                ptr += 1
            curve = pg.PlotCurveItem(x=buf_x, y=buf_y, pen=args.colors[0], symbol='o', brush='b', size=1)
        else:
            curve = pg.PlotCurveItem(x=buf_x, y=buf_y, pen=args.colors[1], symbol='o', brush='b', size=1)
        p.addItem(img)
        p.addItem(curve)
        p2.plot(buf_error)

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(33)

    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    '''


def generate_basic_exploration_metrics_from_intermediate_files_command_line():
    """
    This function can be called from a python script to wrap the
    generate_basic_exploration_metrics_from_intermediate_files function in command line arguments which will be parsed
    and passed to the command.
    """
    # Parse inputs
    parser = argparse.ArgumentParser(
        description='This script will process a *_path.csv file (first input arg) and generate basic exploration '
                    'metrics (distance, time, distance/time) for the various segments of data (a segment is any '
                    'section of data points which share a subject_id, trial_number, room_by_order/room_by_color). The '
                    'data will be saved in the output_path parameter (second input arg) in csv format.')
    parser.add_argument('path', help='The full input path to the *_path.csv file generated by ' +
                                     'Holodeck_GenerateIntermediateFiles.py that you wish to process.')
    parser.add_argument('output_path',
                        help='The full input output path (ending with .csv) of the file to be generated ' +
                             'by this script containing the expected output fields formatted as csv.')

    parser.add_argument('--log_level', default=20, type=int,
                        help='Logging level of the application (default=20/INFO). ' +
                             'See https://docs.python.org/2/library/logging.html#levels for more info.')
    parser.set_defaults(log_level=20)

    args = parser.parse_args()

    generate_basic_exploration_metrics_from_intermediate_files(args.path, args.output_path, args.log_level)


def generate_basic_exploration_metrics_from_intermediate_files(input_path, output_path):
    """
    This function will generate basic exploration metrics given the intermediate files from generate_intermediate_files.

    :param input_path: the path to the intermediate file
    :param output_path: the output path to which to save the metrics

    :return: Nothing
    """

    logging.info("Done parsing command line arguments.")

    if not os.path.isfile(input_path):
        logging.error(
            "Error: Input file %s not found. Make sure you use the complete folder+filename path." % input_path)
        exit()

    logging.info("Found input file %s. Attempting to parse into segments." % input_path)

    fp_reader = open(input_path, 'rb')
    reader = csv.reader(fp_reader)
    # noinspection PyUnusedLocal
    header = reader.next()

    data_dict = dict()

    for line in reader:
        segment_id = line[0] + line[1] + line[6] + line[7]
        if segment_id in data_dict:
            data_dict[segment_id].append(line)
        else:
            data_dict[segment_id] = [line]

    logging.info("Done parsing input files into segments. %d segments found. Closing input file." % len(data_dict))

    fp_reader.close()

    if len(data_dict) == 0:
        logging.error("Error: No segments found in input file. Are you sure this is a properly formatted input file?")
        exit()

    logging.info("Creating output file %s." % output_path)

    fp_writer = open(output_path, 'wb')
    writer = csv.writer(fp_writer)
    writer.writerow(['subject_id', 'trial_number', 'room_by_order', 'room_by_color', 'start_time', 'end_time',
                     'start_position_x', 'start_position_y', 'start_position_z', 'end_position_x', 'end_position_y',
                     'end_position_z', 'start_items_clicked', 'end_items_clicked', 'items_clicked_in_segment',
                     'time_in_segment', 'distance_in_segment', 'distance_over_time_in_segment'])

    def compute_metrics(lines):
        first_line = lines[0]
        last_line = lines[-1]
        subject_id = first_line[0]
        trial_number = first_line[1]
        room_by_order = first_line[6]
        room_by_color = first_line[7]
        start_time = first_line[2]
        end_time = last_line[2]
        time_in_segment = float(end_time) - float(start_time)
        start_position_x = first_line[3]
        start_position_y = first_line[4]
        start_position_z = first_line[5]
        end_position_x = last_line[3]
        end_position_y = last_line[4]
        end_position_z = last_line[5]
        start_items_clicked = first_line[8]
        end_items_clicked = last_line[8]
        items_clicked_in_segment = float(end_items_clicked) - float(start_items_clicked)

        distance_in_segment = 0
        for l in lines:
            distance_in_segment += float(l[9])

        distance_over_time_in_segment = float(distance_in_segment) / float(time_in_segment)

        return [subject_id, trial_number, room_by_order, room_by_color, start_time, end_time,
                start_position_x, start_position_y, start_position_z, end_position_x, end_position_y, end_position_z,
                start_items_clicked, end_items_clicked,
                items_clicked_in_segment, time_in_segment, distance_in_segment, distance_over_time_in_segment]

    logging.info("Computing/saving metrics for %d segments." % len(data_dict))

    for segment_id in data_dict:
        writer.writerow(compute_metrics(data_dict[segment_id]))

    logging.info("Closing output file.")

    fp_writer.close()

    logging.info("Done!")


def generate_segmentation_analysis_command_line():
    """
    This function can be called from a python script to wrap the generate_segmentation_analysis function in command
    line arguments which will be parsed and passed to the command.
    """
    # Parse inputs
    parser = argparse.ArgumentParser(
        description='This script will process a test*.csv file (first input arg) and generate segmentation metrics' +
                    '(triples) for the various trials of data. The data ' +
                    'will be saved in the output_path parameter (second input arg) in csv format.')
    parser.add_argument('path', help='The full input path to the test*.csv file generated by ' +
                                     'Holodeck_GenerateIntermediateFiles.py that you wish to process.')
    parser.add_argument('output_path',
                        help='The full input output path (ending with .csv) of the file to be generated ' +
                             'by this script containing the expected output fields formatted as csv.')

    # parser.add_argument('--point_speed', default=100, type=int,
    #                     help='The number of points to be drawn per 33ms iteration.')
    # parser.set_defaults(point_speed=100)

    parser.add_argument('--log_level', default=20, type=int,
                        help='Logging level of the application (default=20/INFO). ' +
                             'See https://docs.python.org/2/library/logging.html#levels for more info.')
    parser.set_defaults(log_level=20)

    args = parser.parse_args()

    generate_segmentation_analysis(args.path, args.output_path, args.log_level)


def generate_segmentation_analysis(input_path, output_path):
    """
    This function will generate a segmentation analysis given the intermediate files from generate_intermediate_files.

    :param input_path: the path to the intermediate file
    :param output_path: the output path to which to save the metrics

    :return: Nothing
    """

    # These values represent the context-triples set of for testing segmentation effects (and labels for convenience)
    # noinspection PyUnusedLocal
    triples_labels = ["red->green", "green->yellow", "yellow->blue", "blue->red"]
    context_crossing_triples_indicies = [(7, 8), (15, 14), (6, 0), (2, 4)]
    noncontext_crossing_triples_indicies = [(11, 7), (13, 15), (5, 6), (1, 2)]

    logging.info("Done parsing command line arguments.")

    fp_reader = open(input_path, 'rb')
    reader = csv.reader(fp_reader)
    # noinspection PyUnusedLocal
    header = reader.next()

    data_dict = dict()

    for line in reader:
        segment_id = line[0] + line[1]
        if segment_id in data_dict:
            data_dict[segment_id].append(line)
        else:
            data_dict[segment_id] = [line]

    logging.info("Done parsing input files into segments. %d segments found. Closing input file." % len(data_dict))

    fp_reader.close()

    if len(data_dict) == 0:
        logging.error("Error: No segments found in input file. Are you sure this is a properly formatted input file?")
        exit()

    logging.info("Creating output file %s." % output_path)

    fp_writer = open(output_path, 'wb')
    writer = csv.writer(fp_writer)
    writer.writerow(['subject_id', 'trial_number', 'context_crossing_class',
                     'first_item_name', 'first_item_order_clicked_study',
                     'first_item_expected_order_room', 'first_item_expected_color_room',
                     'first_item_actual_order_room', 'first_item_actual_color_room',
                     'second_item_name', 'second_item_order_clicked_study',
                     'second_item_expected_order_room', 'second_item_expected_color_room',
                     'second_item_actual_order_room', 'second_item_actual_color_room',
                     'actual_distance', 'expected_distance', 'normed_distance'])

    logging.info("Computing/saving metrics for %d segments." % len(data_dict))

    def compute_metrics(lines):
        out_lines = []

        is_study = (lines[0][2] in log_parser.study_labels)

        for pair in context_crossing_triples_indicies:
            if is_study:
                labels_to_use = log_parser.study_labels
            else:
                labels_to_use = log_parser.test_labels
            first_pair_name = labels_to_use[pair[0]]
            second_pair_name = labels_to_use[pair[1]]
            first_pair_loc_actual = None
            first_pair_loc_expected = None
            second_pair_loc_actual = None
            second_pair_loc_expected = None
            first_line = None
            second_line = None
            for l in lines:
                if l[2] == first_pair_name:
                    first_line = l
                    first_pair_loc_actual = (float(l[3]), float(l[4]))
                    first_pair_loc_expected = (float(l[5]), float(l[6]))
                if l[2] == second_pair_name:
                    second_line = l
                    second_pair_loc_actual = (float(l[3]), float(l[4]))
                    second_pair_loc_expected = (float(l[5]), float(l[6]))

            if not first_pair_loc_actual or not first_pair_loc_expected or not second_pair_loc_actual or \
                    not second_pair_loc_expected:
                logging.error("Item pair not found in Subject %s, Trial %s, Items %s and %s." % (
                    l[0], l[1], first_pair_name, second_pair_name))

            actual_distance = distance.euclidean(first_pair_loc_actual, second_pair_loc_actual)
            expected_distance = distance.euclidean(first_pair_loc_expected, second_pair_loc_expected)

            normed_distance = actual_distance / expected_distance

            out_lines.append(
                [first_line[0], first_line[1], 'context-crossing', first_line[2], first_line[7], first_line[8],
                 first_line[9], first_line[10], first_line[11], second_line[2], second_line[7], second_line[8],
                 second_line[9], second_line[10], second_line[11], actual_distance, expected_distance,
                 normed_distance])

        for pair in noncontext_crossing_triples_indicies:
            if is_study:
                labels_to_use = log_parser.study_labels
            else:
                labels_to_use = log_parser.test_labels
            first_pair_name = labels_to_use[pair[0]]
            second_pair_name = labels_to_use[pair[1]]
            first_pair_loc_actual = None
            first_pair_loc_expected = None
            second_pair_loc_actual = None
            second_pair_loc_expected = None
            first_line = None
            second_line = None
            for l in lines:
                if l[2] == first_pair_name:
                    first_line = l
                    first_pair_loc_actual = (float(l[3]), float(l[4]))
                    first_pair_loc_expected = (float(l[5]), float(l[6]))
                if l[2] == second_pair_name:
                    second_line = l
                    second_pair_loc_actual = (float(l[3]), float(l[4]))
                    second_pair_loc_expected = (float(l[5]), float(l[6]))

            if not first_pair_loc_actual or not first_pair_loc_expected or not second_pair_loc_actual or \
                    not second_pair_loc_expected:
                logging.error("Item pair not found in Subject %s, Trial %s, Items %s and %s." % (
                    l[0], l[1], first_pair_name, second_pair_name))

            actual_distance = distance.euclidean(first_pair_loc_actual, second_pair_loc_actual)
            expected_distance = distance.euclidean(first_pair_loc_expected, second_pair_loc_expected)

            normed_distance = actual_distance / expected_distance

            out_lines.append(
                [first_line[0], first_line[1], 'non-context-crossing', first_line[2], first_line[7], first_line[8],
                 first_line[9], first_line[10], first_line[11], second_line[2], second_line[7], second_line[8],
                 second_line[9], second_line[10], second_line[11], actual_distance, expected_distance,
                 normed_distance])

        return out_lines

    for segment_id in data_dict:
        metrics = compute_metrics(data_dict[segment_id])
        writer.writerows(metrics)

    logging.info("Closing output file.")

    fp_writer.close()

    logging.info("Done!")


def convert_to_iposition(input_path, output_path, expected_number_of_trials=4):
    """
    This function takes a path to a test intermediate file and generates a directory of iPosition formatted files.

    Note: The correct coordinates are from log_parser.study_realX, log_parser.study_realY, thus the assumption is made
    that this will only be run on the VR form of the task (not the 2D form), or that the 2D form data will be converted
    to the coordinate system of the VR task.

    :param expected_number_of_trials: the expected number of trials to output (error will throw if number exceeds
                                      expectation, but no error will occur if the data is incomplete)
    :param output_path: the output directory into which to save the iposition files (will be created if doesn't exist)
    :param input_path: the path to the test intermediate file
    """

    # Read the file
    fp_reader = open(input_path, 'rb')
    reader = csv.reader(fp_reader)
    # noinspection PyUnusedLocal
    header = reader.next()

    if not os.path.exists(output_path):
        os.mkdir(os.path.abspath(output_path))

    write_data = {}
    order_data = {}
    category_data = {}
    # Iterate through trial/subid data
    for line in reader:
        sub_id = line[0]
        if sub_id not in write_data:
            write_data[sub_id] = [[[] for _ in range(len(log_parser.study_labels))]
                                  for _ in range(expected_number_of_trials)]
        if sub_id not in order_data:
            order_data[sub_id] = [[] for _ in range(expected_number_of_trials)]
        if sub_id not in category_data:
            category_data[sub_id] = [[0 for _ in range(len(log_parser.study_labels))]
                                     for _ in range(expected_number_of_trials)]
        # Iterate through items
        # Find the index of the items in the lut and placed coordinates
        trial = line[1]
        item_idx = log_parser.study_labels.index(line[2])
        x = line[3]
        y = line[4]
        room_color = line[11]
        category_number = log_parser.context_labels.index(room_color)
        # Add them to the appropriate output section
        if int(trial) > expected_number_of_trials:
            logging.error('trial number {0} exceeds expected number {1} on '
                          'participant {2}.'.format(int(trial), expected_number_of_trials, sub_id))
        if item_idx > len(log_parser.study_labels):
            logging.error('item with name {0} found index {1} which exceeds the length of the labels ({2}) on '
                          'participant {3}.'.format(line[2], item_idx, len(log_parser.study_labels), sub_id))
        write_data[sub_id][int(trial)][item_idx] = [x, y]
        order_data[sub_id][int(trial)].append(item_idx)
        category_data[sub_id][int(trial)][item_idx] = category_number
        # Flatten the trial coordinates and tab separate them in the proper order
    for sub_id in write_data:
        out_lines = []
        coords = write_data[sub_id]
        for trial_coords in coords:
            flattened_trial_coords = [item for sublist in trial_coords for item in sublist]
            out_lines.append('\t'.join([str(_a) for _a in flattened_trial_coords]) + '\r\n')
        # Write the position data to file
        with open(os.path.abspath(os.path.join(output_path, '{0}{1}'.format(sub_id, data_coordinates_file_suffix))),
                  'wb') as fp:
            fp.writelines(out_lines)

    for sub_id in order_data:
        out_lines = []
        coords = order_data[sub_id]
        for trial_coords in coords:
            out_lines.append('\t'.join([str(_a) for _a in trial_coords]) + '\r\n')
        # Write the position data to file
        with open(os.path.abspath(os.path.join(output_path, '{0}{1}'.format(sub_id, order_file_suffix))),
                  'wb') as fp:
            fp.writelines(out_lines)

    for sub_id in category_data:
        out_lines = []
        coords = category_data[sub_id]
        for trial_coords in coords:
            out_lines.append('\t'.join([str(_a) for _a in trial_coords]) + '\r\n')
        # Write the position data to file
        with open(os.path.abspath(os.path.join(output_path, '{0}{1}'.format(sub_id, category_file_suffix))),
                  'wb') as fp:
            fp.writelines(out_lines)

    with open(os.path.abspath(os.path.join(output_path, actual_coordinates_file_suffix)), 'wb') as fp:
        actual_coordinates = [item for sublist in
                              np.transpose([log_parser.study_realX, log_parser.study_realY]).tolist()
                              for item in sublist]
        for _ in range(expected_number_of_trials):
            fp.write('\t'.join([str(_a) for _a in actual_coordinates]) + '\r\n')

    categories = [0 for _ in range(len([item for sublist in log_parser.context_item_indicies for item in sublist]))]
    for cat, idxs in enumerate(log_parser.context_item_indicies):
        for idx in idxs:
            categories[idx] = cat

    with open(os.path.abspath(os.path.join(output_path, category_file_suffix)), 'wb') as fp:
        for _ in range(expected_number_of_trials):
            fp.write('\t'.join([str(_a) for _a in categories]) + '\r\n')

    fp_reader.close()


if __name__ == "__main__":
    import os
    import cogrecon.core.data_flexing.spatial_navigation.spatial_navigation_parser as parser
    parser.study_labels = ['PurseCube', 'CrownCube', 'BasketballCube', 'BootCube', 'CloverCube', 'GuitarCube',
                           'HammerCube', 'LemonCube', 'IceCubeCube', 'BottleCube']
    directory = r'Z:\Kelsey\2017 Summer RetLu\Virtual_Navigation_Task\v5_2\NavigationTask_Data\Logged_Data\2RoomTestAnonymous'
    raw_filepath = os.path.join(directory, 'RawLog_Sub124_Trial1_13_15_57_30-05-2017.csv')
    summary_filepath = os.path.join(directory, 'SummaryLog_Sub124_Trial1_13_15_57_30-05-2017.csv')
    generate_intermediate_files(directory, full_study_path=False, full_study_look=False, full_test_path=False,
                                           full_test_look=False, full_practice_path=False, full_practice_look=False,
                                           full_test_2d=False, full_test_vr=True, min_num_trials=4)