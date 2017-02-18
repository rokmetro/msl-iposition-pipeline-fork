# noinspection PyUnresolvedReferences
import argparse
import datetime
# noinspection PyUnresolvedReferences
import logging
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import sys
import time
# noinspection PyCompatibility
from tkFileDialog import askdirectory

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

import numpy

from types import *
from full_pipeline import *

logging.basicConfig(level=logging.INFO)


def find_data_files_in_directory(directory):
    """
    This function crawls the specified directory, recursively looking for the actual coordinate file and data files

    :rtype: string (or None), list of strings (or empty list)
    :param directory: the directory (string) in which to recursively search for data files
    :return: the actual coordinate filename/path (None if no file was found), a list of the data filenames/paths
    (empty list if no files were found)
    """
    assert type(directory) is StringType, "directory is not a string: {0}".format(directory)

    if not os.path.exists(directory):
        raise IOError('The input path was not found.')

    start_time = time.time()
    data_files = []
    actual_coordinate_file = None
    actual_coordinate_contents = None
    for root, dirs, files in os.walk(directory):
        for f in files:
            if os.path.basename(f) == "actual_coordinates.txt":  # If we find an actual coordinate file
                if actual_coordinate_file is None:  # And we haven't found a coordinate file before
                    actual_coordinate_file = os.path.join(root, f)  # Set the coordinate file
                    with open(actual_coordinate_file) as fp:  # Save its contents
                        actual_coordinate_contents = fp.read()
                    logging.debug('Found actual_coordinates.txt ({0}).'.format(actual_coordinate_file))
                else:  # If we have found an additional coordinate file
                    with open(os.path.join(root, f)) as fp:  # Get its contents
                        new_contents = fp.read()
                    if new_contents != actual_coordinate_contents:  # Compare its contents to the first found file
                        # If the contents are not the same, quit - because we don't know which to use.
                        logging.error(('Found multiple actual_coordinates.txt with different contents, ' +
                                       'program will now exit (found {0}).').format(actual_coordinate_file))
                        exit()
                    else:  # Otherwise continue and warn the user
                        logging.warning('Found multiple actual_coordinates.txt but contents were identical.')
            if f.endswith("position_data_coordinates.txt"):  # If we find a data file, save it to the file list
                logging.debug('Found data file ({0}).'.format(f))
                data_files.append(os.path.join(root, f))
    logging.info('Found {0} data files in {1} seconds and {2} actual coordinate file.'.format(len(data_files),
                                                                                              time.time() - start_time,
                                                                                              actual_coordinate_file))

    return actual_coordinate_file, data_files


def validate_list_format(l, require_numeric=False, dimension=None, list_name="list"):
    """
    This function validates that a list is the correct type, dimension, and
    contains only int and float values (if specified).

    :param l: a list whose type, dimensionality, and contents should be checked; valid types are list, tuple, or
    numpy array, dimensionality should match dimension, and the contents should all be int or float
    :param require_numeric: (optional) if True, elements must be int or float
    :param dimension: (optional) the expected number of dimensions (integer greater than 0) of the list
    (default is None, meaning it is not checked)
    :param list_name: (optional) the name (string) of the list for debugging purposes (default is "list")
    """
    assert type(require_numeric) is BooleanType, "require_numeric is not a bool: {0}".format(require_numeric)
    assert type(list_name) is StringType, "list_name is not string: {0}".format(list_name)
    assert type(l) is ListType or type(l) is TupleType or type(l) is ndarray, \
        "{1} should be list or numpy array: {0}".format(l, list_name)
    if dimension:
        assert type(dimension) is IntType, "dimension is not an integer: {0}".format(dimension)
        assert dimension > 0, "dimension is not greater than 0: {0}".format(dimension)
        assert len(array(l).shape) == dimension, \
            ("{1} should be a 3d list or numpy array of form (Nt, Ni, d) where Nt is the number of " +
             "trials, Ni is the number of items, and d is the dimensionality of the data: {0}").format(l, list_name)
    if require_numeric:
        assert all(isinstance(x, int) or isinstance(x, float) for x in
                   ndarray.flatten(array(l))), "{1} contains some non int or float values: {0}".format(l, list_name)

    return True


def validate_equal_list_shapes(l1, l2, expected_shape=None, l1_name="list1", l2_name="list2"):
    """
    This function validates that two numeric
    :param l1: a list, tuple, or numpy array whose shape should be equal to the shape of l1
    and expected_shape (if specified)
    :param l2: a list, tuple, or numpy array whose shape should be equal to the shape of l2
    and expected_shape (if specified)
    :param expected_shape: (optional) a shape (list, tuple or numpy array) against which both l1 and l2 should be
    compared to ensure they are equal to it and each other
    :param l1_name: (optional) the name (string) of l1 for debugging
    :param l2_name: (optional) the name (string) of l2 for debugging
    """
    assert type(l1_name) is StringType, "l1_name is not string: {0}".format(l1_name)
    assert type(l2_name) is StringType, "l2_name is not string: {0}".format(l2_name)
    validate_list_format(l1)
    validate_list_format(l2)
    if expected_shape:
        validate_list_format(expected_shape, require_numeric=True, dimension=1)
        assert array(array(l1).shape) == array(expected_shape), \
            "{0} does not match expected shape: {1}".format(l1_name, expected_shape)
        assert array(array(l2).shape) == array(expected_shape), \
            "{0} does not match expected shape: {1}".format(l1_name, expected_shape)
    assert array(l1).shape == array(l2).shape, \
        ("shapes of {2} and {3} are not the same, " +
         "actual: {0}, data: {1}").format(shape(l1), shape(l2), l1_name, l2_name)

    return True


# threshold values for each process step
def get_single_file_result(actual_coordinates, dat, label="", accuracy_z_value=1.96, trial_by_trial_accuracy=True,
                           flags=PipelineFlags.All):
    """
    This function generates the results for a specific file's data structure, usually containing multiple trials

    :rtype: list (or empty list)
    :param actual_coordinates: the correct coordinates for the points - an (Nt, Ni, d) sized list of floats where Nt is
    the number of trials, Ni is the number of items, and d is the dimensionality of the points
    :param dat: the data coordinates for the points - an (Nt, Ni, d) sized list of floats where Nt is the number of
    trials, Ni is the number of items, and d is the dimensionality of the points
    :param label: (optional) the label (string) identifying the participant ID for this file, used for debugging
    purposes only (default is empty string)
    :param accuracy_z_value: (optional) a value (float or int) representing the z threshold for counting something as
    :param trial_by_trial_accuracy: (optional) when True, z_value thresholds are used on a trial-by-trial basis for
    accuracy calculations, when False, the thresholds are computed then collapsed across an individual's trials
    (default is True)
    :param flags: (optional) the value (PipelineFlags) describing what pipeline elements should/should not be run on
    the data (default is PipelineFlags.All)
    :return: a list, (Nt, r), where Nt is the number of trials and r is the number of result metrics, of results values
    from the analysis for each trial on a particular file's data
    """
    validate_list_format(actual_coordinates, require_numeric=True, dimension=3, list_name="actual_coordinates")
    validate_list_format(dat, require_numeric=True, dimension=3, list_name="dat")

    validate_equal_list_shapes(actual_coordinates, dat, l1_name="actual_coordinates", l2_name="dat")

    assert type(label) is StringType, "label must be a string: {0}".format(label)
    assert type(accuracy_z_value) is IntType or type(accuracy_z_value) is FloatType, \
        "accuracy_z_value must be int or float: {0}".format(accuracy_z_value)
    assert accuracy_z_value > 0, \
        "accuracy_z_value must be greater than 0: {0}".format(accuracy_z_value)
    assert isinstance(flags, PipelineFlags), \
        "flags is not of type PipelineFlags: {0}".format(flags)

    # Process the participant
    return full_pipeline(actual_coordinates, dat,
                         accuracy_z_value=accuracy_z_value,
                         trial_by_trial_accuracy=trial_by_trial_accuracy,
                         flags=flags,
                         debug_labels=[label])


def detect_shape_from_file(path, dimension):
    """

    :rtype: int, int
    :param path: a value (string) containing the path of the file from which structure should be detected
    :param dimension: a value (integer) which represents the dimensionality of the data
    :return: the trial count, the item count
    """
    assert type(path) is StringType, 'path is not string: {0}'.format(path)
    assert os.path.exists(path), 'path does not exist: {0}'.format(path)

    with open(path) as tsv:
        trial_count = 0
        item_count_list = []
        for line in tsv:
            trial_count += 1
            item_count = 0
            split_line = line.strip().split('\t')
            for _ in split_line:
                item_count += 1
            item_count_list.append(item_count)
        assert len(item_count_list) > 0, 'no items detected in file: {0}'.format(path)
        assert all(x == item_count_list[0] for x in item_count_list), \
            'inconsistent item count detected in file ({1}): {0}'.format(path, item_count_list)
        assert trial_count > 0, "no trials detected: {0}".format(path)
        assert item_count_list[0] > 0, "no items detected".format(path)

        return trial_count, int(float(item_count_list[0]) / float(dimension))


def batch_pipeline(search_directory, out_filename, data_shape=None, accuracy_z_value=1.96,
                   trial_by_trial_accuracy=True,
                   flags=PipelineFlags.All,
                   collapse_trials=True, dimension=2):
    """
    This function allows the easy running of the pipeline on a directory and all of the appropriate files in its
    subdirectories. It will search for the actual coordinates and data files and process them all as specified
    by the other parameters.

    :rtype: None
    :param search_directory: the directory (string) in which to recursively search for data files
    :param data_shape: (optional) a shape (list, tuple or numpy array) which describes the structure of the date;
    (Nt, Ni, d) such that Nt is the number of trials, Ni is the number of items and d is the number of dimensions;
    if None is given, an attempt will be made to automatically detect the shape from the actual_coordinates file
    (default is None)
    :param out_filename: the filename and path (string) into which the data should be saved
    :param accuracy_z_value: (optional) a value (float or int) representing the z threshold for counting something as
    accurate (default is 1.96, i.e. 95% confidence interval)
    :param trial_by_trial_accuracy: (optional) when True, z_value thresholds are used on a trial-by-trial basis for
    accuracy calculations, when False, the thresholds are computed then collapsed across an individual's trials
    (default is True)
    :param flags: (optional) the value (PipelineFlags) describing what pipeline elements should/should not be run on
    the data (default is PipelineFlags.All)
    :param collapse_trials: (optional) if True, the output file will contain one row per participant, otherwise each
    trial will be output in an individual row
    :param dimension: (optional) the dimensionality of the data (default is 2)
    """
    assert type(search_directory) is StringType, "search_directory must be a string: {0}".format(search_directory)
    assert len(search_directory) > 0, "search_directory must have length greater than 0: {0}".format(search_directory)
    if data_shape:
        validate_list_format(data_shape, dimension=1, require_numeric=True, list_name="data_shape")
    assert type(out_filename) is StringType, "out_filename is not string: {0}".format(out_filename)
    assert len(out_filename) > 0, "out_filename must have length greater than 0: {0}".format(out_filename)
    assert type(accuracy_z_value) is IntType or type(accuracy_z_value) is FloatType, \
        "accuracy_z_value must be int or float: {0}".format(accuracy_z_value)
    assert accuracy_z_value > 0, \
        "accuracy_z_value must be greater than 0: {0}".format(accuracy_z_value)
    assert isinstance(flags, PipelineFlags), \
        "flags is not of type PipelineFlags: {0}".format(flags)
    assert type(collapse_trials) is BooleanType, "collapse_trials is not a bool: {0}".format(collapse_trials)
    assert type(trial_by_trial_accuracy) is BooleanType, \
        "trial_by_trial_accuracy is not a bool: {0}".format(trial_by_trial_accuracy)

    logging.info('Finding files in folder {0}.'.format(search_directory))

    # Find the files
    actual_coordinates_filename = data_coordinates_filenames = None
    try:
        actual_coordinates_filename, data_coordinates_filenames = find_data_files_in_directory(search_directory)
        data_coordinates_filenames = sort(data_coordinates_filenames)
    except IOError:
        logging.error('The input path was not found.')
        exit()

    logging.info('Parsing files with expected shape {0}.'.format(data_shape))

    if data_shape is None:
        num_trials, num_items = detect_shape_from_file(actual_coordinates_filename, dimension)
        data_shape = (num_trials, num_items, dimension)

    # Parse the files
    actual_coordinates = get_coordinates_from_file(actual_coordinates_filename, data_shape)
    data_coordinates = [get_coordinates_from_file(filename, data_shape) for filename in data_coordinates_filenames]
    data_labels = [get_id_from_file_prefix(filename) for filename in data_coordinates_filenames]
    logging.info('The following ids were found and are being processed: {0}'.format(data_labels))

    # Get the labels and aggregation methods
    agg_functions = get_aggregation_functions()
    header_labels = get_header_labels()

    # Add cross-trial labels and aggregation methods
    agg_functions.append(nansum)
    header_labels.append('num_rows_with_nan')

    # Generate the output file and write the header
    out_fp = open(out_filename, 'w')
    if collapse_trials:
        header = "subID,{0}\n".format(','.join(header_labels))
    else:
        header = "subID,trial,{0}\n".format(','.join(header_labels))

    out_fp.write(header)

    # Iterate through the participants
    for index, (dat, label) in enumerate(zip(data_coordinates, data_labels)):
        logging.debug('Parsing {0}.'.format(label))
        # Get results
        results = get_single_file_result(actual_coordinates, dat, label=label,
                                         accuracy_z_value=accuracy_z_value,
                                         flags=flags, trial_by_trial_accuracy=trial_by_trial_accuracy)

        new_results = []
        # Append the across-trial variables
        # Look for NaNs
        for line in results:
            num_rows_with_nan = 0
            for item in line:
                if item is numpy.nan:
                    num_rows_with_nan += 1
            new_results.append(append(line, [num_rows_with_nan]))
        results = new_results

        if collapse_trials:
            # Apply the aggregation function to each value
            result = []
            for idx in range(len(results[0])):
                result.append(agg_functions[idx]([row[idx] for row in results]))

            # Write to file
            out_fp.write(
                '{0},{1}\n'.format(
                    label,
                    ','.join(['"{0}"'.format(str(r)) if ',' in str(r) else str(r) for r in result]))  # Filter commas
            )
        else:
            for idx, row in enumerate(results):
                out_fp.write(
                    '{0},{1},{2}\n'.format(label,
                                           idx,
                                           ','.join(['"{0}"'.format(str(r)) if ',' in str(r) else str(r) for r in row]))
                )

    out_fp.close()

    logging.info('Done processing all files. Data can be found in {0}.'.format(out_filename))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    parser.add_argument('--search_directory', type=str, help='the root directory in which to search for the actual and '
                                                             'data coordinate files (actual_coordinates.txt and '
                                                             '###position_data_coordinates.txt, respectively)',
                        default=None)
    parser.add_argument('--num_trials', type=int, help='the number of trials in each file', default=None)
    parser.add_argument('--num_items', type=int, help='the number of items to be analyzed', default=None)
    parser.add_argument('--pipeline_mode', type=int, help='the mode in which the pipeline should process; \n\t0 for '
                                                          'just accuracy+swaps, \n\t1 for '
                                                          'accuracy+deanonymization+swaps, \n\t2 for accuracy+global '
                                                          'transformations+swaps, \n\t3 for '
                                                          'accuracy+deanonymization+global transformations+swaps \n('
                                                          'default is 3)', default=3)
    parser.add_argument('--accuracy_z_value', type=float, help='the z value to be used for accuracy exclusion ('
                                                               'default is 1.96, corresponding to 95% confidence',
                        default=1.96)
    parser.add_argument('--collapse_trials', type=int, help='if 0, one row per trial will be output, otherwise one '
                                                            'row per participant will be output (default is 1)',
                        default=1)
    parser.add_argument('--dimension', type=int, help='the dimensionality of the data (default is 2)', default=2)
    parser.add_argument('--trial_by_trial_accuracy', type=int, help='when not 0, z_value thresholds are used on a '
                                                                    'trial-by-trial basis for accuracy calculations, '
                                                                    'when 0, the thresholds are computed then '
                                                                    'collapsed across an individual\'s trials',
                        default=1)
    if len(sys.argv) > 1:
        args = parser.parse_args()
        if args.search_directory is None:
            tk_root = tk.Tk()
            tk_root.withdraw()
            selected_directory = str(askdirectory())
        else:
            selected_directory = args.search_directory
        if len(selected_directory) == 0:  # Gracefully exit if cancel is clicked
            exit()
        if not args.num_trials or not args.num_items:
            logging.warning('Either num_items or num_trials was not provided. The data shape will be automatically ' +
                            'detected from the actual coordinates.')
            d_shape = None
        else:
            d_shape = (args.num_trials, args.num_items, args.dimension)
        batch_pipeline(selected_directory,
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"),
                       data_shape=d_shape,
                       accuracy_z_value=args.accuracy_z_value,
                       trial_by_trial_accuracy=args.trial_by_trial_accuracy != 0,
                       flags=PipelineFlags(args.pipeline_mode),
                       collapse_trials=args.collapse_trials != 0)
        exit()

    logging.info("No arguments found - assuming running in test mode.")

    tk_root = tk.Tk()
    tk_root.withdraw()
    selected_directory = str(askdirectory())

    if os.path.exists(selected_directory):
        batch_pipeline(selected_directory, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"))
    elif selected_directory is not '':
        logging.error('Directory not found.')
        exit()
    else:
        exit()
        # batch_pipeline("Z:\\Kevin\\iPosition\\Hillary\\MRE",
        #                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"),
        #                data_shape=(15, 5, 2))
