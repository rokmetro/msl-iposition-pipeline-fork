# noinspection PyUnresolvedReferences
import argparse
import datetime
# noinspection PyUnresolvedReferences
import logging
# noinspection PyUnresolvedReferences
import os
# noinspection PyUnresolvedReferences
import sys
# noinspection PyCompatibility
from tkFileDialog import askdirectory

try:
    import tkinter as tk
except ImportError:
    import Tkinter as tk

import numpy as np

import types

from .full_pipeline import full_pipeline, get_aggregation_functions, get_header_labels
from .file_io import get_coordinates_from_file, get_id_from_file_prefix, find_data_files_in_directory
from .data_structures import TrialData, ParticipantData, AnalysisConfiguration, PipelineFlags

logging.basicConfig(level=logging.INFO)


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
    assert isinstance(require_numeric, types.BooleanType), "require_numeric is not a bool: {0}".format(require_numeric)
    assert isinstance(list_name, types.StringType), "list_name is not string: {0}".format(list_name)
    assert isinstance(l, types.ListType) or isinstance(l, types.TupleType) or isinstance(l, np.ndarray), \
        "{1} should be list or numpy array: {0}".format(l, list_name)
    if dimension:
        assert isinstance(dimension, types.IntType), "dimension is not an integer: {0}".format(dimension)
        assert dimension > 0, "dimension is not greater than 0: {0}".format(dimension)
        assert len(np.array(l).shape) == dimension, \
            ("{1} should be a 3d list or numpy array of form (Nt, Ni, d) where Nt is the number of " +
             "trials, Ni is the number of items, and d is the dimensionality of the data: {0}").format(l, list_name)
    if require_numeric:
        assert all(isinstance(x, int) or isinstance(x, float) for x in
                   np.ndarray.flatten(np.array(l))), "{1} contains some non int or float values: {0}".format(l,
                                                                                                             list_name)

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
    assert isinstance(l1_name, types.StringType), "l1_name is not string: {0}".format(l1_name)
    assert isinstance(l2_name, types.StringType), "l2_name is not string: {0}".format(l2_name)
    validate_list_format(l1)
    validate_list_format(l2)
    if expected_shape:
        validate_list_format(expected_shape, require_numeric=True, dimension=1)
        assert np.array(np.array(l1).shape) == np.array(expected_shape), \
            "{0} does not match expected shape: {1}".format(l1_name, expected_shape)
        assert np.array(np.array(l2).shape) == np.array(expected_shape), \
            "{0} does not match expected shape: {1}".format(l1_name, expected_shape)
    assert np.array(l1).shape == np.array(l2).shape, \
        ("shapes of {2} and {3} are not the same, " +
         "actual: {0}, data: {1}").format(np.shape(l1), np.shape(l2), l1_name, l2_name)

    return True


# threshold values for each process step
def get_single_file_result(actual_coordinates, dat, label="", accuracy_z_value=1.96,
                           trial_by_trial_accuracy=True, manual_threshold=None,
                           flags=PipelineFlags.All):
    """
    This function generates the results for a specific file's data structure, usually containing multiple trials

    :param manual_threshold: 
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

    assert isinstance(label, types.StringType), "label must be a string: {0}".format(label)
    assert isinstance(accuracy_z_value, types.IntType) or isinstance(accuracy_z_value, types.FloatType), \
        "accuracy_z_value must be int or float: {0}".format(accuracy_z_value)
    assert accuracy_z_value > 0, \
        "accuracy_z_value must be greater than 0: {0}".format(accuracy_z_value)
    assert isinstance(flags, PipelineFlags), \
        "flags is not of type PipelineFlags: {0}".format(flags)

    _analysis_configuration = AnalysisConfiguration(z_value=accuracy_z_value,
                                                    flags=flags,
                                                    debug_labels=[label],
                                                    trial_by_trial_accuracy=trial_by_trial_accuracy,
                                                    manual_threshold=manual_threshold)
    _participant_data = ParticipantData([TrialData(_a, _d) for _a, _d in zip(actual_coordinates, dat)])

    # Process the participant
    return full_pipeline(_participant_data, _analysis_configuration)


def detect_shape_from_file(path, dimension):
    """

    :rtype: int, int
    :param path: a value (string) containing the path of the file from which structure should be detected
    :param dimension: a value (integer) which represents the dimensionality of the data
    :return: the trial count, the item count
    """
    assert isinstance(path, types.StringType), 'path is not string: {0}'.format(path)
    assert os.path.exists(path), 'path does not exist: {0}'.format(path)

    with open(path) as tsv:
        trial_count = 0
        item_count_list = []
        for tsv_line in tsv:
            trial_count += 1
            item_count = 0
            split_line = tsv_line.strip().split('\t')
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
                   collapse_trials=True, dimension=2, prefix_length=3,
                   actual_coordinate_prefixes=False, manual_threshold=None):
    """
    This function allows the easy running of the pipeline on a directory and all of the appropriate files in its
    subdirectories. It will search for the actual coordinates and data files and process them all as specified
    by the other parameters.

    :param manual_threshold: 
    :param actual_coordinate_prefixes: 
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
    :param prefix_length: the number of characters at the beginning of the data filenames which constitute the
    subject ID (default is 3)
    """
    assert isinstance(search_directory, types.StringType), \
        "search_directory must be a string: {0}".format(search_directory)
    assert len(search_directory) > 0, "search_directory must have length greater than 0: {0}".format(search_directory)
    if data_shape:
        validate_list_format(data_shape, dimension=1, require_numeric=True, list_name="data_shape")
    assert isinstance(out_filename, types.StringType), "out_filename is not string: {0}".format(out_filename)
    assert len(out_filename) > 0, "out_filename must have length greater than 0: {0}".format(out_filename)
    assert isinstance(accuracy_z_value, types.IntType) or isinstance(accuracy_z_value, types.FloatType), \
        "accuracy_z_value must be int or float: {0}".format(accuracy_z_value)
    assert accuracy_z_value > 0, \
        "accuracy_z_value must be greater than 0: {0}".format(accuracy_z_value)
    assert isinstance(flags, PipelineFlags), \
        "flags is not of type PipelineFlags: {0}".format(flags)
    assert isinstance(collapse_trials, types.BooleanType), "collapse_trials is not a bool: {0}".format(collapse_trials)
    assert isinstance(trial_by_trial_accuracy, types.BooleanType), \
        "trial_by_trial_accuracy is not a bool: {0}".format(trial_by_trial_accuracy)

    logging.info('Finding files in folder {0}.'.format(search_directory))

    # Find the files
    actual_coordinates_filename = data_coordinates_filenames = None
    try:
        actual_coordinates_filename, data_coordinates_filenames = \
            find_data_files_in_directory(search_directory,
                                         actual_coordinate_prefixes=actual_coordinate_prefixes,
                                         prefix_length=prefix_length)
        data_coordinates_filenames = np.sort(data_coordinates_filenames)
    except IOError:
        logging.error('The input path was not found.')
        exit()

    logging.info('Parsing files with expected shape {0}.'.format(data_shape))

    # Parse the files
    if not actual_coordinate_prefixes:
        if data_shape is None:
            num_trials, num_items = detect_shape_from_file(actual_coordinates_filename, dimension)
            data_shape = (num_trials, num_items, dimension)
        actual_coordinates = get_coordinates_from_file(actual_coordinates_filename, data_shape)
        data_coordinates = [get_coordinates_from_file(filename, data_shape) for filename in data_coordinates_filenames]
    else:
        data_shapes = []
        for acf in actual_coordinates_filename:
            num_trials, num_items = detect_shape_from_file(acf, dimension)
            data_shapes.append((num_trials, num_items, dimension))
        actual_coordinates = [get_coordinates_from_file(filename, data_shapes[iidx]) for iidx, filename in
                              enumerate(actual_coordinates_filename)]
        data_coordinates = [get_coordinates_from_file(filename, data_shapes[iidx]) for iidx, filename in
                            enumerate(data_coordinates_filenames)]
    data_labels = [get_id_from_file_prefix(filename, prefix_length=prefix_length) for filename in
                   data_coordinates_filenames]
    logging.info('The following ids were found and are being processed: {0}'.format(data_labels))

    # Get the labels and aggregation methods
    agg_functions = get_aggregation_functions()
    header_labels = get_header_labels()

    # Add cross-trial labels and aggregation methods
    agg_functions.append(np.nansum)
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
        mt = None
        if manual_threshold is not None:
            for dat_id, dat_threshold in manual_threshold:
                if dat_id == label:
                    mt = dat_threshold
                    break
        if not actual_coordinate_prefixes:
            # Get results
            results = get_single_file_result(actual_coordinates, dat, label=label,
                                             accuracy_z_value=accuracy_z_value,
                                             flags=flags, trial_by_trial_accuracy=trial_by_trial_accuracy,
                                             manual_threshold=mt)
        else:
            assert np.array(actual_coordinates[index]).shape == np.array(
                dat).shape, "shape mismatch between {0} and {1}".format(actual_coordinates_filename[index],
                                                                        data_coordinates_filenames[index])
            results = get_single_file_result(actual_coordinates[index], dat, label=label,
                                             accuracy_z_value=accuracy_z_value,
                                             flags=flags, trial_by_trial_accuracy=trial_by_trial_accuracy,
                                             manual_threshold=mt)

        new_results = []
        # Append the across-trial variables
        # Look for NaNs
        for data_line in results:
            num_rows_with_nan = 0
            for item in data_line:
                if item is np.nan:
                    num_rows_with_nan += 1
            new_results.append(np.append(data_line, [num_rows_with_nan]))
        results = new_results

        if collapse_trials:
            # Apply the aggregation function to each value
            result = []
            for iidx in range(len(results[0])):
                result.append(agg_functions[iidx]([row[iidx] for row in results]))

            # Write to file
            out_fp.write(
                '{0},{1}\n'.format(
                    label,
                    ','.join(['"{0}"'.format(str(r)) if ',' in str(r) else str(r) for r in result]))  # Filter commas
            )
        else:
            for iidx, row in enumerate(results):
                out_fp.write(
                    '{0},{1},{2}\n'.format(label,
                                           iidx,
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
    parser.add_argument('--prefix_length', type=int, help='the length of the subject ID prefix at the beginning of '
                                                          'the data filenames (default is 3)', default=3)
    parser.add_argument('--actual_coordinate_prefixes', type=int, help='if 0, the normal assumption that all '
                                                                       'participants used the same '
                                                                       'actual_coordinates.txt file will be used. if '
                                                                       'not 0, it is assumed that all '
                                                                       'actual_coordinates.txt files have a prefix '
                                                                       'which is matched in the '
                                                                       'position_data_coordinates.txt prefix. Thus, '
                                                                       'there should be a one-to-one correspondance '
                                                                       'between actual_coordinates.txt and '
                                                                       'position_data_coordinates.txt files and their '
                                                                       'contents.', default=0)
    parser.add_argument('--manual_swap_accuracy_threshold_list', type=str,
                        help='if empty string or none, the value is ignored. if a string (path) pointing to a text '
                             'file containing a new line separated list of id,threshold pairs is provided, '
                             'any files whose participant id matches the first matching id in the list will have the '
                             'associated threshold applied instead of being automatically computed.',
                        default='')
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
        manual_swap_accuracy_threshold_list = None
        if args.manual_swap_accuracy_threshold_list is not None:
            # noinspection PyBroadException
            try:
                with open(args.manual_swap_accuracy_threshold_list) as f:
                    lis = [line.split(',') for line in f]
                    for idx, (_, d_threshold) in enumerate(lis):
                        lis[idx][1] = float(d_threshold)
                    manual_swap_accuracy_threshold_list = lis
            except:
                logging.warning(
                    'the provided manual_swap_accuracy_threshold_list was either not found or invalid - it will be '
                    'skipped')
        batch_pipeline(selected_directory,
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"),
                       data_shape=d_shape,
                       accuracy_z_value=args.accuracy_z_value,
                       trial_by_trial_accuracy=args.trial_by_trial_accuracy != 0,
                       flags=PipelineFlags(args.pipeline_mode),
                       collapse_trials=args.collapse_trials != 0,
                       dimension=args.dimension,
                       prefix_length=args.prefix_length,
                       actual_coordinate_prefixes=args.actual_coordinate_prefixes,
                       manual_threshold=manual_swap_accuracy_threshold_list)
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
