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

import numpy

from types import *
from full_pipeline import *

logging.basicConfig(level=logging.INFO)

# TODO: Documentation needs an audit/overhaul


def find_data_files_in_directory(directory):
    """
    This function crawls the specified directory, recursively looking for the actual coordinate file and data files

    :rtype: string (or None), list of strings (or empty list)
    :param directory: the directory in which to recursively search for data files
    :return: the actual coordinate filename/path (None if no file was found), a list of the data filenames/paths (empty list if no files were found)
    """

    assert type(directory) is StringType, "directory is not a string: {0}".format(str(directory))

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


def get_single_file_result(actual_coordinates, dat, label="", accuracy_z_value=1.96, flags=PipelineFlags.All):
    """
    This function generates the results for a specific file's data structure, usually containing multiple trials

    :rtype: list (or empty list)
    :param actual_coordinates: the correct coordinates for the points - an (Nt, Ni, d) sized list of floats where Nt is the number of trials, Ni is the number of items, and d is the dimensionality of the points
    :param dat: the data coordinates for the points - an (Nt, Ni, d) sized list of floats where Nt is the number of trials, Ni is the number of items, and d is the dimensionality of the points
    :param label: the label identifying the participant ID for this file, used for debugging purposes only (default is empty string)
    :param accuracy_z_value: a float representing the z threshold for counting something as accurate (default is 1.96, i.e. 95% confidence interval)
    :param flags: the PipelineFlags describing what pipeline elements should/should not be run on the data (default is PipelineFlags.All)
    :return: a list, (Nt, r), where Nt is the number of trials and r is the number of result metrics, of results values from the analysis for each trial on a particular file's data
    """

    assert type(actual_coordinates) is ListType or type(actual_coordinates) is type(array()), "actual_coordinates should be list or numpy array: {0}".format(str(actual_coordinates))
    assert type(dat) is ListType or type(dat) is type(array()), "dat should be list or numpy array: {0}".format(str(actual_coordinates))
    assert shape(array(actual_coordinates)) == 3, "actual_coordinates should be a 3d list or numpy array of form (Nt, Ni, d) where Nt is the number of trials, Ni is the number of items, and d is the dimensionality of the data: {0}".format(actual_coordinates)
    assert all(isinstance(x, int) or isinstance(x, float) for x in ndarray.flatten(array(actual_coordinates))), "actual_coodinates contains some non int or float values"
    assert shape(actual_coordinates) == shape(data), "shapes of actual_coordinates and data are not the same, actual: {0}, data: {1}".format(str(shape(actual_coordinates)), str(shape(data)))
    assert shape(actual_coordinates)
    assert type(label) is StringType, "label must be a string: {0}".format(str(label))
    assert type(accuracy_z_value) is IntType or type(accuracy_z_value) is FloatType, "accuracy_z_value must be int or float: {0}".format(str(accuracy_z_value))
    assert accuracy_z_value > 0, "accuracy_z_value must be greater than 0: {0}".format(str(accuracy_z_value))
    assert type(flags) is type(PipelineFlags.All), "flags is not of type PipelineFlags: {0}".format(str(flags))

    results = []
    # Iterate through the trial lines
    for idx, (aline, dline) in enumerate(zip(actual_coordinates, dat)):
        # Process the trial
        line_result = full_pipeline(aline, dline, accuracy_z_value=accuracy_z_value, flags=flags,
                                    debug_labels=[label, idx])
        results.append(line_result)
    return results


def batch_pipeline(search_directory, data_shape, out_filename, accuracy_z_value=1.96, flags=PipelineFlags.All,
                   collapse_trials=True):
    """

    :rtype: None
    :param search_directory:
    :param data_shape:
    :param out_filename:
    :param accuracy_z_value:
    :param flags:
    :param collapse_trials:
    """
    if not os.path.exists(search_directory):
        logging.error('The input path was not found.')
        exit()

    logging.info('Finding files in folder {0}.'.format(search_directory))

    # Find the files
    actual_coordinates_filename, data_coordinates_filenames = find_data_files_in_directory(search_directory)

    logging.info('Parsing files with expected shape {0}.'.format(data_shape))

    # Parse the files
    actual_coordinates = get_coordinates_from_file(actual_coordinates_filename, data_shape)
    data_coordinates = [get_coordinates_from_file(filename, data_shape) for filename in data_coordinates_filenames]
    data_labels = [get_id_from_file(filename) for filename in data_coordinates_filenames]
    logging.info('The following ids were found and are being processed: {0}'.format(data_labels))

    # Get the labels and aggregation methods
    agg_functions = get_aggregation_functions()
    header_labels = get_header_labels()

    # Add cross-trial labels and aggregation methods
    agg_functions.append(nansum)
    header_labels.append('num_rows_with_nan')

    # Generate the output file and write the header
    out_fp = open(out_filename, 'w')
    header = "subID,{0}\n".format(','.join(header_labels))
    out_fp.write(header)

    # Iterate through the participants
    for dat, label in zip(data_coordinates, data_labels):
        logging.debug('Parsing {0}.'.format(label))

        # Get results
        results = get_single_file_result(actual_coordinates, dat, label=label,
                                         accuracy_z_value=accuracy_z_value, flags=flags)

        # Append the across-trial variables
        # Look for NaNs
        for line in results:
            num_rows_with_nan = 0
            for item in line:
                if item is numpy.nan:
                    num_rows_with_nan += 1
            line.append(num_rows_with_nan)

        if collapse_trials:
            # Apply the aggregation function to each value
            result = []
            for idx, column in enumerate(transpose(results)):
                result.append(agg_functions[idx](column))

            # Write to file
            out_fp.write('{0},{1}\n'.format(label, ','.join([str(r) for r in result])))
        else:
            for idx, row in enumerate(results):
                out_fp.write('{0},{1},{2}\n'.format(label, idx, ','.join([str(r) for r in row])))

    out_fp.close()

    logging.info('Done processing all files. Data can be found in {0}.'.format(out_filename))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    parser.add_argument('search_directory', type=str, help='the root directory in which to search for the actual and '
                                                           'data coordinate files (actual_coordinates.txt and '
                                                           '###position_data_coordinates.txt, respectively)')
    parser.add_argument('num_trials', type=int, help='the number of trials in each file')
    parser.add_argument('num_items', type=int, help='the number of items to be analyzed')
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

    if len(sys.argv) > 1:
        args = parser.parse_args()
        batch_pipeline(args.search_directory, (args.num_trials, args.num_items, 2),
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"),
                       accuracy_z_value=args.accuracy_z_value, flags=PipelineFlags(args.pipeline_mode),
                       collapse_trials=args.collapse_trials != 0)
        exit()

    logging.info("No arguments found - assuming running in test mode.")

    batch_pipeline("Z:\\Kevin\\iPosition\\Hillary\\MRE", (15, 5, 2),
                   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"))
