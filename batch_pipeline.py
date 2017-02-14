import argparse
import datetime
import logging
import os
import sys
import time

import numpy

from full_pipeline import full_pipeline, get_header_labels, get_coordinates_from_file, get_id_from_file, PipelineFlags

logging.basicConfig(level=logging.INFO)

# TODO: Documentation needs an audit/overhaul


# This function crawls the specified directory looking for the actual coordinate file and data files
def find_data_files_in_directory(directory):
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


# TODO: Cross-trial metrics should be computed more cleanly
# TODO: An output mode which allows all trials to be output should be added
# TODO: The ability to exclude columns from averaging should be added
def batch_pipeline(search_directory, data_shape, out_filename, accuracy_z_value=1.96, flags=PipelineFlags.All):
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

    # Generate the output file and write the header
    out_fp = open(out_filename, 'w')
    # Keep in mind some columns may be aggregated across trials rather than the mean of trial data - these are
    # populated in the header here.
    header = "subID,{0}\n".format(','.join(get_header_labels() + ["num_rows_with_nan"]))
    out_fp.write(header)
    # Iterate through the participants
    for dat, label in zip(data_coordinates, data_labels):
        logging.debug('Parsing {0}.'.format(label))
        results = []
        # Keep track of NaN values so we know how much to believe a given row
        num_rows_with_nan = 0
        # Iterate through the trial lines
        for idx, (aline, dline) in enumerate(zip(actual_coordinates, dat)):
            # Process the trial
            line_result = full_pipeline(aline, dline, accuracy_z_value=accuracy_z_value, flags=flags,
                                        debug_labels=[label, idx])
            # Look for NaNs
            if numpy.nan in line_result:
                num_rows_with_nan += 1
            results.append(line_result)
        # Take the mean of all values (ignoring NaN)
        result = numpy.nanmean(results, axis=0)
        # Append the across-trial variables
        result = numpy.append(result, num_rows_with_nan)
        out_fp.write('{0},{1}\n'.format(label, ','.join([str(r) for r in result])))
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

    if len(sys.argv) > 1:
        args = parser.parse_args()
        actual = get_coordinates_from_file(args.actual_coordinates, (args.num_trials, args.num_items, 2))
        data = get_coordinates_from_file(args.data_coordinates, (args.num_trials, args.num_items, 2))
        full_pipeline(actual[args.line_number], data[args.line_number],
                      accuracy_z_value=args.accuracy_z_value, flags=PipelineFlags(args.pipeline_mode), visualize=True)
        exit()

    logging.info("No arguments found - assuming running in test mode.")

    batch_pipeline("Z:\\Kevin\\iPosition\\Hillary\\MRE", (15, 5, 2),
                   datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"))
