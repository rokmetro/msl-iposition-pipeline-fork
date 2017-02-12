import logging
import os
import time
import numpy
import datetime
from full_pipeline import full_pipeline, get_header_labels

logging.basicConfig(level=logging.INFO)


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


# This function reads a data file and shapes the data into the appropriate expected shape (usually (Nt, Ni, 2) where
# Nt is the number of trials (rows) and Ni is the number of items (columns / 2), and 2 is the number of dimensions.
def get_coordinates_from_file(path, expected_shape):
    with open(path) as tsv:
        coordinates = zip(*([float(element) for element in line.strip().split('\t')] for line in tsv))
        coordinates = numpy.transpose(coordinates)
        coordinates = numpy.reshape(numpy.array(coordinates), expected_shape)
    if expected_shape is not None:
        assert numpy.array(coordinates).shape == expected_shape, \
            "shape {0} does not equal expectation {1}".format(numpy.array(data).shape, expected_shape)
    return coordinates


# This function grabs the first 3 characters of the filename which are assumed to be the participant id
def get_id_from_file(path):
    return os.path.basename(path)[0:3]


# Global variables
# TODO: Replace this with command line arguments
search_directory = "Z:\\Kevin\\iPosition\\Hillary\\MRE"
data_shape = (15, 5, 2)
out_filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")

logging.info('Finding files in folder {0}.'.format(search_directory))

# Find the files
actual, data = find_data_files_in_directory(search_directory)

logging.info('Parsing files with expected shape {0}.'.format(data_shape))

# Parse the files
actual_coordinates = get_coordinates_from_file(actual, data_shape)

data_coordinates = [get_coordinates_from_file(filename, data_shape) for filename in data]
data_labels = [get_id_from_file(filename) for filename in data]

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
        line_result = full_pipeline(aline, dline, debug_labels=[label, idx])
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
