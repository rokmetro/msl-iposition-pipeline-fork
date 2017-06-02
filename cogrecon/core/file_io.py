import numpy as np
import logging
import os
import time

# TODO: Documentation needs an audit/overhaul


# This function reads a data file and shapes the data into the appropriate expected shape (usually (Nt, Ni, 2) where
# Nt is the number of trials (rows) and Ni is the number of items (columns / 2), and 2 is the number of dimensions.
def get_coordinates_from_file(path, expected_shape):
    with open(os.path.abspath(path), 'rU') as tsv:
        coordinates = zip(*([float(element.strip()) for element in line.strip().split('\t')]
                            for line in tsv if line.strip() is not ''))
        coordinates = np.transpose(coordinates)
    if expected_shape is not None:
        try:
            coordinates = np.reshape(np.array(coordinates), expected_shape)
        except ValueError:
            logging.error("Data found in path ({0}) cannot be transformed " +
                          "into expected shape ({1}).".format(path, expected_shape))
            exit()
        assert np.array(coordinates).shape == expected_shape, \
            "shape {0} does not equal expectation {1}".format(np.array(coordinates).shape, expected_shape)
    return coordinates.tolist()


# This function grabs the first 3 characters of the filename which are assumed to be the participant id
def get_id_from_file_prefix(path, prefix_length=3):
    return os.path.basename(path)[0:prefix_length]


def find_data_files_in_directory(directory, actual_coordinate_prefixes=False, prefix_length=3):
    """
    This function crawls the specified directory, recursively looking for the actual coordinate file and data files

    :param prefix_length: 
    :param actual_coordinate_prefixes: 
    :rtype: string (or None), list of strings (or empty list)
    :param directory: the directory (string) in which to recursively search for data files
    :return: the actual coordinate filename/path (None if no file was found), a list of the data filenames/paths
    (empty list if no files were found)
    """
    assert isinstance(directory, str), "directory is not a string: {0}".format(directory)

    if not os.path.exists(directory):
        raise IOError('The input path was not found.')

    start_time = time.time()
    data_files = []
    actual_coordinate_file = None
    actual_coordinate_contents = None
    if actual_coordinate_prefixes:
        actual_coordinate_file = []
        actual_coordinate_contents = []
    file_index = []
    file_roots_index = []
    for root, dirs, files in os.walk(directory):
        for f_idx in files:
            file_index.append(f_idx)
            file_roots_index.append(root)

    for root, f_idx in zip(file_roots_index, file_index):
        if not actual_coordinate_prefixes:
            if os.path.basename(f_idx) == "actual_coordinates.txt":  # If we find an actual coordinate file
                if actual_coordinate_file is None:  # And we haven't found a coordinate file before
                    actual_coordinate_file = os.path.join(root, f_idx)  # Set the coordinate file
                    with open(actual_coordinate_file) as fp:  # Save its contents
                        actual_coordinate_contents = fp.read()
                    logging.debug('Found actual_coordinates.txt ({0}).'.format(actual_coordinate_file))
                else:  # If we have found an additional coordinate file
                    with open(os.path.join(root, f_idx)) as fp:  # Get its contents
                        new_contents = fp.read()
                    if new_contents != actual_coordinate_contents:  # Compare its contents to the first found file
                        # If the contents are not the same, quit - because we don't know which to use.
                        logging.error(('Found multiple actual_coordinates.txt with different contents, ' +
                                       'program will now exit (found {0}).').format(actual_coordinate_file))
                        exit()
                    else:  # Otherwise continue and warn the user
                        logging.warning('Found multiple actual_coordinates.txt but contents were identical.')
            if f_idx.endswith("position_data_coordinates.txt"):  # If we find a data file, save it to the file list
                logging.debug('Found data file ({0}).'.format(f_idx))
                data_files.append(os.path.join(root, f_idx))
        else:
            if os.path.basename(f_idx).endswith("actual_coordinates.txt"):  # If we find an actual coordinate file
                # And we haven't found a coordinate file before
                if os.path.join(root, f_idx) not in actual_coordinate_file:
                    actual_coordinate_file.append(os.path.join(root, f_idx))  # Set the coordinate file
                    with open(os.path.join(root, f_idx)) as fp:  # Save its contents
                        actual_coordinate_contents.append(fp.read())
                    logging.debug('Found actual_coordinates.txt ({0}).'.format(actual_coordinate_file))
                    prefix = get_id_from_file_prefix(f_idx, prefix_length=prefix_length)
                    for r2, f2 in zip(file_roots_index, file_index):
                        # If we find a data file, save it to the file list
                        if f2.endswith(
                                "position_data_coordinates.txt") and \
                                        prefix == get_id_from_file_prefix(f2, prefix_length=prefix_length):
                            logging.debug('Found data file ({0}).'.format(f2))
                            data_files.append(os.path.join(r2, f2))

                else:
                    a_coords_file_duplic_idx = actual_coordinate_file.index(os.path.join(root, f_idx))
                    assert a_coords_file_duplic_idx >= len(actual_coordinate_file) or a_coords_file_duplic_idx < 0, \
                        'indexing error with duplicate actual coordinate file'
                    with open(os.path.join(root, f_idx)) as fp:  # Get its contents
                        new_contents = fp.read()
                    # Compare its contents to the first found file
                    if new_contents != actual_coordinate_contents[a_coords_file_duplic_idx]:
                        # If the contents are not the same, quit - because we don't know which to use.
                        logging.error(('Found multiple actual_coordinates.txt with different contents, ' +
                                       'program will now exit (found {0}).').format(
                            actual_coordinate_file[a_coords_file_duplic_idx]))
                        exit()
                    else:  # Otherwise continue and warn the user
                        logging.warning(('Found multiple actual_coordinates.txt ' +
                                         'but contents were identical ({0}).').format(
                            actual_coordinate_file[a_coords_file_duplic_idx]))
    logging.info('Found {0} data files in {1} seconds and {2} actual coordinate file.'.format(len(data_files),
                                                                                              time.time() - start_time,
                                                                                              actual_coordinate_file))

    return actual_coordinate_file, data_files
