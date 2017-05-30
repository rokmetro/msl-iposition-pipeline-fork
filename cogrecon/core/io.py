import numpy as np
import logging
import os


# This function reads a data file and shapes the data into the appropriate expected shape (usually (Nt, Ni, 2) where
# Nt is the number of trials (rows) and Ni is the number of items (columns / 2), and 2 is the number of dimensions.
def get_coordinates_from_file(path, expected_shape):
    with open(path, 'rU') as tsv:
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
    return coordinates


# This function grabs the first 3 characters of the filename which are assumed to be the participant id
def get_id_from_file_prefix(path, prefix_length=3):
    return os.path.basename(path)[0:prefix_length]
