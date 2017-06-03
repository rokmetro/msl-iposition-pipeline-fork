import numpy as np
import logging
import os
import time


# TODO: Documentation needs an audit/overhaul


# This function reads a data file and shapes the data into the appropriate expected shape (usually (Nt, Ni, 2) where
# Nt is the number of trials (rows) and Ni is the number of items (columns / 2), and 2 is the number of dimensions.
def get_coordinates_from_file(path, expected_shape, data_type=float):
    with open(os.path.abspath(path), 'rU') as tsv:
        if data_type is not None:
            coordinates = zip(*([data_type(element.strip()) for element in line.strip().split('\t')]
                                for line in tsv if line.strip() is not ''))
        else:
            coordinates = zip(*([element.strip() for element in line.strip().split('\t')]
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
def get_id_from_file_prefix_via_suffix(path, suffix):
    return os.path.basename(path)[:-len(suffix)]


def file_list_contents_equal(file_list):
    contents = None
    for f in file_list:
        with open(f, 'rU') as fp:
            new_contents = fp.read()
            if contents is None:
                contents = new_contents
            elif contents != new_contents:
                return False
    return True


def enforce_single_file_contents(file_list, name):
    if file_list is None or len(file_list) == 0:
        return []
    if len(file_list) == 1:
        file_list = file_list[0]
        return file_list
    else:
        logging.warning("Found {0} {1} files when 1 was expected. Checking contents for "
                        "matching.".format(len(file_list), name))
    if file_list_contents_equal(file_list):
        file_list = file_list[0]
        logging.warning("Contents match, using first found {1} file "
                        "({0}).".format(file_list, name))
    else:
        logging.error("Found {0} {1} files when 1 was expected. Not all files matched. "
                      "Quitting.".format(file_list, name))
        exit()

    return file_list


def make_singular_filename_values_list(value, expected_length):
    if isinstance(value, list) and len(value) > 1:
        return value + ([""] * (expected_length - len(value)))
    elif isinstance(value, list) and len(value) == 1:
        return value * expected_length
    elif value is not None:
        return [value] * expected_length
    else:
        return [""] * expected_length


def extract_prefixes_from_file_list_via_suffix(file_list, suffix):
    out_list = []
    for f in file_list:
        if f == "" or f == []:
            out_list.append("")
        elif len(f) == len(suffix):
            out_list.append(f)
        else:
            base = os.path.basename(f)
            out_list.append(base[:-len(suffix)])
    return out_list


def match_file_prefixes(files, prefixes):
    for idx, (file_list, prefix_list) in enumerate(zip(files, prefixes)):
        sort_idxs = list(range(len(file_list)))
        sort_idxs.sort(key=prefix_list.__getitem__)
        files[idx] = list(map(file_list.__getitem__, sort_idxs))
        prefixes[idx] = list(map(prefix_list.__getitem__, sort_idxs))

    prefix_comparison_list = np.transpose(prefixes)
    for row in prefix_comparison_list:
        if len(filter(lambda a: a == "", list(set(row)))) != 1:
            logging.error("There was a problem matching up files via their prefixes. This is most commonly due to "
                          "inappropriate files being found via search. Check that your files are unique and properly "
                          "formatted then try again.")
            raise Exception("Failure to match items uniquely.")

    return files


def find_data_files_in_directory(directory, actual_coordinate_prefixes=False,
                                 category_prefixes=False, category_independence_enabled=False,
                                 order_prefixes=True, order_greedy_deanonymization_enabled=False):
    """
    This function crawls the specified directory, recursively looking for the actual coordinate file and data files

    :param order_prefixes:
    :param category_prefixes:
    :param order_greedy_deanonymization_enabled:
    :param category_independence_enabled:
    :param actual_coordinate_prefixes:
    :rtype: string (or None), list of strings (or empty list)
    :param directory: the directory (string) in which to recursively search for data files
    :return: the actual coordinate filename/path (None if no file was found), a list of the data filenames/paths
    (empty list if no files were found)
    """
    # Check our data types
    assert isinstance(directory, str), "directory is not a string: {0}".format(directory)

    # Ensure the directory exists
    if not os.path.exists(directory):
        raise IOError('The input path was not found.')

    # Start timing execution
    start_time = time.time()

    # Create file type lists
    data_files = []
    order_files = []
    actual_coordinates_files = []
    category_files = []

    # Populate directory listing
    file_index = []
    file_roots_index = []
    for root, dirs, files in os.walk(directory):
        for f_idx in files:
            file_index.append(f_idx)
            file_roots_index.append(root)

    # Iterate through files and store in appropriate list via suffix
    for root, f_idx in zip(file_roots_index, file_index):
        filepath = os.path.join(root, f_idx)

        if filepath.endswith("position_data_coordinates.txt"):  # If we find a data file, save it to the file list
            logging.debug('Found data file ({0}).'.format(filepath))
            data_files.append(filepath)

        if filepath.endswith("order.txt"):  # If we find a data file, save it to the file list
            logging.debug('Found order file ({0}).'.format(filepath))
            order_files.append(filepath)

        if filepath.endswith("categories.txt"):
            logging.debug('Found category file ({0}).'.format(filepath))
            category_files.append(filepath)

        if filepath.endswith("actual_coordinates.txt"):
            logging.debug('Found actual coordinates file ({0}).'.format(filepath))
            actual_coordinates_files.append(filepath)

    # Ensure that we found at least 1 of each required file and if enabled, at least one of each optional file
    assert len(actual_coordinates_files) >= 1, "there must be at least one actual_coordinates.txt file"
    assert len(data_files) >= 1, "there must be at least one data file ending in position_data_coordinates.txt"
    if order_greedy_deanonymization_enabled:
        assert len(order_files) >= 1, "if order_greedy_deanonymization_enabled is True, there must be at least one " \
                                      "order file ending in order.txt "
    if category_independence_enabled:
        assert len(category_files) >= 1, "if category_independence_enabled is True, there must be at least one " \
                                         "category file ending in category.txt "

    # For each non-data file, we can enforce singular file contents on the file list if enabled
    if not actual_coordinate_prefixes:
        actual_coordinates_files = enforce_single_file_contents(actual_coordinates_files, "actual_coordinates.txt")

    if not category_prefixes:
        category_files = enforce_single_file_contents(category_files, "categories.txt")

    if not order_prefixes:
        order_files = enforce_single_file_contents(order_files, "order.txt")

    # We need to generate temporary lists of equal length so we can pair off the appropriate files with each other
    # For actual_coordinates files, we expect either a list identical values or a list of all unique, prefixed values
    tmp_acf = make_singular_filename_values_list(actual_coordinates_files, len(data_files))
    # For category files, we expect either a list identical values, a list of empty values,
    # or a list of all unique, prefixed values
    tmp_cat = make_singular_filename_values_list(category_files, len(data_files))
    # For order files, we expect either a list identical values, a list of empty values,
    # or a list of all unique, prefixed values
    tmp_order = make_singular_filename_values_list(order_files, len(data_files))

    assert len(data_files) == len(tmp_acf) and len(data_files) == len(tmp_cat) and len(data_files) == len(tmp_order), \
        "input file type length error - not enough files were found of each type to properly associate the data"

    # Next, we need to extract prefixes from all of our file lists for sorting

    files = [
        data_files,
        tmp_acf,
        tmp_cat,
        tmp_order
    ]

    prefixes = [
        extract_prefixes_from_file_list_via_suffix(data_files, "position_data_coordinates.txt"),
        extract_prefixes_from_file_list_via_suffix(tmp_acf, "actual_coordinates.txt"),
        extract_prefixes_from_file_list_via_suffix(tmp_cat, "categories.txt"),
        extract_prefixes_from_file_list_via_suffix(tmp_order, "order.txt")
    ]

    data_files, actual_coordinates_files, category_files, order_files = match_file_prefixes(files, prefixes)

    logging.info('Found {0} data files in {1} seconds.'.format(len(data_files), time.time() - start_time))

    return actual_coordinates_files, data_files, category_files, order_files
