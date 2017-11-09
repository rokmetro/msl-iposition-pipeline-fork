import numpy as np
import logging
import os
import sys
import tempfile
import errno
import time
from .cogrecon_globals import data_coordinates_file_suffix, order_file_suffix, category_file_suffix, \
    actual_coordinates_file_suffix


def get_coordinates_from_file(path, expected_shape, dimension=None, data_type=float):
    """
    This function reads a data file and shapes the data into the appropriate expected shape (usually (Nt, Ni, 2) where
    Nt is the number of trials (rows) and Ni is the number of items (columns / dimensions)

    :param dimension: the dimensionality of the data (i.e. 2, for 2D for x and y)
    :param path: a path to a coordinate file
    :param expected_shape: the expected shape of a coordinate file (if None, the shape will be detected using dimension)
    :param data_type: the data type of the coordinate file

    :return: a list of shape expected_shape and type data_type
    """
    with open(os.path.abspath(path), 'rU') as tsv:
        if data_type is not None:
            coordinates = zip(*([data_type(element.strip()) for element in line.strip().split('\t')]
                                for line in tsv if line.strip() is not ''))
        else:
            coordinates = zip(*([element.strip() for element in line.strip().split('\t')]
                                for line in tsv if line.strip() is not ''))
        coordinates = np.transpose(coordinates)

    if expected_shape is None:
        if dimension is None:
            raise ValueError("Could not detect data shape for {0}. If no expected_shape is provided, a dimension must "
                             "be provided.")
        expected_shape = detect_shape_from_file(os.path.abspath(path), dimension)

    if expected_shape is not None:
        if expected_shape[-1] == 1:
            expected_shape = expected_shape[:2]
        try:
            coordinates = np.reshape(np.array(coordinates), expected_shape)
        except ValueError:
            logging.error(("Data found in path ({0}) cannot be transformed " +
                           "into expected shape ({1})).").format(path, expected_shape))
            raise ValueError("Failed to get data coordinate of expected shape.")
        assert np.array(coordinates).shape == expected_shape, \
            "shape {0} does not equal expectation {1}".format(np.array(coordinates).shape, expected_shape)

    return coordinates.tolist()


def get_id_from_file_prefix_via_suffix(path, suffix):
    """
    This function grabs the first 3 characters of the filename which are assumed to be the participant id

    :param path: the path to a file ending in suffix
    :param suffix: the ending part of a filename
    :return: the os.path.basename of path with characters of length of suffix removed from the end
    """
    return os.path.basename(path)[:-len(suffix)]


def file_list_contents_equal(file_list):
    """
    This function checks a list of files to ensure the contents are all equal across each file.

    :param file_list: a list of file paths
    :return: True if all files are equal in contents, False otherwise
    """
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
    """
    This function ensures that if file_list contains multiple unique files, an error is thrown, otherwise it simply
    returns the first element.

    :param file_list: a list of files which should all be the same
    :param name: the name of the file list type for debugging purposes
    :return: a path to a unique single file or empty list if none was found
    """
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
        raise IOError("Unable to enforce single-file-contents requirement on files which should be identical.")

    return file_list


def make_singular_filename_values_list(value, expected_length):
    """
    This function takes a value and produces an appropriate length list of values. If the input is a list of values
    already, the list will have empty strings appended to it until it is the expected length. If it is greater
    than the expected length already, it will simply be returned as-is. If it is a list with one element, that element
    will be duplicated to make the list expected_length and returned. If it is a value, a list containing
    expected_length numbers of that value will be returned. If none of these apply, a list of empty strings is returned.

    :param value: the value contents of the output list
    :param expected_length: the expected length of the output list

    :return: a list of length expected_length with contents reflected by value
    """
    if isinstance(value, list) and len(value) > 1:
        return value + ([""] * (expected_length - len(value)))
    elif isinstance(value, list) and len(value) == 1:
        return value * expected_length
    elif value is not None:
        return [value] * expected_length
    else:
        return [""] * expected_length


def extract_prefixes_from_file_list_via_suffix(file_list, suffix):
    """
    This function will return a list of prefixes from files given an expected suffix.

    :param file_list: a list of file paths
    :param suffix: a suffix whose contents should be removed from the end of the file_list element's basename
    :return: a list of file prefixes
    """
    out_list = []
    for f in file_list:
        if f == "" or f == []:
            out_list.append("")
            continue
        f_base = os.path.basename(f)
        if len(f_base) == len(suffix):
            out_list.append(f_base)
        else:
            base = os.path.basename(f_base)
            out_list.append(base[:-len(suffix)])
    return out_list


def match_file_prefixes(files, prefixes):
    """
    This function attempts to sort and match the list of files to a list of prefixes for each file.

    :param files: a list of file paths which should be associated with each prefix
    :param prefixes: a list of prefixes which should be associated with each file path
    :return: a list of files, sorted according to prefixes
    """
    for idx, (file_list, prefix_list) in enumerate(zip(files, prefixes)):
        sort_idxs = list(range(len(file_list)))
        sort_idxs.sort(key=prefix_list.__getitem__)
        files[idx] = list(map(file_list.__getitem__, sort_idxs))
        prefixes[idx] = list(map(prefix_list.__getitem__, sort_idxs))

    prefix_comparison_list = np.transpose(prefixes)
    for row in prefix_comparison_list:
        if len(filter(lambda a: a != "" and a != actual_coordinates_file_suffix
           and a != category_file_suffix
           and a != order_file_suffix,
                      list(set(row)))) != 1:
            logging.error("There was a problem matching up files via their prefixes. This is most commonly due to "
                          "inappropriate files being found via search. Check that your files are unique and properly "
                          "formatted then try again.")
            raise Exception("Failure to match items uniquely.")

    return files


def find_data_files_in_directory(directory, actual_coordinate_prefixes=False,
                                 category_prefixes=False, category_independence_enabled=False,
                                 order_prefixes=True, order_greedy_deanonymization_enabled=False,
                                 _data_coordinates_file_suffix=data_coordinates_file_suffix,
                                 _order_file_suffix=order_file_suffix, _category_file_suffix=category_file_suffix,
                                 _actual_coordinates_file_suffix=actual_coordinates_file_suffix):
    """
    This function crawls the specified directory, recursively looking for the actual coordinate file and data files.

    :param directory: the directory (string) in which to recursively search for data files

    :param _category_file_suffix: the category file suffix for which to search
    :param _actual_coordinates_file_suffix: the actual coordinate file suffix for which to search
    :param _order_file_suffix: the order file suffix for which to search
    :param _data_coordinates_file_suffix: the data file suffix for which to search

    :param order_prefixes: if True, it is assumed there will be an equal number of order files as data files with
                           identical prefixes, otherwise one file is expected
    :param category_prefixes: if True, it is assumed there will be an equal number of category files as data files with
                              identical prefixes, otherwise one file is expected
    :param actual_coordinate_prefixes: if True, it is assumed there will be an equal number of actual coordinate files
                                       as data files with identical prefixes, otherwise one file is expected

    :param order_greedy_deanonymization_enabled: whether the greedy, order based deanonymization method
                                                 should be used in determining the mapping of object to location.
                                                 Note that if enabled, an order file (or files) is expected.
    :param category_independence_enabled: whether the items involved have associated categorical information
                                          such that they should be processed independently.
                                          Note that if enabled, a category file (or files) is expected.

    :rtype: string (or None), list of strings (or empty list)

    :return: the actual coordinate filename/path (None if no file was found), a list of the data filenames/paths
             (empty list if no files were found), a list of category filenames/paths (empty list if no files were found
             or requested), and a list of order filenames/paths (empty list if no files were found or requested)
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

        if filepath.endswith(_data_coordinates_file_suffix):  # If we find a data file, save it to the file list
            logging.debug('Found data file ({0}).'.format(filepath))
            data_files.append(filepath)

        if filepath.endswith(_order_file_suffix):  # If we find a data file, save it to the file list
            logging.debug('Found order file ({0}).'.format(filepath))
            order_files.append(filepath)

        if filepath.endswith(_category_file_suffix):
            logging.debug('Found category file ({0}).'.format(filepath))
            category_files.append(filepath)

        if filepath.endswith(_actual_coordinates_file_suffix):
            logging.debug('Found actual coordinates file ({0}).'.format(filepath))
            actual_coordinates_files.append(filepath)

    # Ensure that we found at least 1 of each required file and if enabled, at least one of each optional file
    assert len(actual_coordinates_files) >= 1, \
        "there must be at least one {0} file".format(_actual_coordinates_file_suffix)
    assert len(data_files) >= 1, \
        "there must be at least one data file ending in {0}".format(_data_coordinates_file_suffix)
    if order_greedy_deanonymization_enabled:
        assert len(order_files) >= 1, "if order_greedy_deanonymization_enabled is True, there must be at least one " \
                                      "order file ending in {0}".format(_order_file_suffix)
    if category_independence_enabled:
        assert len(category_files) >= 1, "if category_independence_enabled is True, there must be at least one " \
                                         "category file ending in {0}".format(_category_file_suffix)

    # For each non-data file, we can enforce singular file contents on the file list if enabled
    if not actual_coordinate_prefixes:
        actual_coordinates_files = enforce_single_file_contents(actual_coordinates_files,
                                                                _actual_coordinates_file_suffix)

    if not category_prefixes and category_independence_enabled:
        category_files = enforce_single_file_contents(category_files,
                                                      _category_file_suffix)

    if not order_prefixes and order_greedy_deanonymization_enabled:
        order_files = enforce_single_file_contents(order_files,
                                                   _order_file_suffix)

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
        extract_prefixes_from_file_list_via_suffix(data_files, _data_coordinates_file_suffix),
        extract_prefixes_from_file_list_via_suffix(tmp_acf, _actual_coordinates_file_suffix),
        extract_prefixes_from_file_list_via_suffix(tmp_cat, _category_file_suffix),
        extract_prefixes_from_file_list_via_suffix(tmp_order, _order_file_suffix)
    ]

    data_files, actual_coordinates_files, category_files, order_files = match_file_prefixes(files, prefixes)

    logging.info('Found {0} data files in {1} seconds.'.format(len(data_files), time.time() - start_time))

    return actual_coordinates_files, data_files, category_files, order_files


def is_pathname_valid(pathname):
    """
    Windows-specific error code indicating an invalid pathname.

    See Also: https://msdn.microsoft.com/en-us/library/windows/desktop/ms681382%28v=vs.85%29.aspx
    Official listing of all such codes.


    :param pathname: the pathname about which to determine validity
    :return: `True` if the passed pathname is a valid pathname for the current OS;
             `False` otherwise.

    """

    # Sadly, Python fails to provide the following magic number for us.
    ERROR_INVALID_NAME = 123

    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)  # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
        # If any other exception was raised, this is an unrelated fatal issue
        # (e.g., a bug). Permit this exception to unwind the call stack.
        #
        # Did we mention this should be shipped with Python already?


def is_path_sibling_creatable(pathname):
    """
    This function helps determine if a path is creatable.

    :param pathname: the pathname about which to determine if it is creatable
    :return: `True` if the current user has sufficient permissions to create **siblings**
             (i.e., arbitrary files in the parent directory) of the passed pathname;
             `False` otherwise.
    """
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()

    try:
        # For safety, explicitly close and hence delete this temporary file
        # immediately after creating it in the passed path's parent directory.
        with tempfile.TemporaryFile(dir=dirname):
            pass
        return True
    # While the exact type of exception raised by the above function depends on
    # the current version of the Python interpreter, all such types subclass the
    # following exception superclass.
    except EnvironmentError:
        return False


def is_path_exists_or_creatable_portable(pathname):
    """
    This function helps determine if a pathname exists or is creatable.

    This function is guaranteed to _never_ raise exceptions.

    :param pathname: the pathname about which it should be determined if it exists or is creatable
    :return: `True` if the passed pathname is a valid pathname on the current OS _and_
             either currently exists or is hypothetically creatable in a cross-platform
             manner optimized for POSIX-unfriendly filesystems; `False` otherwise.
    """
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_sibling_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False


def detect_shape_from_file(path, dimension):
    """
    This function uses the requested dimensionality and path contents of a coordinate file to automatically
    determine the data shape.


    :param path: a value (string) containing the path of the file from which structure should be detected
    :param dimension: a value (integer) which represents the dimensionality of the data

    :rtype: int, int, int
    :return: a tuple containing the trial count, the item count, and the dimensionality
    """
    assert isinstance(path, str), 'path is not string: {0}'.format(path)
    assert os.path.exists(path), 'path does not exist: {0}'.format(path)

    with open(path, 'rU') as tsv:
        trial_count = 0
        item_count_list = []
        for tsv_line in tsv:
            if tsv_line.strip() == '':
                continue
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

        return trial_count, int(float(item_count_list[0]) / float(dimension)), dimension
