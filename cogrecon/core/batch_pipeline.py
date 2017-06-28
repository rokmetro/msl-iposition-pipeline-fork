import logging
import numpy as np
import copy
import datetime

from .full_pipeline import full_pipeline, get_aggregation_functions, get_header_labels
from .file_io import get_coordinates_from_file, get_id_from_file_prefix_via_suffix, find_data_files_in_directory
from .data_structures import TrialData, ParticipantData, AnalysisConfiguration, PipelineFlags
from .data_flexing.dimension_removal import remove_dimensions
from .cogrecon_globals import default_z_value, default_pipeline_flags, default_dimensions, data_coordinates_file_suffix
from .._version import __version__


logging.basicConfig(level=logging.INFO)


def validate_list_format(l, require_numeric=False, dimension=None, list_name="list"):

    """

    This function validates that a list is the correct type, dimension, and
    contains only int and float values (if specified).

    :param l: a list whose type, dimensionality, and contents should be checked; valid types are list, tuple, or
              numpy array, dimensionality should match dimension, and the contents should all be int or float
    :param require_numeric: if True, elements must be int or float
    :param dimension: the expected number of dimensions (integer greater than 0) of the list
    :param list_name: the name (string) of the list for debugging purposes

    """

    assert isinstance(require_numeric, bool), "require_numeric is not a bool: {0}".format(require_numeric)
    assert isinstance(list_name, str), "list_name is not string: {0}".format(list_name)
    assert isinstance(l, list) or isinstance(l, tuple) or isinstance(l, np.ndarray), \
        "{1} should be list or numpy array: {0}".format(l, list_name)
    if dimension:
        assert isinstance(dimension, int), "dimension is not an integer: {0}".format(dimension)
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

    This function validates that two numeric lists have equal shape.

    :param l1: a list, tuple, or numpy array whose shape should be equal to the shape of l1
               and expected_shape (if specified)
    :param l2: a list, tuple, or numpy array whose shape should be equal to the shape of l2
               and expected_shape (if specified)
    :param expected_shape: a shape (list, tuple or numpy array) against which both l1 and l2 should be
                           compared to ensure they are equal to it and each other
    :param l1_name: the name (string) of l1 for debugging
    :param l2_name: the name (string) of l2 for debugging

    """

    assert isinstance(l1_name, str), "l1_name is not string: {0}".format(l1_name)
    assert isinstance(l2_name, str), "l2_name is not string: {0}".format(l2_name)
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
def get_single_file_result(actual_coordinates, dat, categories=None, data_orders=None,
                           label="", accuracy_z_value=default_z_value,
                           trial_by_trial_accuracy=True, manual_threshold=None,
                           flags=PipelineFlags(default_pipeline_flags), remove_dims=None,
                           category_independence_enabled=False,
                           order_greedy_deanonymization_enabled=False):

    """

    This function generates the results for a specific file's data structure, usually containing multiple trials

    :param actual_coordinates: the correct coordinates for the points - an (Nt, Ni, d) sized list of floats where Nt is
                               the number of trials, Ni is the number of items, and d is the dimensionality of the
                               points
    :param dat: the data coordinates for the points - an (Nt, Ni, d) sized list of floats where Nt is the number of
                trials, Ni is the number of items, and d is the dimensionality of the points

    :param order_greedy_deanonymization_enabled: whether the greedy, order based deanonymization method
                                                 should be used in determining the mapping of object to location.
                                                 Note that if enabled, an order file (or files) is expected.
    :param category_independence_enabled: whether the items involved have associated categorical information
                                          such that they should be processed independently.
                                          Note that if enabled, a category file (or files) is expected.
    :param remove_dims: a list of dimension indicies to remove from processing
    :param data_orders: a list of integer order values for the associated dat input data
                        (should be same shape as dat but without multiple axis dimensions)
    :param categories: a list of values for the associated dat input categories
                       (should be same shape as dat but without multiple axis dimensions)
    :param manual_threshold: a list of manual swap threshold values associated with the specified
                             trials in dat (should be of the same length as the number of trials)
    :param label: the label (string) identifying the participant ID for this file, used for debugging purposes only
    :param accuracy_z_value: a value (float or int) representing the z threshold for counting something as
    :param trial_by_trial_accuracy: when True, z_value thresholds are used on a trial-by-trial basis for
                                    accuracy calculations, when False, the thresholds are computed then collapsed
                                    across an individual's trials
    :param flags: the value (PipelineFlags) describing what pipeline elements should/should not be run on the data

    :return: a list, (Nt, r), where Nt is the number of trials and r is the number of result metrics, of results values
             from the analysis for each trial on a particular file's data
    :rtype: list (or empty list)

    """

    validate_list_format(actual_coordinates, require_numeric=True, dimension=3, list_name="actual_coordinates")
    validate_list_format(dat, require_numeric=True, dimension=3, list_name="dat")

    validate_equal_list_shapes(actual_coordinates, dat, l1_name="actual_coordinates", l2_name="dat")

    assert isinstance(label, str), "label must be a string: {0}".format(label)
    assert isinstance(accuracy_z_value, int) or isinstance(accuracy_z_value, float), \
        "accuracy_z_value must be int or float: {0}".format(accuracy_z_value)
    assert accuracy_z_value > 0, \
        "accuracy_z_value must be greater than 0: {0}".format(accuracy_z_value)
    assert isinstance(flags, PipelineFlags), \
        "flags is not of type PipelineFlags: {0}".format(flags)

    _analysis_configuration = AnalysisConfiguration(z_value=accuracy_z_value,
                                                    flags=flags,
                                                    debug_labels=[label],
                                                    trial_by_trial_accuracy=trial_by_trial_accuracy,
                                                    manual_threshold=manual_threshold,
                                                    process_categories_independently=category_independence_enabled,
                                                    greedy_order_deanonymization=order_greedy_deanonymization_enabled)
    if categories is None:
        categories = [None] * len(actual_coordinates)
    else:
        validate_list_format(categories, require_numeric=False, dimension=2, list_name="categories")
        assert len(categories) == len(actual_coordinates), "categories must be same length as actual_coordinates"
    if data_orders is None:
        data_orders = [None] * len(actual_coordinates)
    else:
        validate_list_format(data_orders, require_numeric=False, dimension=2, list_name="data_orders")
        assert len(categories) == len(dat), "data_orders must be same length as data_coordinates"
    _participant_data = ParticipantData([TrialData(_a, _d, cateogry_labels=_c, data_order=_o)
                                         for _a, _d, _c, _o in zip(actual_coordinates, dat, categories, data_orders)])

    if remove_dims is not None:
        _participant_data = remove_dimensions(_participant_data, removal_dim_indices=remove_dims)

    # Process the participant
    return full_pipeline(_participant_data, _analysis_configuration)


def batch_pipeline(search_directory, out_filename, data_shape=None, dimension=default_dimensions,
                   accuracy_z_value=default_z_value,
                   trial_by_trial_accuracy=True,
                   flags=PipelineFlags(default_pipeline_flags),
                   collapse_trials=True, manual_threshold=None,
                   actual_coordinate_prefixes=False,
                   category_independence_enabled=False, category_prefixes=False,
                   order_greedy_deanonymization_enabled=False, order_prefixes=True,
                   removal_dim_indicies=None):

    """

    This function allows the easy running of the pipeline on a directory and all of the appropriate files in its
    subdirectories. It will search for the actual coordinates and data files and process them all as specified
    by the other parameters.

    :param search_directory: the directory (string) in which to recursively search for data files
    :param out_filename: the filename and path (string) into which the data should be saved

    :param data_shape: a shape (list, tuple or numpy array) which describes the structure of the date;
                       (Nt, Ni, d) such that Nt is the number of trials, Ni is the number of items and d is the number
                       of dimensions; if None is given, an attempt will be made to automatically detect the shape from
                       the actual_coordinates file
    :param accuracy_z_value: a value (float or int) representing the z threshold for counting something as accurate
    :param trial_by_trial_accuracy: when True, z_value thresholds are used on a trial-by-trial basis for
                                    accuracy calculations, when False, the thresholds are computed then collapsed
                                    across an individual's trials
    :param flags: the value (PipelineFlags) describing what pipeline elements should/should not be run on the data
    :param collapse_trials: if True, the output file will contain one row per participant, otherwise each
                            trial will be output in an individual row
    :param removal_dim_indicies: a list of dimension indicies to remove from processing
    :param dimension: the number of dimensions of the input data
    :param order_greedy_deanonymization_enabled: whether the greedy, order based deanonymization method
                                                 should be used in determining the mapping of object to location.
                                                 Note that if enabled, an order file (or files) is expected.
    :param category_independence_enabled: whether the items involved have associated categorical information
                                          such that they should be processed independently.
                                          Note that if enabled, a category file (or files) is expected.
    :param manual_threshold: a list of manual swap threshold values associated with the specified participant
                             prefixes and trials in the batch process (should be of the same length as the number
                             of trials)
    :param order_prefixes: whether or not we will look for files associated with order in a one-to-one
                           fashion with data files based on the prefix values
    :param category_prefixes: whether or not we will look for files associated with category in a one-to-one
                              fashion with data files based on the prefix values
    :param actual_coordinate_prefixes: whether or not we will look for actual coordinate files in a
                                       one-to-one fashion with data files based on the prefix values

    :rtype: None

    """

    assert isinstance(search_directory, str), \
        "search_directory must be a string: {0}".format(search_directory)
    assert len(search_directory) > 0, "search_directory must have length greater than 0: {0}".format(search_directory)
    if data_shape:
        validate_list_format(data_shape, dimension=1, require_numeric=True, list_name="data_shape")
    assert isinstance(out_filename, str), \
        "out_filename is not string: {0}".format(out_filename)
    assert len(out_filename) > 0, \
        "out_filename must have length greater than 0: {0}".format(out_filename)
    assert isinstance(accuracy_z_value, int) or isinstance(accuracy_z_value, float), \
        "accuracy_z_value must be int or float: {0}".format(accuracy_z_value)
    assert accuracy_z_value > 0, \
        "accuracy_z_value must be greater than 0: {0}".format(accuracy_z_value)
    assert isinstance(flags, PipelineFlags), \
        "flags is not of type PipelineFlags: {0}".format(flags)
    assert isinstance(collapse_trials, bool), \
        "collapse_trials is not a bool: {0}".format(collapse_trials)
    assert isinstance(trial_by_trial_accuracy, bool), \
        "trial_by_trial_accuracy is not a bool: {0}".format(trial_by_trial_accuracy)
    assert isinstance(category_independence_enabled, bool), \
        "category_independence_enabled is not a bool {0}".format(category_independence_enabled)
    assert isinstance(order_greedy_deanonymization_enabled, bool), \
        "order_greedy_deanonymization_enabled is not a bool {0}".format(order_greedy_deanonymization_enabled)

    logging.info('Finding files in folder {0}.'.format(search_directory))

    # Find the files
    try:
        actual_coordinates_filename, data_coordinates_filenames, category_filenames, order_filenames = \
            find_data_files_in_directory(search_directory,
                                         actual_coordinate_prefixes=actual_coordinate_prefixes,
                                         category_independence_enabled=category_independence_enabled,
                                         order_greedy_deanonymization_enabled=order_greedy_deanonymization_enabled,
                                         category_prefixes=category_prefixes,
                                         order_prefixes=order_prefixes)
    except IOError:
        logging.error('The input path was not found.')
        raise IOError('Failed to find input path.')

    logging.info('Parsing files with expected shape {0}.'.format(data_shape))

    data_coordinates = []
    for f in data_coordinates_filenames:
        if f is None or f == "":
            data_coordinates.append(None)
        else:
            data_coordinates.append(get_coordinates_from_file(f, data_shape, dimension=dimension))

    actual_coordinates = []
    for idx, f in enumerate(actual_coordinates_filename):
        if f is None or f == "":
            actual_coordinates.append(None)
        else:
            try:
                index_check = actual_coordinates_filename.index(f)
            except ValueError:
                index_check = len(actual_coordinates) + 1
            if index_check < idx:
                actual_coordinates.append(copy.copy(actual_coordinates[index_check]))
            else:
                actual_coordinates.append(get_coordinates_from_file(f, data_shape, dimension=dimension))

    categories = []
    for idx, f in enumerate(category_filenames):
        if f is None or f == "" or f == []:
            categories.append(None)
        else:
            try:
                index_check = category_filenames.index(f)
            except ValueError:
                index_check = len(categories) + 1
            if index_check < idx:
                categories.append(copy.copy(categories[index_check]))
            else:
                if data_shape is None:
                    categories.append(get_coordinates_from_file(f, None, dimension=1))
                else:
                    categories.append(get_coordinates_from_file(f, tuple(list(data_shape[:2]) + [1]), dimension=1))

    data_orders = []
    for idx, f in enumerate(order_filenames):
        if f is None or f == "" or f == []:
            data_orders.append(None)
        else:
            try:
                index_check = order_filenames.index(f)
            except ValueError:
                index_check = len(data_orders) + 1
            if index_check < idx:
                data_orders.append(copy.copy(data_orders[index_check]))
            else:
                if data_shape is None:
                    data_orders.append(get_coordinates_from_file(f, None, data_type=int, dimension=1))
                else:
                    data_orders.append(get_coordinates_from_file(f, tuple(list(data_shape[:2]) + [1]),
                                                                 data_type=int, dimension=1))

    data_labels = [get_id_from_file_prefix_via_suffix(filename, data_coordinates_file_suffix) for filename in
                   data_coordinates_filenames]

    logging.info('The following ids were found and are being processed: {0}'.format(data_labels))

    # Get the labels and aggregation methods
    agg_functions = get_aggregation_functions()
    header_labels = get_header_labels()

    # Add cross-trial labels and aggregation methods
    agg_functions.append(np.nansum)
    header_labels.append('num_rows_with_nan')

    # Generate the output file and write the headers
    out_fp = open(out_filename, 'w')

    top_header = 'Generated with the msl-iposition-pipeline (https://github.com/kevroy314/msl-iposition-pipeline) ' \
                 'version {0} on {1}. Note: datetime provided may not match filename datetime if system is ' \
                 'slow.\n'.format(__version__, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"))

    out_fp.write(top_header)

    if collapse_trials:
        header = "subID,{0}\n".format(','.join(header_labels))
    else:
        header = "subID,trial,{0}\n".format(','.join(header_labels))

    out_fp.write(header)

    # Iterate through the participants
    for index, (label, actual, data, category, order) \
            in enumerate(zip(data_labels, actual_coordinates, data_coordinates, categories, data_orders)):
        logging.debug('Parsing {0}.'.format(label))
        mt = None
        if manual_threshold is not None:
            for dat_id, dat_threshold in manual_threshold:
                if dat_id == label:
                    mt = dat_threshold
                    break

        assert np.array(actual_coordinates[index]).shape == np.array(data).shape, \
            "shape mismatch between {0} and {1}".format(actual_coordinates_filename[index],
                                                        data_coordinates_filenames[index])
        results = get_single_file_result(actual, data, categories=category, data_orders=order,
                                         label=label, accuracy_z_value=accuracy_z_value,
                                         flags=flags, trial_by_trial_accuracy=trial_by_trial_accuracy,
                                         manual_threshold=mt, remove_dims=removal_dim_indicies,
                                         category_independence_enabled=category_independence_enabled,
                                         order_greedy_deanonymization_enabled=order_greedy_deanonymization_enabled)
        # noinspection PySimplifyBooleanCheck
        if results != []:
            if category_independence_enabled:
                for cat_result in results:
                    output_results(cat_result, collapse_trials, agg_functions, out_fp, label)
            else:
                output_results(results, collapse_trials, agg_functions, out_fp, label)

    out_fp.close()

    logging.info('Done processing all files. Data can be found in {0}.'.format(out_filename))


def output_results(results, collapse_trials, agg_functions, out_fp, label):
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
            # noinspection PyBroadException
            try:
                result.append(agg_functions[iidx]([row[iidx] for row in results]))
            except Exception:
                logging.error('index {0} in result {1} failed to aggregate, '
                              'nan will be returned instead'.format(iidx, np.array(results)[:, iidx]))
                result.append(np.nan)
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
