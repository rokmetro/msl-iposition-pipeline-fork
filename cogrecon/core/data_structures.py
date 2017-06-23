from enum import Enum
from operator import attrgetter
import numpy as np
import inspect

from .file_io import get_coordinates_from_file
from .cogrecon_globals import default_z_value, default_pipeline_flags


class PipelineFlags(Enum):
    """
    This class serves as an Enum in order to describe which, if any, pipeline steps should be run on execution.

    It acts as a set of flags for each pipeline step (Deanonymize and Global Transform currently), enabling them or
    disabling them as appropriate. When used as an integer, the integer value can be through of as a binary value whose
    digit places are associated with boolean values specifying the enabled/disabled state of each pipeline step.

    .. note::

       Simple and Unknown both disable all pipeline stages except accuracy and cycle/swap analysis. All runs all steps.

    """
    Unknown = 0
    Simple = 0
    Deanonymize = 1
    GlobalTransformation = 2
    All = 3
    value = 0

    def __or__(self, other):
        """

        :param other: the other value to be 'or'-ed with
        :return: the result of the bitwise 'or' operation
        """
        return PipelineFlags(self.value | other.value)

    def __eq__(self, other):
        """

        :param other: the value against which equality should be compared
        :return: the result of the equality check (an 'and' operation comparing each bit)
        """
        return (self.value & other.value) != 0


class TrialData(object):
    """

    This class is used to store individual trial data. Some functions can and should be only run on a single trial
    worth of data rather than a whole participant worth.

    .. note::

       This object is typically contained within a ParticipantData object.

    """
    def __init__(self, actual_points=None, data_points=None, actual_labels=None, data_labels=None,
                 cateogry_labels=None, data_order=None):
        """

        :param actual_points:
        :param data_points:
        :param actual_labels:
        :param data_labels:
        :param cateogry_labels:
        :param data_order:
        """
        if data_labels is None:
            data_labels = []
        if actual_labels is None:
            actual_labels = []
        if data_points is None:
            data_points = []
        if actual_points is None:
            actual_points = []

        assert all(np.sort(actual_labels) == np.sort(data_labels)), \
            ("swaps actual_labels and data_labels are " +
             "not unequal: actual, {0}; data, {1}").format(actual_labels, data_labels)

        self._actual_points = None
        self._data_points = None
        self._actual_labels = None
        self._data_labels = None
        self._distance_accuracy_map = None
        self._distance_threshold = None
        self._category_labels = None
        self._data_order = None

        self.actual_points = actual_points
        self.data_points = data_points
        if not actual_labels:
            self.actual_labels = range(0, len(self.actual_points))
        else:
            self.actual_labels = actual_labels
        if not data_labels:
            self.data_labels = range(0, len(self.data_points))
        else:
            self.data_labels = data_labels

        if cateogry_labels is not None:
            self.category_labels = cateogry_labels
        if data_order is not None:
            self.data_order = data_order

    @property
    def category_labels(self):
        """

        :return: the category label list
        """
        return self._category_labels

    @property
    def data_order(self):
        """

        :return: the data order list
        """
        return self._data_order

    @property
    def distance_accuracy_map(self):
        """

        :return: the distance accuracy map, if set by the accuracy function
        """
        return self._distance_accuracy_map

    @property
    def distance_threshold(self):
        """

        :return: the distance threshold, if set by the accuracy function
        """
        return self._distance_threshold

    @property
    def actual_points(self):
        """

        :return: the actual/correct/target points
        """
        return self._actual_points

    @property
    def data_points(self):
        """

        :return: the participant data points
        """
        return self._data_points

    @property
    def actual_labels(self):
        """

        :return: the actual/correct/target labels
        """
        return self._actual_labels

    @property
    def data_labels(self):
        """

        :return: the participant data labels
        """
        return self._data_labels

    @actual_points.setter
    def actual_points(self, value):
        """

        :param value: a float list of actual/correct/target point lists (i.e. [[0., 0.], [1., 1.,]] etc)
        """
        assert isinstance(value, list), "TrialData actual_points must be type list"
        # assert len(value) > 0, "TrialData actual_points must be non-empty"
        assert all([isinstance(_x, float) for _x in np.ndarray.flatten(np.array(value))]), \
            "TrialData actual_points must contain only floats"
        self._actual_points = value

    @data_points.setter
    def data_points(self, value):
        """

        :param value: a float list of data point lists (i.e. [[0., 0.], [1., 1.,]] etc)
        """
        assert isinstance(value, list), "TrialData data_points must be type list"
        # assert len(value) > 0, "TrialData data_points must be non-empty"
        assert all([isinstance(_x, float) for _x in np.ndarray.flatten(np.array(value))]), \
            "TrialData actual_points must contain only floats"
        self._data_points = value

    @actual_labels.setter
    def actual_labels(self, value):
        """

        :param value: a list of unique values of any type to label the actual/correct/target points
        """
        assert isinstance(value, list), "TrialData actual_labels must be type list"
        assert np.unique(value).shape == np.array(value).shape, \
            "swaps actual_labels are not unique: {0}".format(value)
        self._actual_labels = value

    @data_labels.setter
    def data_labels(self, value):
        """

        :param value: a list of unique values of any type to label the participant data points
        """
        assert isinstance(value, list), "TrialData data_labels must be type list"
        assert np.unique(value).shape == np.array(value).shape, \
            "swaps data_labelsare not unique: {0}".format(value)
        self._data_labels = value

    @distance_accuracy_map.setter
    def distance_accuracy_map(self, value):
        """

        :param value: a list of bool values describing the accuracy of each data_points element
        """
        assert isinstance(value, list), "TrialData distance_accuracy_map must be type list"
        assert all([isinstance(x, (bool, np.bool_)) for x in value]), \
            "TrialData distance_accuracy_map must only contain bools"
        self._distance_accuracy_map = value

    @distance_threshold.setter
    def distance_threshold(self, value):
        """

        :param value: a float describing the threshold used to determine the distance_accuracy_map
        """
        assert isinstance(value, float) or value is None, "TrialData distance_threshold must be type float"
        self._distance_threshold = value

    @category_labels.setter
    def category_labels(self, value):
        """

        :param value: a list of labels for the categories of each data_points element
        """
        assert isinstance(value, list), "TrialData category_labels must be type list"
        self._category_labels = value

    @data_order.setter
    def data_order(self, value):
        """

        :param value: a list of integers describing the order in which data_points elements were placed
        """
        assert isinstance(value, list), "TrialData data_order must be type list"
        assert all([isinstance(x, int) for x in value]), "TrialData data_order must only contain int"
        self._data_order = value


class ParticipantData(object):
    """
    This object contains an entire participant's data across all trials. It really just wraps up the TrialData object
    in a list such that any attribute in TrialData can be called in ParticipantData, but the getting and setting
    functionality will include the trial dimension. This allows data to be conveniently gotten and set in a
    multi-dimensional way rather than always trial-by-trial.

    .. note::

       All attributes in TrialData can be called in ParticipantData.

    """
    def __init__(self, trials, identity=None):
        """

        :param trials: a non-empty list of TrialData objects
        :param identity: a label for the ParticipantData
        """
        assert isinstance(trials, list) and len(trials) > 0 and all([isinstance(t, TrialData) for t in trials]), \
            "ParticipantData trials must be a non-empty list containing TrialData objects"
        self.__dict__['trials'] = None
        self.__dict__['id'] = None

        self.trials = trials
        self.id = identity

    def get_all_from_trials(self, attribute):
        """
        A helper function which wil lget all the attributes in the TrialData in self.trials as a list of values.

        :param attribute: the attribute in a TrialData object in self.trials to get
        :return: a list of values from every self.trials element corresponding to the given attribute
        """
        return list(map(attrgetter(attribute), self.trials))

    def set_all_to_trials(self, attribute, value):
        """
        A helper function which will set all the attributes in the TrialData in self.trials to a list of values.

        :param attribute: the attribute in a TrialData object in self.trials to set
        :param value: the value list to which each self.trials element should be set
        """
        for v, trial in zip(value, self.trials):
            setattr(trial, attribute, v)

    def __getattr__(self, attribute):
        """

        :param attribute: the attribute to be gotten
        :return: the value at the attribute
        """
        if attribute in self.__dict__.keys():
            return self.__dict__[attribute]
        elif attribute in [x[0] for x in inspect.getmembers(TrialData([0.], [0.])) if not x[0].startswith('__')]:
            return self.get_all_from_trials(attribute)
        raise AttributeError('asd')

    def __setattr__(self, attribute, value):
        """

        :param attribute: the attribute to be set
        :param value: the value to which the attribute should be set
        """
        if attribute in self.__dict__.keys():
            self.__dict__[attribute] = value
        elif attribute in [x[0] for x in inspect.getmembers(TrialData([0.], [0.])) if not x[0].startswith('__')]:
            self.set_all_to_trials(attribute, value)
        else:
            self.__dict__[attribute] = value

    # noinspection PyUnusedLocal
    @staticmethod
    def category_split_participant(original_participant_data, unique_categories):

        # noinspection PyTypeChecker
        """
        This helper function splits a ParticipantData object into multiple objects containing unique categories defined
        by unique_categories. If an item has no category label or does not match one of the labels in unique_categories,
        it will be added to the unknown_category_participant_data object (the second output in output tuple)

        :param original_participant_data: a ParticipantData object containing categorical data
        :param unique_categories: a unique list of category labels
        :return: a tuple whose first element is a list of ParticipantData objects associated one-to-one with each
                 unique category and whose second element is a ParticipantData object containing all unlabelled or
                 unknown category items
        """

        # Confirm unique_categories are, in fact, unique
        # noinspection PyTypeChecker
        assert len(np.unique(np.array(unique_categories)).tolist()) == len(np.array(unique_categories).tolist())

        # Generate a split data object for each category
        # noinspection PyTypeChecker
        split_participant_data = [ParticipantData([TrialData([], [], [], [], [], [])
                                                   for _ in original_participant_data.trials])
                                  for _ in range(len(unique_categories))]
        for dat in split_participant_data:
            for t in dat.trials:
                t.distance_accuracy_map = []  # Fill accuracy map with empty lists so we can copy that too

        # Generate an additional data object for unknown category data
        unknown_category_participant_data = ParticipantData([TrialData([], [], [], [], [], [])
                                                             for _ in original_participant_data.trials])
        for t in unknown_category_participant_data.trials:
            t.distance_accuracy_map = []  # Fill accuracy map with empty lists so we can copy that too

        # Iterate through all the trials in the original data set
        for trial_idx, t in enumerate(original_participant_data.trials):
            if not t.distance_accuracy_map:
                t_d_accuracy_map = [False] * len(t.data_points)
            else:
                t_d_accuracy_map = t.distance_accuracy_map

            if not t.data_order:
                t_d_data_order = [None] * len(t.data_points)
            else:
                t_d_data_order = t.data_order

            # Iterate through all the points in the trial
            for i, (d_points, a_points, d_labels, a_labels, c_labels, d_order, d_accuracy_map) \
                    in enumerate(zip(t.data_points, t.actual_points, t.data_labels, t.actual_labels,
                                     t.category_labels, t_d_data_order, t_d_accuracy_map)):
                # Store the distance threshold for the trial for convenience
                dt = t.distance_threshold
                # Check if there is a category label
                cl = None
                if c_labels is not None:
                    cl = t.category_labels[i]
                # If there is a category label and it is in our list of unique categories
                if cl is not None and cl in unique_categories:
                    # Get the index of the output data in which to store the point
                    idx = unique_categories.index(cl)
                    # Store the distance threshold
                    split_participant_data[idx].trials[trial_idx].distance_threshold = dt

                    # Go through each list and, if there is a valid point at that index, append it to the new object
                    if d_points is not None:
                        split_participant_data[idx].trials[trial_idx].data_points.append(d_points)
                    if d_labels is not None:
                        split_participant_data[idx].trials[trial_idx].data_labels.append(d_labels)
                    if a_points is not None:
                        split_participant_data[idx].trials[trial_idx].actual_points.append(a_points)
                    if a_labels is not None:
                        split_participant_data[idx].trials[trial_idx].actual_labels.append(a_labels)
                    if c_labels is not None:
                        split_participant_data[idx].trials[trial_idx].category_labels.append(c_labels)
                    if d_order is not None:
                        split_participant_data[idx].trials[trial_idx].data_order.append(d_order)
                    if d_accuracy_map is not None:
                        split_participant_data[idx].trials[trial_idx].distance_accuracy_map.append(d_accuracy_map)
                # If either there was no category information or it was not in our list, the points belong in the
                # unknown category object
                else:
                    # Set the distance threshold
                    unknown_category_participant_data.trials[trial_idx].distance_threshold = dt

                    # Go through each list and, if there is a valid point at that index, append it to the new object
                    if d_points is not None:
                        split_participant_data[idx].trials[trial_idx].data_points.append(d_points)
                    if d_labels is not None:
                        split_participant_data[idx].trials[trial_idx].data_labels.append(d_labels)
                    if a_points is not None:
                        split_participant_data[idx].trials[trial_idx].actual_points.append(a_points)
                    if a_labels is not None:
                        split_participant_data[idx].trials[trial_idx].actual_labels.append(a_labels)
                    if c_labels is not None:
                        split_participant_data[idx].trials[trial_idx].category_labels.append(c_labels)
                    if d_order is not None:
                        split_participant_data[idx].trials[trial_idx].data_order.append(d_order)
                    if d_accuracy_map is not None:
                        split_participant_data[idx].trials[trial_idx].distance_accuracy_map.append(d_accuracy_map)

        # return the list of unique_categories sorted data and the object with unknown cateogry data
        return split_participant_data, unknown_category_participant_data

    @staticmethod
    def load_from_file(actual_coordinates_filepath, data_coordinates_filepath, expected_shape,
                       category_filepath=None, order_filepath=None):
        """
        This helper function takes paths to data files and an expected shape and generates a ParticipantData object
        which contains the file data.

        :param actual_coordinates_filepath: the path to the actual_coordinates.txt file, containing actual/correct/target
                                            data coordinates
        :param data_coordinates_filepath: the path to the position_data_coordinates.txt file, containing
                                          participant data coordinates
        :param expected_shape: a tuple of integers specifying the shape the data should be
        :param category_filepath: a path to the category.txt file, specifying item categories
        :param order_filepath: a path to the order.txt file, specifying order of placement

        :return: the ParticipantData object containing all appropriate information given the input file paths
        """
        actual = get_coordinates_from_file(actual_coordinates_filepath, expected_shape)
        data = get_coordinates_from_file(data_coordinates_filepath, expected_shape)
        category = [None] * len(actual)
        order = [None] * len(actual)
        if category_filepath is not None:
            # categories are always 1D, so we strip the dimensionality and add a [1] to the list
            # categories can be any type, so we set type to None
            category = get_coordinates_from_file(category_filepath, tuple(list(expected_shape[:2]) + [1]),
                                                 data_type=None)
        if order_filepath is not None:
            # order is always 1D, so we strip the dimensionality and add a [1] to the list
            # order should be an integer, so we set type to int
            order = get_coordinates_from_file(order_filepath, tuple(list(expected_shape[:2]) + [1]),
                                              data_type=int)
        _participant_data = ParticipantData([TrialData(_a, _d, cateogry_labels=_c, data_order=_o)
                                             for _a, _d, _o, _c in zip(actual, data, category, order)])

        return _participant_data


class AnalysisConfiguration:
    """
    The AnalysisConfiguration object is used to store the various global analysis variables of interest which
    determine the behavior of the pipeline. It is passed to functions which require one of these variables.


    """
    # noinspection PyDefaultArgument
    def __init__(self, z_value=default_z_value,
                 trial_by_trial_accuracy=True, manual_threshold=None,
                 flags=PipelineFlags(default_pipeline_flags),
                 greedy_order_deanonymization=False,
                 process_categories_independently=False, is_category=False, category_label=None,
                 debug_labels=['']):
        """

        :param z_value: a value (float or int) representing the z threshold for counting something as accurate
        :param trial_by_trial_accuracy: when True, z_value thresholds are used on a trial-by-trial basis for
                                        accuracy calculations, when False, the thresholds are computed then collapsed
                                        across an individual's trials
        :param manual_threshold: a list of manual swap threshold values associated with the specified participant
                                 prefixes and trials in the batch process (should be of the same length as the number
                                 of trials)
        :param flags: the value (PipelineFlags) describing what pipeline elements should/should not be run on the data
        :param greedy_order_deanonymization: whether the greedy, order based deanonymization method
                                             should be used in determining the mapping of object to location.
                                             Note that if enabled, an order file (or files) is expected.
        :param process_categories_independently: whether the items involved have associated categorical information
                                                 such that they should be processed independently.
                                                 Note that if enabled, a category file (or files) is expected.
        :param is_category: if true, this particular run of data is categorical
        :param category_label: the label of this category (only valid if is_category is True)
        :param debug_labels: a list of labels to be printed during debugging
        """
        self.z_value = float(z_value)
        self.trial_by_trial_accuracy = bool(trial_by_trial_accuracy)
        if manual_threshold is not None:
            assert isinstance(manual_threshold, float), "AnalysisConfiguration manual_threshold must be of type float"
        self.manual_threshold = manual_threshold

        self.greedy_order_deanonymization = bool(greedy_order_deanonymization)
        self.category_independence = bool(process_categories_independently)
        self.is_category = bool(is_category)
        self.category_label = category_label
        self.debug_labels = list(debug_labels)

        assert isinstance(flags, PipelineFlags), 'AnalysisConfiguration flags must be of type PipelineFlags'

        self.flags = flags
