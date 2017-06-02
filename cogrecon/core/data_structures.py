from enum import Enum
from operator import attrgetter
import numpy as np
import inspect

from .file_io import get_coordinates_from_file

# TODO: Documentation needs an audit/overhaul


class PipelineFlags(Enum):
    Unknown = 0
    Simple = 0
    Deanonymize = 1
    GlobalTransformation = 2
    All = 3
    value = 0

    def __or__(self, other):
        return PipelineFlags(self.value | other.value)

    def __eq__(self, other):
        return (self.value & other.value) != 0


class TrialData(object):
    def __init__(self, actual_points=None, data_points=None, actual_labels=None, data_labels=None):
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

    @property
    def distance_accuracy_map(self):
        return self._distance_accuracy_map

    @property
    def distance_threshold(self):
        return self._distance_threshold

    @property
    def actual_points(self):
        return self._actual_points

    @property
    def data_points(self):
        return self._data_points

    @property
    def actual_labels(self):
        return self._actual_labels

    @property
    def data_labels(self):
        return self._data_labels

    @actual_points.setter
    def actual_points(self, value):
        assert isinstance(value, list), "TrialData actual_points must be type list"
        assert len(value) > 0, "TrialData actual_points must be non-empty"
        assert all([isinstance(_x, float) for _x in np.ndarray.flatten(np.array(value))]), \
            "TrialData actual_points must contain only floats"
        self._actual_points = value

    @data_points.setter
    def data_points(self, value):
        assert isinstance(value, list), "TrialData data_points must be type list"
        assert len(value) > 0, "TrialData data_points must be non-empty"
        assert all([isinstance(_x, float) for _x in np.ndarray.flatten(np.array(value))]), \
            "TrialData actual_points must contain only floats"
        self._data_points = value

    @actual_labels.setter
    def actual_labels(self, value):
        assert isinstance(value, list), "TrialData actual_labels must be type list"
        assert np.unique(value).shape == np.array(value).shape, \
            "swaps actual_labels are not unique: {0}".format(value)
        self._actual_labels = value

    @data_labels.setter
    def data_labels(self, value):
        assert isinstance(value, list), "TrialData data_labels must be type list"
        assert np.unique(value).shape == np.array(value).shape, \
            "swaps data_labelsare not unique: {0}".format(value)
        self._data_labels = value

    @distance_accuracy_map.setter
    def distance_accuracy_map(self, value):
        assert isinstance(value, list), "TrialData distance_accuracy_map must be type list"
        assert all([isinstance(x, (bool, np.bool_)) for x in value]), \
            "TrialData distance_accuracy_map must only contain bools"
        self._distance_accuracy_map = value

    @distance_threshold.setter
    def distance_threshold(self, value):
        assert isinstance(value, float), "TrialData distance_threshold must be type float"
        self._distance_threshold = value


class ParticipantData(object):
    def __init__(self, trials, identity=None):
        assert isinstance(trials, list) and len(trials) > 0 and \
               all([isinstance(trial, TrialData) for trial in trials]), \
               "ParticipantData trials must be a non-empty list containing TrialData objects"
        self.__dict__['trials'] = None
        self.__dict__['id'] = None

        self.trials = trials
        self.id = identity

    def get_all_from_trials(self, attribute):
        return list(map(attrgetter(attribute), self.trials))

    def set_all_to_trials(self, attribute, value):
        for v, trial in zip(value, self.trials):
            setattr(trial, attribute, v)

    def __getattr__(self, attribute):
        if attribute in self.__dict__.keys():
            return self.__dict__[attribute]
        elif attribute in [x[0] for x in inspect.getmembers(TrialData([0.], [0.])) if not x[0].startswith('__')]:
            return self.get_all_from_trials(attribute)
        raise AttributeError('asd')

    def __setattr__(self, attribute, value):
        if attribute in self.__dict__.keys():
            self.__dict__[attribute] = value
        elif attribute in [x[0] for x in inspect.getmembers(TrialData([0.], [0.])) if not x[0].startswith('__')]:
            self.set_all_to_trials(attribute, value)
        else:
            self.__dict__[attribute] = value

    @staticmethod
    def load_from_file(actual_coordinates_filepath, data_coordinates_filepath, expected_shape):
        actual = get_coordinates_from_file(actual_coordinates_filepath, expected_shape)
        data = get_coordinates_from_file(data_coordinates_filepath, expected_shape)
        _participant_data = ParticipantData([TrialData(_a, _d) for _a, _d in zip(actual, data)])

        return _participant_data


class AnalysisConfiguration:
    # noinspection PyDefaultArgument
    def __init__(self, z_value=1.96,
                 trial_by_trial_accuracy=True, manual_threshold=None,
                 flags=PipelineFlags(PipelineFlags.All),
                 debug_labels=['']):
        self.z_value = float(z_value)
        self.trial_by_trial_accuracy = bool(trial_by_trial_accuracy)
        if manual_threshold is not None:
            assert isinstance(manual_threshold, float), "AnalysisConfiguration manual_threshold must be of type float"
        self.manual_threshold = manual_threshold
        self.debug_labels = list(debug_labels)

        assert isinstance(flags, PipelineFlags), 'AnalysisConfiguration flags must be of type PipelineFlags'

        self.flags = flags
