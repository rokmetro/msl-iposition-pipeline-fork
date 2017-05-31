from full_pipeline import PipelineFlags
from operator import attrgetter


class TrialData:
    def __init__(self, actual_points=None, data_points=None, actual_labels=None, data_labels=None):
        if data_labels is None:
            data_labels = []
        if actual_labels is None:
            actual_labels = []
        if data_points is None:
            data_points = []
        if actual_points is None:
            actual_points = []
        self.actual_points = actual_points
        self.data_points = data_points
        self.actual_labels = actual_labels
        self.data_labels = data_labels
        self.distance_accuracy_map = None
        self.distance_threshold = None


class ParticipantData:
    def __init__(self, trials, identity=None):
        self.trials = trials
        self.id = identity

    def get_all_from_trials(self, attribute):
        return list(map(attrgetter(attribute), self.trials))

    def set_all_to_trials(self, attribute, value):
        for v, trial in zip(value, self.trials):
            setattr(trial, attribute, v)

    @property
    def actual_points(self):
        return self.get_all_from_trials("actual_points")

    @property
    def data_points(self):
        return self.get_all_from_trials("data_points")

    @property
    def actual_labels(self):
        return self.get_all_from_trials("actual_labels")

    @property
    def data_labels(self):
        return self.get_all_from_trials("data_labels")

    @property
    def distance_accuracy_map(self):
        return self.get_all_from_trials("distance_accuracy_map")

    @property
    def distance_threshold(self):
        return self.get_all_from_trials("distance_threshold")

    @actual_points.setter
    def actual_points(self, value):
        self.set_all_to_trials("actual_points", value)

    @data_points.setter
    def data_points(self, value):
        self.set_all_to_trials("data_points", value)

    @actual_labels.setter
    def actual_labels(self, value):
        self.set_all_to_trials("actual_labels", value)

    @data_labels.setter
    def data_labels(self, value):
        self.set_all_to_trials("data_labels", value)

    @distance_accuracy_map.setter
    def distance_accuracy_map(self, value):
        self.set_all_to_trials("distance_accuracy_map", value)

    @distance_threshold.setter
    def distance_threshold(self, value):
        self.set_all_to_trials("distance_threshold", value)


class AnalysisConfiguration:
    # noinspection PyDefaultArgument
    def __init__(self, z_value=1.96,
                 trial_by_trial_accuracy=True, manual_threshold=True,
                 flags=PipelineFlags.All,
                 debug_labels=['']):
        self.z_value = z_value
        self.trial_by_trial_accuracy = trial_by_trial_accuracy
        self.manual_threshold = manual_threshold
        self.flags = flags
        self.debug_labels = debug_labels
