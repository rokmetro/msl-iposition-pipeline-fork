import numpy as np
import copy
from ..data_structures import ParticipantData
from ..tools import validate_type


def remove_dimensions(participant_data, removal_dim_indices=None):
    # Validate data types
    """

    :param participant_data:
    :param removal_dim_indices:
    :return:
    """
    validate_type(removal_dim_indices, list, "removal_dim_indices", "remove_dimensions")
    validate_type(participant_data, ParticipantData, "participant_data", "remove_dimensions")
    assert [isinstance(x, int) for x in removal_dim_indices], "remove_dimensions dim_indices must only contain int"

    # Copy the data
    data = copy.deepcopy(participant_data)

    # If none, just return the copy
    if removal_dim_indices is None:
        return data

    # Extract the trial-by-trial dimensionality of the data
    actual_dimensions = [len(t.actual_points[0]) for t in data.trials]
    data_dimensions = [len(t.data_points[0]) for t in data.trials]

    # Find the unique dimensionalities
    unique_actual_dimensions = np.unique(actual_dimensions)
    unique_data_dimensions = np.unique(data_dimensions)

    # Confirm that we have one unique dimension in both actual and data
    assert len(unique_actual_dimensions) == 1 and len(unique_data_dimensions) == 1, \
        'The dimensions of the data across all trials are not identical, so dimensional removal is not possible (' \
        'found actual dims {0} and data dims {1}).'.format(unique_actual_dimensions, unique_data_dimensions)

    # Confirm that actual and data dimension are the same
    assert unique_actual_dimensions[0] == unique_data_dimensions[0], \
        'The dimensions of the participant data ({0}) do not match the label data ({1}), so dimensional removal is ' \
        'not possible '.format(unique_actual_dimensions[0], unique_data_dimensions[0])

    # Store dimensionality of data
    dim = unique_actual_dimensions[0]

    # Confirm that all requested removal indicies are within the appropriate range
    assert all([dim_idx < dim for dim_idx in removal_dim_indices]), \
        'The provided removal_dim_indices are not all less than the number of available dimensions ({0}).'.format(dim)

    assert np.unique(removal_dim_indices).tolist() == removal_dim_indices, \
        'Duplicate indicies found in remova_dim_indices. Please provide only unique indicies.'

    mask = [1 for i in range(0, dim) if i not in removal_dim_indices]

    data.actual_points = np.transpose(np.array(data.actual_points))[mask]
    data.data_points = np.transpose(np.array(data.data_points))[mask]

    return data
