import logging
import argparse
import sys

import numpy as np
import scipy.stats as st

from cogrecon.core.full_pipeline import full_pipeline
from cogrecon.core.data_structures import ParticipantData, AnalysisConfiguration, PipelineFlags
from cogrecon.core.file_io import get_id_from_file_prefix_via_suffix
from cogrecon.core.data_flexing.dimension_removal import remove_dimensions
from cogrecon.core.cogrecon_globals import default_z_value, default_pipeline_flags, default_dimensions, \
    data_coordinates_file_suffix

"""
This module is meant to be run exclusively from the command line via:

python single_file_command.py <arguments>

.

Note that this function assumes its input is meant to be visualized and will act accordingly.
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Parse the inputs to the batch_pipeline function (see help=??? for details on each param)
    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    # Required File Configuration
    parser.add_argument('actual_coordinates', type=str,
                        help='the path to the file containing the actual coordinates')
    parser.add_argument('data_coordinates', type=str,
                        help='the path to the file containing the data coordinates')
    # Note: No File Out Configuration because single_file_command.py is meant for running on single trials for
    # visualization and other investigative purposes
    # Dimensionality Configuration
    parser.add_argument('dimension', type=int,
                        help='the dimensionality of the data (default is {0})'.format(default_dimensions))
    parser.add_argument('num_trials', type=int,
                        help='the number of trials in the file')
    parser.add_argument('num_items', type=int,
                        help='the number of items to be analyzed')
    parser.add_argument('line_number', type=int,
                        help='the line number to be processed (starting with 0) - typically the last trial number '
                             'minus 1.')
    # Run Configuration
    parser.add_argument('--pipeline_mode', type=int, default=int(default_pipeline_flags),
                        help='the mode in which the pipeline should process; \n\t0 for just accuracy+swaps, \n\t1 for '
                             'accuracy+deanonymization+swaps, \n\t2 for accuracy+global transformations+swaps, '
                             '\n\t3 for accuracy+deanonymization+global transformations+swaps \n(default is '
                             '{0})'.format(int(default_pipeline_flags)))
    # Accuracy Configuration
    parser.add_argument('--accuracy_z_value', type=float, default=default_z_value,
                        help='the z value to be used for accuracy exclusion (default is {0}), corresponding to {1}% '
                             'confidence'.format(default_z_value,
                                                 int(np.round((1 - st.norm.sf(default_z_value) * 2) * 100))))
    # Category Configuration
    parser.add_argument('--category_file', type=str, default=None,
                        help='the path to the file containing the category data with respect to both '
                             'actual_coordinates and data_coordinates')
    # Order Configuration
    parser.add_argument('--order_file', type=str, default=None,
                        help='the path to the file containing the order information with respect to data_coordinates')
    # Visualization Settings
    parser.add_argument('--plot_extent', type=float, nargs=4, default=None,
                        help='a list of length 4 of values representing the visual extent of the visualization plot. '
                             'the values are ordered x_min, x_max, y_min, y_max. if None is provided, extent will be '
                             'determined automatically.')
    # Data Flexing/Reorganization
    parser.add_argument('--remove_dims', metavar='N', type=int, nargs='+', default=None,
                        help='a list of dimensions (starting with 0) to remove from processing (default is None)')

    # If we receive some parameters
    if len(sys.argv) > 1:
        # Attempt to parse the parameters
        args = parser.parse_args()

        # Generate configuration for analysis from args
        _analysis_configuration = AnalysisConfiguration(z_value=args.accuracy_z_value,
                                                        flags=PipelineFlags(args.pipeline_mode),
                                                        debug_labels=[get_id_from_file_prefix_via_suffix(
                                                            args.data_coordinates, data_coordinates_file_suffix),
                                                            args.line_number])
        # Load participant data
        _participant_data = ParticipantData.load_from_file(args.actual_coordinates, args.data_coordinates,
                                                           (args.num_trials, args.num_items, args.dimension),
                                                           order_filepath=args.order_file,
                                                           category_filepath=args.category_file)
        # Because we're just visualizing one trial, strip away all but the requested trial
        _participant_data.trials = [_participant_data.trials[args.line_number]]

        # If it has been requested to remove particular dimensions, remove them
        if args.remove_dims is not None:
            _participant_data = remove_dimensions(_participant_data, removal_dim_indices=args.remove_dims)

        # If a particular extent has been requeste for visualization, validate its form/shape
        if args.plot_extent is not None:
            plot_extent = np.array(args.plot_extent).reshape((2, 2)).tolist()
        else:
            plot_extent = None

        # Run the pipeline with visualization
        full_pipeline(_participant_data, _analysis_configuration, visualize=True, visualization_extent=plot_extent)

    else:
        # Quit gracefully if no parameters were provided
        logging.info("No arguments found - quitting.")
