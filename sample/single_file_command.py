import logging
import argparse
import sys
from cogrecon.core.full_pipeline import full_pipeline
from cogrecon.core.data_structures import ParticipantData, AnalysisConfiguration, PipelineFlags
from cogrecon.core.file_io import get_id_from_file_prefix_via_suffix

# TODO: Documentation needs an audit/overhaul

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    parser.add_argument('actual_coordinates', type=str, help='the path to the file containing the actual coordinates')
    parser.add_argument('data_coordinates', type=str, help='the path to the file containing the data coordinates')
    parser.add_argument('num_trials', type=int, help='the number of trials in the file')
    parser.add_argument('num_items', type=int, help='the number of items to be analyzed')
    parser.add_argument('line_number', type=int, help='the line number to be processed (starting with 0) - typically '
                                                      'the trial number minus 1.')
    parser.add_argument('--pipeline_mode', type=int, help='the mode in which the pipeline should process; \n\t0 for '
                                                          'just accuracy+swaps, \n\t1 for '
                                                          'accuracy+deanonymization+swaps, \n\t2 for accuracy+global '
                                                          'transformations+swaps, \n\t3 for '
                                                          'accuracy+deanonymization+global transformations+swaps \n'
                                                          '(default is 3)', default=3)
    parser.add_argument('--accuracy_z_value', type=float, help='the z value to be used for accuracy exclusion ('
                                                               'default is 1.96), corresponding to 95% confidence; if ',
                        default=1.96)
    parser.add_argument('--dimension', type=int, help='the dimensionality of the data (default is 2)', default=2)

    parser.add_argument('--order_file', type=str, help='the path to the file containing the order information with '
                                                       'respect to data_coordinates',
                        default=None)
    parser.add_argument('--category_file', type=str, help='the path to the file containing the category data with '
                                                          'respect to both actual_coordinates and data_coordinates',
                        default=None)

    if len(sys.argv) > 1:
        args = parser.parse_args()

        # Generate configuration for analysis from args
        _analysis_configuration = AnalysisConfiguration(z_value=args.accuracy_z_value,
                                                        flags=PipelineFlags(args.pipeline_mode),
                                                        debug_labels=[get_id_from_file_prefix_via_suffix(
                                                            args.data_coordinates, "position_data_coordinates.txt"),
                                                            args.line_number])
        # Load participant data
        _participant_data = ParticipantData.load_from_file(args.actual_coordinates, args.data_coordinates,
                                                           (args.num_trials, args.num_items, args.dimension),
                                                           order_filepath=args.order_file,
                                                           category_filepath=args.category_file)
        # Because we're just visualizing one trial, strip away all but the requested trial
        _participant_data.trials = [_participant_data.trials[args.line_number]]

        # Run the pipeline
        full_pipeline(_participant_data, _analysis_configuration, visualize=True)

        exit()

    logging.info("No arguments found - quitting.")
