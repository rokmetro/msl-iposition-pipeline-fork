import logging
import argparse
import sys
import easygui
import datetime
from cogrecon.core.batch_pipeline import batch_pipeline
from cogrecon.core.data_structures import PipelineFlags

# TODO: Documentation needs an audit/overhaul
# TODO: Modify for category and order data

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    parser.add_argument('--search_directory', type=str, help='the root directory in which to search for the actual and '
                                                             'data coordinate files (actual_coordinates.txt and '
                                                             '###position_data_coordinates.txt, respectively)',
                        default=None)

    parser.add_argument('--category_independence_enabled', type=int,
                        help='if 0, the program will run without an assumption of categories. if 1, the program will '
                             'search the search_directory for a category file (categories.txt) with the appropriate '
                             'shape (same as actual_coordinates.txt), and the category data will be used to break up '
                             'the analysis such that item categories will be processed independently from one '
                             'another.',
                        default=0)
    parser.add_argument('--order_greedy_deanonymization_enabled', type=int,
                        help='if 0, the program will run without using order information in deanonymization (a global '
                             'minimum will be used). if 1, the program will take a greedy approach to '
                             'deanonymization. First it will search for ###order.txt files which should be associated '
                             'with the ###position_data_coordinates.txt files in a 1-to-1 fashion. Then the item '
                             'being placed first will be associated with its minimum-distance true value, '
                             'then the second will be associated with the minimum distance remaining values, '
                             'etc until all values are associated. This effectively weights the importance of '
                             'deanonymization minimization in accordance with the placement order.',
                        default=0)

    parser.add_argument('--num_trials', type=int, help='the number of trials in each file', default=None)
    parser.add_argument('--num_items', type=int, help='the number of items to be analyzed', default=None)
    parser.add_argument('--pipeline_mode', type=int, help='the mode in which the pipeline should process; \n\t0 for '
                                                          'just accuracy+swaps, \n\t1 for '
                                                          'accuracy+deanonymization+swaps, \n\t2 for accuracy+global '
                                                          'transformations+swaps, \n\t3 for '
                                                          'accuracy+deanonymization+global transformations+swaps \n('
                                                          'default is 3)', default=3)
    parser.add_argument('--accuracy_z_value', type=float, help='the z value to be used for accuracy exclusion ('
                                                               'default is 1.96, corresponding to 95% confidence',
                        default=1.96)
    parser.add_argument('--collapse_trials', type=int, help='if 0, one row per trial will be output, otherwise one '
                                                            'row per participant will be output (default is 1)',
                        default=1)
    parser.add_argument('--dimension', type=int, help='the dimensionality of the data (default is 2)', default=2)
    parser.add_argument('--trial_by_trial_accuracy', type=int, help='when not 0, z_value thresholds are used on a '
                                                                    'trial-by-trial basis for accuracy calculations, '
                                                                    'when 0, the thresholds are computed then '
                                                                    'collapsed across an individual\'s trials',
                        default=1)
    parser.add_argument('--prefix_length', type=int, help='the length of the subject ID prefix at the beginning of '
                                                          'the data filenames (default is 3)', default=3)
    parser.add_argument('--actual_coordinate_prefixes', type=int, help='if 0, the normal assumption that all '
                                                                       'participants used the same '
                                                                       'actual_coordinates.txt file will be used. if '
                                                                       'not 0, it is assumed that all '
                                                                       'actual_coordinates.txt files have a prefix '
                                                                       'which is matched in the '
                                                                       'position_data_coordinates.txt prefix. Thus, '
                                                                       'there should be a one-to-one correspondance '
                                                                       'between actual_coordinates.txt and '
                                                                       'position_data_coordinates.txt files and their '
                                                                       'contents.', default=0)
    parser.add_argument('--manual_swap_accuracy_threshold_list', type=str,
                        help='if empty string or none, the value is ignored. if a string (path) pointing to a text '
                             'file containing a new line separated list of id,threshold pairs is provided, '
                             'any files whose participant id matches the first matching id in the list will have the '
                             'associated threshold applied instead of being automatically computed.',
                        default='')
    if len(sys.argv) > 1:
        args = parser.parse_args()
        if args.search_directory is None:
            selected_directory = easygui.diropenbox()
        else:
            selected_directory = args.search_directory
        if len(selected_directory) == 0:  # Gracefully exit if cancel is clicked
            exit()
        if not args.num_trials or not args.num_items:
            logging.warning('Either num_items or num_trials was not provided. The data shape will be automatically ' +
                            'detected from the actual coordinates.')
            d_shape = None
        else:
            d_shape = (args.num_trials, args.num_items, args.dimension)
        manual_swap_accuracy_threshold_list = None
        if args.manual_swap_accuracy_threshold_list is not None:
            # noinspection PyBroadException
            try:
                with open(args.manual_swap_accuracy_threshold_list) as f:
                    lis = [line.split(',') for line in f]
                    for idx, (_, d_threshold) in enumerate(lis):
                        lis[idx][1] = float(d_threshold)
                    manual_swap_accuracy_threshold_list = lis
            except:
                logging.warning(
                    'the provided manual_swap_accuracy_threshold_list was either not found or invalid - it will be '
                    'skipped')
        batch_pipeline(selected_directory,
                       datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv"),
                       data_shape=d_shape,
                       accuracy_z_value=args.accuracy_z_value,
                       trial_by_trial_accuracy=args.trial_by_trial_accuracy != 0,
                       flags=PipelineFlags(args.pipeline_mode),
                       collapse_trials=args.collapse_trials != 0,
                       dimension=args.dimension,
                       actual_coordinate_prefixes=args.actual_coordinate_prefixes,
                       manual_threshold=manual_swap_accuracy_threshold_list,
                       category_independence_enabled=args.category_independence_enabled != 0,
                       order_greedy_deanonymization_enabled=args.order_greedy_deanonymization_enabled != 0
                       )
        exit()

    logging.info("No arguments found - quitting.")
