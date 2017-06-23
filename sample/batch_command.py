import logging
import argparse
import sys
import easygui
import datetime

import numpy as np
import scipy.stats as st

from cogrecon.core.batch_pipeline import batch_pipeline
from cogrecon.core.data_structures import PipelineFlags
from cogrecon.core.cogrecon_globals import data_coordinates_file_suffix, actual_coordinates_file_suffix, \
    order_file_suffix, \
    category_file_suffix, default_dimensions, default_z_value, default_pipeline_flags
from cogrecon.core.file_io import is_path_exists_or_creatable_portable

"""
This module is meant to be run exclusively from the command line via:

python batch_command.py <arguments>

where the arguments are as follows:

    --search_directory ; The root directory in which to search for the actual and data coordinate files 
    (default is None which will result in a popup dialog for selecting the directory).
    
    --category_prefixes ; If 0, one category file (typically named category.txt) is expected. Otherwise, a category 
    file for each data file is expected (with a prefix that matches the data file and a suffix matching the configured 
    suffix - typically category.txt) (default is 0).
    
    --order_prefixes ; If 0, one order file (typically named order.txt) is expected. Otherwise, an order file for each 
    data file is expected (with a prefix that matches the data file and a suffix matching the configured suffix - 
    typically order.txt) (default is 0).
    
    --actual_coordinate_prefixes ; If 0, the normal assumption that all participants used the same (typically 
    actual_coordinates.txt) file will be used. If not 0, it is assumed that all files (typically ending with 
    actual_coordinates.txt) have a prefix which is matched in the data file prefix. Thus, there should be a one-to-one 
    correspondence between data and actual coordinate files and their content shapes (default is 0).
    
    --output_filename ; If None, the current datetime and local directory are used. If a valid path string, the output 
    file will be saved as that filepath (including relative and absolute location; default is None).
    
    --num_trials ; The number of trials in each file (will be detected automatically from newlines if left out) 
    (default is None).
    
    --num_items ; The number of items to be analyzed (will be detected automaticall from tabs if left out) 
    (default is None).
    
    --dimension ; The dimensionality of the data. The dimension will be used to help determine num_items automatically 
    if not specified manually (default is 2).
    
    --pipeline_mode ; The mode in which the pipeline should process (default is 3); 
        0 for just accuracy+swaps,
        1 for accuracy+deanonymization+swaps,
        2 for accuracy+global transformations+swaps,
        3 for accuracy+deanonymization+global transformations+swaps.
    
    --collapse_trials ; If 0, one row per trial will be output, otherwise one row per participant will be output 
    (default is 1).
    
    --accuracy_z_value ; The z value to be used for accuracy exclusion (default is typically 1.96, 
    corresponding to 95% confidence).
    
    --trial_by_trial_accuracy ; When not 0, z_value thresholds are used on a trial-by-trial basis for accuracy 
    calculations, when 0, the thresholds are computed then collapsed across participant trials (default is 0).
    
    --manual_swap_accuracy_threshold_list ; If empty string or none, the value is ignored. If a string (path) pointing 
    to a text file containing a new line separated list of id,threshold pairs is provided, any files whose participant 
    id matches the first matching id in the list will have the associated threshold applied instead of being 
    automatically computed (default is empty string).
    
    --category_independence_enabled ; If 0, the program will run without an assumption of categories. If 1, the program 
    will search the search_directory for a category file (typically category.txt with the appropriate shape (same as 
    the data shape with only 1 final dimension), and the category data will be used to break up the analysis such that 
    item categories will be processed independently from one another (default is 0).
    
    --order_greedy_deanonymization_enabled ; If 0, the program will run without using order information in 
    deanonymization (a global minimum will be used which may take longer to compute). If 1, the program will take a 
    greedy approach to deanonymization. First it will search for appropriate order files which should be associated 
    with the data files in a 1-to-1 fashion (or an order.txt file which will be used across all participants). Then the 
    item being placed first will be associated with its minimum-distance true value, then the second will be associated 
    with the minimum distance remaining values, etc until all values are associated. This effectively weights the 
    importance of deanonymization minimization in accordance with the placement order (default is 0).
    
    --remove_dims ; A list of dimensions (starting with 0) to remove from processing (default is None).

Note that this function assumes its input is NOT meant to be visualized and will act accordingly.
"""

if __name__ == "__main__":
    # Parse the inputs to the batch_pipeline function (see help=??? for details on each param)
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process a single set of points from a single trial in iPosition '
                                                 'compared to a set of correct points. This will not generate an '
                                                 'output file, but will instead print the resulting values and show a '
                                                 'visualizer of the results.')
    # File Searching Configuration
    parser.add_argument('--search_directory', type=str, default=None,
                        help='the root directory in which to search for the actual and data coordinate files ({1} and '
                             '###{0}, respectively)'.format(data_coordinates_file_suffix,
                                                            actual_coordinates_file_suffix))
    parser.add_argument('--category_prefixes', type=int, default=0,
                        help='if 0, one category file {0} is expected. otherwise, a category file for each data file '
                             'is expected (with a prefix that matches the data file and a suffix matching {0})'
                             '.'.format(category_file_suffix))
    parser.add_argument('--order_prefixes', type=int, default=0,
                        help='if 0, one order file {0} is expected. otherwise, an order file for each data file is '
                             'expected (with a prefix that matches the data file and a suffix matching {0}.'.format(
                              order_file_suffix))
    parser.add_argument('--actual_coordinate_prefixes', type=int, default=0,
                        help='if 0, the normal assumption that all participants used the same {0} file will be used. '
                             'if not 0, it is assumed that all {0} files have a prefix which is '
                             'matched in the {1} prefix. Thus, there should be a one-to-one correspondence between '
                             '{0} and {1} files and their contents.'.format(actual_coordinates_file_suffix,
                                                                            data_coordinates_file_suffix))
    # File Out Configuration
    parser.add_argument('--output_filename', type=str, default=None,
                        help='if None, the current datetime and local directory are used. If a valid path string, '
                             'the output file will be saved as that filepath (including relative and absolute '
                             'location.')
    # Dimensionality Configuration
    parser.add_argument('--num_trials', type=int, default=None,
                        help='the number of trials in each file')
    parser.add_argument('--num_items', type=int, default=None,
                        help='the number of items to be analyzed')
    parser.add_argument('--dimension', type=int, default=default_dimensions,
                        help='the dimensionality of the data (default is {0})'.format(default_dimensions))
    # Run Configuration
    parser.add_argument('--pipeline_mode', type=int, default=int(default_pipeline_flags),
                        help='the mode in which the pipeline should process; \n\t0 for just accuracy+swaps, \n\t1 for '
                             'accuracy+deanonymization+swaps, \n\t2 for accuracy+global transformations+swaps, '
                             '\n\t3 for accuracy+deanonymization+global transformations+swaps \n(default is '
                             '{0})'.format(int(default_pipeline_flags)))
    parser.add_argument('--collapse_trials', type=int, default=1,
                        help='if 0, one row per trial will be output, otherwise one row per participant will be '
                             'output (default is 1)')
    # Accuracy Configuration
    parser.add_argument('--accuracy_z_value', type=float, default=default_z_value,
                        help='the z value to be used for accuracy exclusion (default is {0}, corresponding to {1}% '
                             'confidence'.format(default_z_value,
                                                 int(np.round((1 - st.norm.sf(default_z_value) * 2) * 100))))
    parser.add_argument('--trial_by_trial_accuracy', type=int, default=0,
                        help='when not 0, z_value thresholds are used on a trial-by-trial basis for accuracy '
                             'calculations, when 0, the thresholds are computed then collapsed across an individual\'s '
                             'trials')
    parser.add_argument('--manual_swap_accuracy_threshold_list', type=str, default='',
                        help='if empty string or none, the value is ignored. if a string (path) pointing to a text '
                             'file containing a new line separated list of id,threshold pairs is provided, '
                             'any files whose participant id matches the first matching id in the list will have the '
                             'associated threshold applied instead of being automatically computed.')
    # Category Configuration
    parser.add_argument('--category_independence_enabled', type=int, default=0,
                        help='if 0, the program will run without an assumption of categories. if 1, the program will '
                             'search the search_directory for a category file ({1}) with the appropriate '
                             'shape (same as {0}), and the category data will be used to break up '
                             'the analysis such that item categories will be processed independently from one '
                             'another.'.format(actual_coordinates_file_suffix, category_file_suffix))

    # Order Configuration
    parser.add_argument('--order_greedy_deanonymization_enabled', type=int, default=0,
                        help='if 0, the program will run without using order information in deanonymization (a global '
                             'minimum will be used). if 1, the program will take a greedy approach to '
                             'deanonymization. First it will search for ###{1} files which should be associated '
                             'with the ###{0} files in a 1-to-1 fashion. Then the item '
                             'being placed first will be associated with its minimum-distance true value, '
                             'then the second will be associated with the minimum distance remaining values, '
                             'etc until all values are associated. This effectively weights the importance of '
                             'deanonymization minimization in accordance with the placement order.'.format(
                              data_coordinates_file_suffix, order_file_suffix))

    # Data Flexing/Reorganization
    parser.add_argument('--remove_dims', type=int, nargs='+', default=None,
                        help='a list of dimensions (starting with 0) to remove from processing (default is None)')

    # If we receive some parameters
    if len(sys.argv) > 1:
        # Attempt to parse the parameters
        args = parser.parse_args()
        # If a search directory is not provided
        if args.search_directory is None:
            # Prompt the user for a search directory
            selected_directory = easygui.diropenbox()
        else:
            # If a search directory is provided, store it
            selected_directory = args.search_directory
        # Gracefully exit if cancel is clicked
        if len(selected_directory) == 0:
            exit()
        # If either number of trials or number of items is not provided, it is determined automatically (via dimension)
        if not args.num_trials or not args.num_items:
            logging.warning('Either num_items or num_trials was not provided. The data shape will be automatically ' +
                            'detected from the actual coordinates.')
            d_shape = None
        else:
            # If all 3 shape params are provided, store them
            d_shape = (args.num_trials, args.num_items, args.dimension)
        manual_swap_accuracy_threshold_list = None
        # If a manual threshold list is provided
        if args.manual_swap_accuracy_threshold_list is not None:
            # Read the manual threshold list
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
        # If an output path is provided and is valid
        if args.output_filename is not None and is_path_exists_or_creatable_portable(args.output_filename):
            # Save it
            outfilepath = args.output_filename.strip()
        else:
            # If no output path is provided, use the current datetime
            outfilepath = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S.csv")

        # Run the batch pipeline with the provided parameters
        batch_pipeline(selected_directory,
                       outfilepath,
                       data_shape=d_shape,
                       accuracy_z_value=args.accuracy_z_value,
                       trial_by_trial_accuracy=args.trial_by_trial_accuracy != 0,
                       flags=PipelineFlags(args.pipeline_mode),
                       collapse_trials=args.collapse_trials != 0,
                       dimension=args.dimension,
                       actual_coordinate_prefixes=args.actual_coordinate_prefixes,
                       manual_threshold=manual_swap_accuracy_threshold_list,
                       category_independence_enabled=args.category_independence_enabled != 0,
                       category_prefixes=args.category_prefixes != 0,
                       order_greedy_deanonymization_enabled=args.order_greedy_deanonymization_enabled != 0,
                       order_prefixes=args.order_prefixes != 0,
                       removal_dim_indicies=args.remove_dims
                       )
    else:
        # Quit gracefully if no parameters were provided
        logging.info("No arguments found - quitting.")
