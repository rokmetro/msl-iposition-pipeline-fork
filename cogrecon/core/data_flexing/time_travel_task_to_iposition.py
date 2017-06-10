import os
import numpy as np

if __name__ == "__main__":
    from time_travel_task_binary_reader import find_data_files_in_directory, get_item_details, read_binary_file, \
        parse_test_items, get_filename_meta_data, get_items_solutions
    from cogrecon.core.cogrecon_globals import data_coordinates_file_suffix, actual_coordinates_file_suffix, \
        order_file_suffix, category_file_suffix
else:
    from .time_travel_task_binary_reader import find_data_files_in_directory, get_item_details, read_binary_file, \
        parse_test_items, get_filename_meta_data, get_items_solutions
    from ..cogrecon_globals import data_coordinates_file_suffix, actual_coordinates_file_suffix, \
        order_file_suffix, category_file_suffix


def save_iposition_items_to_file(_filename, _items):
    """
    This function saves a set of items in iposition (TSV) format given a filename and item dictionary list

    :param _filename: a string filename in which to save the data
    :param _items: a dictionary containing position information stored in a "pos" key to be saved
    """
    with open(_filename, 'ab') as fp:
        poses = []
        for item in _items:
            if item is not None and item['pos'] is not None:
                poses += list(item['pos'])
            else:
                poses += [np.nan, np.nan, np.nan]
        line = '\t'.join([str(x) for x in poses])
        fp.write(line + '\r\n')
        fp.flush()


# noinspection PyTypeChecker
def extract_basic_order(order_list, first=True):
    """
    This function extracts simplified order information from an order list, assuming that we want to know the first
    item placed as 0 and last as 'n'.

    :param order_list: an order list of integers which contains gaps
    :param first: a boolean determining if we want to order first to last or last to first
    :return: an integer list 0 to 'n' ordered according to the input from smallest to largest integer
    """
    if order_list is None or len(order_list) == 0 or order_list[0] == []:
        return []
    indicies = list(range(len(order_list)))
    if first:
        try:
            indicies.sort(key=[x[0] for x in order_list].__getitem__)
        except IndexError:
            print(order_list)
            print(first)
    else:
        indicies.sort(key=[x[-1] for x in order_list].__getitem__)
    result = [None for _ in range(0, len(indicies))]
    for idx in range(0, len(indicies)):
        result[indicies[idx]] = idx

    return result


def save_tsv(_filename, _list):
    """
    This helper function saves simple TSV files via a _list (assumed to be 2D).

    :param _filename: the string filename in which to save the TSV
    :param _list: a 2D list to be saved where each row (first dimension) is saved per line and each element is
    separated by a tab
    """
    with open(_filename, 'ab') as fp:
        line = '\t'.join([str(el) for el in _list])
        fp.write(line + '\r\n')
        fp.flush()


def time_travel_task_to_iposition(input_dir, output_dir,
                                  file_regex="\d\d\d_\d_2_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat",
                                  order_first=True, exclude_incorrect_category=False, output_user_categories=True):

    """
    This function is a global helper function meant to take a directory in which many Time Travel Task binary files
    exist and convert them all into iposition (TSV) compatible files to be saved in an output directory.

    :param output_user_categories: a boolean which, if true, will cause the conversion to output *categories.txt files
    populated with the user's categorization of the items. If false, the correct categorization given the user's
    meta information (i.e. inversion) will be used.
    :param input_dir: the string directory to search for binary files via a regular expression stored in file_regex
    :param output_dir: the string output directory in which to save files
    :param file_regex: the string regular expression to use to determine if a file should be included as input
    :param order_first: boolean value which, if true, order is used first-to-last - if false, order is used
    last-to-first
    :param exclude_incorrect_category: a boolean value which, if true, removes items which are the wrong category from
    the output data
    """
    files = find_data_files_in_directory(input_dir, file_regex=file_regex)

    event_state_labels, item_number_label, item_label_filename, cols = get_item_details()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    actual_coords_path = '{0}' + actual_coordinates_file_suffix
    out_path = '{0}' + data_coordinates_file_suffix
    order_path = '{0}' + order_file_suffix
    category_path = '{0}' + category_file_suffix

    for _path in files:
        iterations = read_binary_file(_path)
        reconstruction_items, order = parse_test_items(iterations, cols, item_number_label, event_state_labels)
        meta = get_filename_meta_data(os.path.basename(_path))
        items, times, directions = get_items_solutions(meta)

        print(meta['subID'] + ',' + meta['trial'])

        if exclude_incorrect_category:
            excluded_list = np.transpose([(r, i, o) for r, i, o in zip(reconstruction_items, items, order)
                                          if ('direction' in r and 'direction' in i
                                              and r['direction'] == i['direction'])]).tolist()

            if not excluded_list:
                reconstruction_items = []
                items = []
                order = []
            else:
                reconstruction_items, items, order = excluded_list

        if output_user_categories:
            categories = [r['direction'] for r in reconstruction_items
                          if (r is not None and r['direction'] is not None)]
        else:
            categories = [i['direction'] for i in items
                          if (i is not None and i['direction'] is not None)]

        order = extract_basic_order(order, first=order_first)

        save_iposition_items_to_file(output_dir+'\\'+out_path.format(meta['subID']), reconstruction_items)
        save_iposition_items_to_file(output_dir+'\\'+actual_coords_path.format(meta['subID']), items)
        # noinspection PyTypeChecker
        if len(np.array(order).flatten().tolist()) != 0:
            save_tsv(output_dir + '\\' + order_path.format(meta['subID']), order)

        save_tsv(output_dir + '\\' + category_path.format(meta['subID']), categories)

# TODO: Generalize this module to be run outside of the __main__ environment
if __name__ == "__main__":
    in_directory = 'C:\\Users\\Kevin\\Desktop\\Work\\Time Travel Task\\v2'
    path = os.path.dirname(os.path.realpath(__file__))
    # time_travel_task_to_iposition(in_directory,
    #                               path+'..\\..\\..\\..\\saved_data\\iPositionConversion',
    #                               exclude_incorrect_category=False)
    # time_travel_task_to_iposition(in_directory,
    #                               path + '..\\..\\..\\..\\saved_data\\iPositionConversion_ExcludeIncorrectCategory',
    #                               exclude_incorrect_category=True)
    time_travel_task_to_iposition(in_directory,
                                  path + '..\\..\\..\\..\\saved_data\\iPositionConversion_GlobalCategories',
                                  exclude_incorrect_category=False, output_user_categories=False)
