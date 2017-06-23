import easygui
import os

if __name__ == '__main__':
    # noinspection PyUnresolvedReferences
    from cogrecon.core.file_io import match_file_prefixes, find_data_files_in_directory, \
        extract_prefixes_from_file_list_via_suffix, get_coordinates_from_file
    from cogrecon.core.cogrecon_globals import data_coordinates_file_suffix, category_file_suffix, \
        actual_coordinates_file_suffix
else:
    # noinspection PyUnresolvedReferences
    from ..file_io import match_file_prefixes, find_data_files_in_directory, \
        extract_prefixes_from_file_list_via_suffix, get_coordinates_from_file
    from ..cogrecon_globals import data_coordinates_file_suffix, category_file_suffix, \
        actual_coordinates_file_suffix


def process_category_files(selected_directory=None, output_path='..\\..\\..\\saved_data\\category_reprocessed\\'):
    """
    This function performs a very specific task as requested by a researcher. It first prompts for the selection
    of a particular directory. It searches that directory and sub directories for files with a particular suffix assumed
    to be in the custom category format. It also finds the associated data coordinates file and splits the files into
    ###_category_position_data_coordinates.txt, ###_nocategory_position_data_coordinates.txt,
    ###_category_categories.txt, ###_nocategory_categories.txt. It also takes the root actual_coordinates.txt file and
    generates ###_nocategory_actual_corodinates.txt, ###_category_actual_coordinates.txt files for each participant.

    The result is written to a specified output path (created if it does not already exist).

    :param selected_directory: The string path to a directory to scan for files ending in study_iposition_data.txt.
                               If left empty, a popup dialog will be presented to select a directory.
    :param output_path: The directory into which the output files should be saved.
    """
    if selected_directory is None:
        selected_directory = easygui.diropenbox()

    actual_coordinates_files, data_files, category_files, order_files = \
        find_data_files_in_directory(selected_directory, _category_file_suffix='study_iposition_data.txt')

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    actual_coordinates_data = get_coordinates_from_file(actual_coordinates_files[0], (40, 6, 2))

    for df, cf in zip(data_files, category_files):
        prefix = extract_prefixes_from_file_list_via_suffix([df], suffix=data_coordinates_file_suffix)[0]

        print('Parsing {0}'.format(prefix))

        category_data = []
        categorization_data = []

        # Get the categorization split information
        with open(cf, 'rU') as fp:
            contents = fp.readlines()
            for line in contents:
                split_line = line.split('\t')
                categorization_data.append(int(split_line[1]))
                category_data.append([int(split_line[i]) for i in range(2, 8)])

        data_coordinates_data = get_coordinates_from_file(df, (40, 6, 2))

        # Write data files
        with open(os.path.join(output_path, prefix+'category_'+data_coordinates_file_suffix), 'w') as fp:
            for trial, cat in zip(data_coordinates_data, categorization_data):
                if cat == 2:
                    fp.write('\t'.join([str(item) for sublist in trial for item in sublist]) + '\n')

        with open(os.path.join(output_path, prefix + 'nocategory_' + data_coordinates_file_suffix), 'w') as fp:
            for trial, cat in zip(data_coordinates_data, categorization_data):
                if cat == 1:
                    fp.write('\t'.join([str(item) for sublist in trial for item in sublist]) + '\n')

        # Write category files
        with open(os.path.join(output_path, prefix + 'category_' + category_file_suffix), 'w') as fp:
            for trial, cat in zip(category_data, categorization_data):
                if cat == 2:
                    fp.write('\t'.join([str(item) for item in trial]) + '\n')

        with open(os.path.join(output_path, prefix + 'nocategory_' + category_file_suffix), 'w') as fp:
            for trial, cat in zip(category_data, categorization_data):
                if cat == 1:
                    fp.write('\t'.join([str(item) for item in trial]) + '\n')

        # Write actual coordinate files
        with open(os.path.join(output_path,
                               prefix + 'category_' + actual_coordinates_file_suffix), 'w') as fp:
            for trial, cat in zip(actual_coordinates_data, categorization_data):
                if cat == 2:
                    fp.write('\t'.join([str(item) for sublist in trial for item in sublist]) + '\n')

        with open(os.path.join(output_path,
                               prefix + 'nocategory_' + actual_coordinates_file_suffix), 'w') as fp:
            for trial, cat in zip(actual_coordinates_data, categorization_data):
                if cat == 1:
                    fp.write('\t'.join([str(item) for sublist in trial for item in sublist]) + '\n')

    print("Done!")


if __name__ == '__main__':
    process_category_files(selected_directory=r'Z:\Kevin\iPosition\Hillary\Category_Squig_iPos')
