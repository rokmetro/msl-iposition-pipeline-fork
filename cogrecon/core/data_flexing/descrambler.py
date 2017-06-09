import easygui
import os

from ..batch_pipeline import find_data_files_in_directory


def descrambler():
    """
    This descrambler performs a very specific task as requested by a researcher. It first prompts for the selection
    of a particular directory. It searches that directory for files (any file) and parses that file as a TSV.
    The TSV is expected to have a final element on each line which represents the order the line is actually
    meant to be in. The file is, thus, reordered according to this last element (assumed to be an integer).

    The result is written to an output path which is the input path with 'descrambled_' preprended to the filename.
    """
    selected_directory = easygui.diropenbox()

    _, files = find_data_files_in_directory(selected_directory)

    for f in files:
        print("descrambling file {0}".format(f))
        lines = []
        with open(f, 'rU') as fp:
            for line in fp:
                lines.append(line)
        output = [''] * len(lines)
        for line in lines:
            line_split = line.split('\t')
            output[int(line_split[0].strip())-1] = '\t'.join(line_split[1:])
        outpath = os.path.join(os.path.dirname(f), "descrambled_" + os.path.basename(f))
        print("writing to {0}".format(outpath))
        with open(outpath, 'wb') as fp:
            fp.writelines(output)

    print("Done!")
