import cogrecon.core.batch_pipeline as bpipe
import easygui
import os


def descrambler():
    selected_directory = easygui.diropenbox()

    _, files = bpipe.find_data_files_in_directory(selected_directory)

    for f in files:
        print("descrambling file {0}".format(f))
        lines = []
        with open(f, 'rb') as fp:
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
