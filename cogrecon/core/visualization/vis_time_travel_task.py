import logging
import os
import easygui
import math
import sys

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from pyqtgraph.Qt import QtCore, QtGui
from scipy.misc import imread

import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == '__main__':
    from cogrecon.core.data_flexing.time_travel_task.time_travel_task_binary_reader import parse_test_items, \
        get_item_details, get_click_locations_and_indicies, get_items_solutions, phase_num_to_str, \
        read_binary_file, get_filename_meta_data, find_data_files_in_directory
else:
    from ..data_flexing.time_travel_task.time_travel_task_binary_reader import parse_test_items, get_item_details, \
        get_click_locations_and_indicies, get_items_solutions, phase_num_to_str, read_binary_file, \
        get_filename_meta_data, find_data_files_in_directory


# TODO: Fix issue with pyqt globals (should not used globals)
def visualize_time_travel_data(path=None, automatically_rotate=True):
    """
    This function visualizes data from the Time Travel Task in 3D.

    :param path: the path to a data file to visualize
    :param automatically_rotate: If True, the figure will automatically rotate in an Abs(Sin(x)) function shape,
                                 otherwise the user can interact with the figure.

    :return: Nothing
    """
    # noinspection PyGlobalUndefined
    global path_line, idx, timer, iterations, click_scatter, click_pos, click_color, click_size, window, meta, \
        reconstruction_items, num_points_to_update, line_color, line_color_state, auto_rotate

    auto_rotate = automatically_rotate

    ####################################################################################################################
    # Setup
    ####################################################################################################################

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.WARNING)

    # Get Log File Path and Load File
    local_directory = os.path.dirname(os.path.realpath(__file__))  # The directory of this script
    # filename = '001_1_1_1_2016-08-29_10-26-03.dat'  # The relative path to the data file (CHANGE ME)
    # path = os.path.join(local_directory, filename)
    if path is None:
        path = easygui.fileopenbox()
    if path is '':
        logging.info('No file selected. Closing.')
        exit()
    if not os.path.exists(path):
        logging.error('File not found. Closing.')
        exit()

    meta = None
    # noinspection PyBroadException
    try:
        meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
    except:
        logging.error('There was an error reading the filename meta-information. Please confirm this is a valid log '
                      'file.')
        exit()

    logging.info("Parsing file (" + str(path) + ")...")
    # First we populate a list of each iteration's data
    # This section of code contains some custom binary parser data which won't be explained here
    iterations = read_binary_file(path)
    # Output the iterations count for debugging purposes
    logging.info("Plotting " + str(len(iterations)) + " iterations.")

    # Generate UI Window and Set Camera Settings

    app = QtGui.QApplication([])
    window = gl.GLViewWidget()
    window.opts['center'] = pg.Qt.QtGui.QVector3D(0, 0, 30)
    window.opts['distance'] = 200
    window.setWindowTitle('Timeline Visualizer' +
                          ' - Subject {0}, Trial {1}, Phase {2}'.format(meta['subID'], meta['trial'],
                                                                        phase_num_to_str(int(meta['phase']))))

    ####################################################################################################################
    # Generate static graphical items
    ####################################################################################################################

    # Make Grid

    grid_items = []

    def make_grid_item(loc, rot, scale):
        g = gl.GLGridItem()
        g.scale(scale[0], scale[1], scale[2])
        g.rotate(rot[0], rot[1], rot[2], rot[3])
        g.translate(loc[0], loc[1], loc[2])
        return g

    if meta['phase'] == '0' or meta['phase'] == '3' or meta['phase'] == '6':
        g0 = make_grid_item((-19, 0, 15), (90, 0, 1, 0), (1.5, 1.9, 1.9))
        g1 = make_grid_item((0, -19, 15), (90, 1, 0, 0), (1.9, 1.5, 1.9))
        grid_items.append(g0)
        grid_items.append(g1)
        window.addItem(g0)
        window.addItem(g1)
    else:
        g0 = make_grid_item((-19, 0, 15), (90, 0, 1, 0), (1.5, 1.9, 1.9))
        g1 = make_grid_item((-19, 0, 45), (90, 0, 1, 0), (1.5, 1.9, 1.9))
        g2 = make_grid_item((0, -19, 15), (90, 1, 0, 0), (1.9, 1.5, 1.9))
        g3 = make_grid_item((0, -19, 45), (90, 1, 0, 0), (1.9, 1.5, 1.9))
        grid_items.append(g0)
        grid_items.append(g1)
        grid_items.append(g2)
        grid_items.append(g3)
        window.addItem(g0)
        window.addItem(g1)
        window.addItem(g2)
        window.addItem(g3)
    gn = make_grid_item((0, 0, 0), (0, 0, 0, 0), (1.9, 1.9, 1.9))
    grid_items.append(gn)
    window.addItem(gn)

    # Make Image Base

    # Determine the background image according to meta phase
    img_location = './media/time_travel_task/'
    bg_path = 'studyBG.png'
    if meta['phase'] == '0' or meta['phase'] == '3':
        bg_path = 'practiceBG.png'
    elif meta['phase'] == '6':
        bg_path = 'practiceBG.png'
    elif meta['phase'] == '7' or meta['phase'] == '8':
        bg_path = 'studyBG.png'
    img = imread(os.path.abspath(os.path.join(img_location, bg_path)))

    image_scale = (19.0 * 2.0) / float(img.shape[0])
    tex1 = pg.makeRGBA(img)[0]
    base_image = gl.GLImageItem(tex1)
    base_image.translate(-19, -19, 0)
    base_image.rotate(270, 0, 0, 1)
    base_image.scale(image_scale, image_scale, image_scale)
    window.addItem(base_image)

    # Make Timeline Colored Bars

    color_bars = []

    def make_color_bar(rgb, p, r, s):
        v = gl.GLImageItem(np.array([[rgb + (255,)]]))
        v.translate(p[0], p[1], p[2])
        v.scale(s[0], s[1], s[2])
        v.rotate(r[0], r[1], r[2], r[3])
        return v

    color_bar_length = 15

    if meta['phase'] == '0' or meta['phase'] == '3' or meta['phase'] == '6':
        times = [0, 7.5, 15, 22.5]
        color_bar_length = 7.5
    else:
        times = [0, 15, 30, 45]
    if meta['inverse'] == '1':
        times.reverse()

    v0 = make_color_bar((255, 255, 0), (19, times[0], 19), (90, 1, 0, 0), (5, color_bar_length, 0))
    v1 = make_color_bar((255, 0, 0), (19, times[1], 19), (90, 1, 0, 0), (5, color_bar_length, 0))
    v2 = make_color_bar((0, 255, 0), (19, times[2], 19), (90, 1, 0, 0), (5, color_bar_length, 0))
    v3 = make_color_bar((0, 0, 255), (19, times[3], 19), (90, 1, 0, 0), (5, color_bar_length, 0))
    color_bars.append(v0)
    color_bars.append(v1)
    color_bars.append(v2)
    color_bars.append(v3)
    window.addItem(v0)
    window.addItem(v1)
    window.addItem(v2)
    window.addItem(v3)

    # Generate Path Line

    forwardColor = (255, 255, 255, 255)
    backwardColor = (255, 0, 255, 255)
    line_color = np.empty((len(iterations), 4))
    line_color_state = np.empty((len(iterations), 4))
    x = []
    y = []
    z = []
    for idx, i in enumerate(iterations):
        x.append(float(i['x']))
        y.append(float(i['z']))
        z.append(float(i['time_val']))
        c = forwardColor
        if i['timescale'] <= 0:
            c = backwardColor
        line_color[idx] = pg.glColor(c)
        line_color_state[idx] = pg.glColor((0, 0, 0, 0))

    pts = np.vstack([x, y, z]).transpose()
    path_line = gl.GLLinePlotItem(pos=pts, color=line_color_state, mode='line_strip', antialias=True)
    window.addItem(path_line)

    # Generate Item Lines (ground truth)
    # noinspection PyUnusedLocal
    items, times, directions = get_items_solutions(meta)

    if meta['phase'] == '0' or meta['phase'] == '3' or meta['phase'] == '6':
        times = [2, 12, 18, 25]
        directions = [2, 1, 2, 1]  # Fall = 2, Fly = 1, Stay = 0
        if meta['inverse'] == '1':
            times.reverse()
            directions.reverse()
        items = [{'direction': directions[0], 'pos': (2, -12, times[0]), 'color': (255, 255, 0)},
                 {'direction': directions[1], 'pos': (2, 13, times[1]), 'color': (255, 0, 0)},
                 {'direction': directions[2], 'pos': (-13, 2, times[2]), 'color': (0, 255, 0)},
                 {'direction': directions[3], 'pos': (-12, -17, times[3]), 'color': (0, 0, 255)},
                 {'direction': 0, 'pos': (13, 5, 0), 'color': (128, 0, 128)}]
    # elif meta['phase'] == '7' or meta['phase'] == '8':
    #    times = [2, 8, 17, 23]
    #    directions = [2, 1, 1, 2]  # Fall = 2, Fly = 1, Stay = 0
    #    if meta['inverse'] == '1':
    #        times.reverse()
    #        directions.reverse()
    #    items = [{'direction': directions[0], 'pos': (16, -14, times[0]), 'color': (255, 255, 0)},
    #             {'direction': directions[1], 'pos': (-10, -2, times[1]), 'color': (255, 0, 0)},
    #             {'direction': directions[2], 'pos': (15, -8, times[2]), 'color': (0, 255, 0)},
    #             {'direction': directions[3], 'pos': (-15, -15, times[3]), 'color': (0, 0, 255)},
    #             {'direction': 0, 'pos': (-2, 10, 0), 'color': (128, 0, 128)}]
    else:
        times = [4, 10, 16, 25, 34, 40, 46, 51]
        directions = [2, 1, 1, 2, 2, 1, 2, 1]  # Fall = 2, Fly = 1, Stay = 0
        if meta['inverse'] == '1':
            times.reverse()
            directions.reverse()
        items = [{'direction': directions[0], 'pos': (18, -13, times[0]), 'color': (255, 255, 0)},
                 {'direction': directions[1], 'pos': (-13, 9, times[1]), 'color': (255, 255, 0)},
                 {'direction': directions[2], 'pos': (-10, -2, times[2]), 'color': (255, 0, 0)},
                 {'direction': directions[3], 'pos': (6, -2, times[3]), 'color': (255, 0, 0)},
                 {'direction': directions[4], 'pos': (17, -8, times[4]), 'color': (0, 255, 0)},
                 {'direction': directions[5], 'pos': (-2, -7, times[5]), 'color': (0, 255, 0)},
                 {'direction': directions[6], 'pos': (-15, -15, times[6]), 'color': (0, 0, 255)},
                 {'direction': directions[7], 'pos': (6, 18, times[7]), 'color': (0, 0, 255)},
                 {'direction': 0, 'pos': (14, 6, 0), 'color': (128, 0, 128)},
                 {'direction': 0, 'pos': (-2, 10, 0), 'color': (128, 0, 128)}]

    item_lines = []
    pos = np.empty((len(items), 3))
    size = np.empty((len(items)))
    color = np.empty((len(items), 4))
    end_time = 60
    if meta['phase'] == '0' or meta['phase'] == '3' or meta['phase'] == '6':
        end_time = 30
    for idx, i in enumerate(items):
        pos[idx] = i['pos']
        size[idx] = 2
        if i['direction'] == 0:
            size[idx] = 0
        color[idx] = (i['color'][0] / 255, i['color'][1] / 255, i['color'][2] / 255, 1)
        idx += 1
        end = i['pos']
        if i['direction'] == 1:
            end = (end[0], end[1], 0)
        elif i['direction'] == 2 or i['direction'] == 0:
            end = (end[0], end[1], end_time)
        line = gl.GLLinePlotItem(pos=np.vstack([[i['pos'][0], end[0]],
                                                [i['pos'][1], end[1]],
                                                [i['pos'][2], end[2]]]).transpose(),
                                 color=pg.glColor(i['color']), width=3, antialias=True)
        item_lines.append(line)
        window.addItem(line)

    item_scatter_plot = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
    window.addItem(item_scatter_plot)

    ####################################################################################################################
    # Generate data graphical items
    ####################################################################################################################

    # If Study/Practice, label click events
    '''click_pos = np.empty((len(items), 3))
    click_size = np.zeros((len(iterations), len(items)))
    click_color = np.empty((len(items), 4))
    if meta['phase'] == '0' or meta['phase'] == '1' or meta['phase'] == '3' or meta['phase'] == '4' \
            or meta['phase'] == '6' or meta['phase'] == '7':
        for idx, i in enumerate(iterations):
            if idx + 1 < len(iterations):
                for idxx, (i1, i2) in enumerate(zip(i['itemsclicked'], iterations[idx + 1]['itemsclicked'])):
                    if i['itemsclicked'][idxx]:
                        click_size[idx][idxx] = 0.5
                    if not i1 == i2:
                        click_pos[idxx] = (i['x'], i['z'], i['time'])
                        click_color[idxx] = (128, 128, 128, 255)
            else:
                for idxx, i1 in enumerate(i['itemsclicked']):
                    if i['itemsclicked'][idxx]:
                        click_size[idx][idxx] = 0.5
    '''
    click_pos, _, click_size, click_color = get_click_locations_and_indicies(iterations, items, meta)
    click_scatter = gl.GLScatterPlotItem(pos=click_pos, size=click_size[0], color=click_color, pxMode=False)
    window.addItem(click_scatter)

    # If Test, Generate Reconstruction Items

    event_state_labels, item_number_label, item_label_filename, cols = get_item_details()

    # if meta['phase'] == '7' or meta['phase'] == '8':
    #    item_number_label = ['bottle', 'clover', 'boot', 'bandana', 'guitar']
    #    item_label_filename = ['bottle.jpg', 'clover.jpg', 'boot.jpg', 'bandana.jpg', 'guitar.jpg']
    #    cols = [(255, 255, pastel_factor), (255, pastel_factor, pastel_factor), (pastel_factor, 255, pastel_factor),
    #            (pastel_factor, pastel_factor, 255), (128, pastel_factor / 2, 128)]
    reconstruction_item_scatter_plot = None
    reconstruction_item_lines = []
    if meta['phase'] == '2' or meta['phase'] == '5' or meta['phase'] == '8':
        billboard_item_labels, reconstruction_items = parse_test_items(iterations, cols,
                                                                       item_number_label, event_state_labels)
        pos = np.empty((len(reconstruction_items), 3))
        size = np.empty((len(reconstruction_items)))
        color = np.empty((len(reconstruction_items), 4))
        # Iterate through the reconstruction items and visualize them
        for idx, i in enumerate(reconstruction_items):
            pos[idx] = i['pos']
            size[idx] = 2
            if i['direction'] == 0:
                size[idx] = 0
            color[idx] = (i['color'][0] / 255, i['color'][1] / 255, i['color'][2] / 255, 1)
            end = pos[idx]
            if i['direction'] == 1:
                end = (end[0], end[1], 0)
            elif i['direction'] == 2 or i['direction'] == 0:
                end = (end[0], end[1], end_time)
            line = gl.GLLinePlotItem(pos=np.vstack([[pos[idx][0], end[0]],
                                                    [pos[idx][1], end[1]],
                                                    [pos[idx][2], end[2]]]).transpose(),
                                     color=pg.glColor(i['color']), width=3, antialias=True)
            reconstruction_item_lines.append(line)
            window.addItem(line)

            img_path = item_label_filename[idx]
            img = imread(os.path.join(local_directory, img_path))
            expected_size = 2.0
            image_scale = expected_size / float(img.shape[0])
            offset_param = 0.0 - image_scale / 2 - expected_size / 2
            tex = pg.makeRGBA(img)[0]
            label_image = gl.GLImageItem(tex)
            t = pos[idx][2]
            if i['direction'] == 0:
                t = end_time
            label_image.translate(pos[idx][0] + offset_param, pos[idx][1] + offset_param, t)
            label_image.scale(image_scale, image_scale, image_scale)
            window.addItem(label_image)
            billboard_item_labels.append(label_image)
        reconstruction_item_scatter_plot = gl.GLScatterPlotItem(pos=pos, size=size, color=color, pxMode=False)
        window.addItem(reconstruction_item_scatter_plot)

    ####################################################################################################################
    # Show UI
    ####################################################################################################################

    window.show()
    logging.info("Showing plot. Close plot to exit program.")

    ####################################################################################################################
    # Custom Keyboard Controls
    ####################################################################################################################

    # These variables are modified by the keyboard controls
    idx = 0
    num_points_to_update = 5
    saved_points_to_update = 0
    paused = False

    # GUI Callbacks
    def speed_up():
        global num_points_to_update, paused
        if not paused:
            num_points_to_update += 5
            logging.info("Setting speed to " + str(num_points_to_update) + " points per tick.")

    def speed_down():
        global num_points_to_update, paused
        if not paused:
            num_points_to_update -= 5
            logging.info("Setting speed to " + str(num_points_to_update) + " points per tick.")

    def pause():
        global num_points_to_update, saved_points_to_update, paused
        if not paused:
            logging.info("Paused.")
            saved_points_to_update = num_points_to_update
            num_points_to_update = 0
            paused = True
        else:
            logging.info("Unpaused.")
            num_points_to_update = saved_points_to_update
            saved_points_to_update = -0.5
            paused = False

    def reset():
        global idx, line_color_state
        logging.info("Resetting to time zero.")
        idx = 0
        for index in range(0, len(line_color_state) - 1):
            line_color_state[index] = (0, 0, 0, 0)

    def go_to_end():
        global idx, line_color_state, line_color
        logging.info("Going to end.")
        idx = len(line_color_state) - 1
        for index in range(0, len(line_color_state) - 1):
            line_color_state[index] = line_color[index]

    def close_all():
        global timer, app
        logging.info("User Shutdown Via Button Press")
        timer.stop()
        app.closeAllWindows()

    # Visibility Variables
    grid_visible = True
    base_visible = True
    color_bars_visible = True
    items_visible = True
    path_line_visible = True
    reconstruction_item_lines_visible = True
    billboard_item_labels_visible = True

    def toggle_grid_visible():
        global grid_visible
        if grid_visible:
            for g in grid_items:
                g.hide()
            grid_visible = False
        else:
            for g in grid_items:
                g.show()
            grid_visible = True

    def toggle_base_visible():
        global base_visible
        if base_visible:
            base_image.hide()
            base_visible = False
        else:
            base_image.show()
            base_visible = True

    def toggle_color_bars_visible():
        global color_bars_visible
        if color_bars_visible:
            for bar in color_bars:
                bar.hide()
            color_bars_visible = False
        else:
            for bar in color_bars:
                bar.show()
            color_bars_visible = True

    def toggle_items_visible():
        global items_visible
        if items_visible:
            item_scatter_plot.hide()
            for il in item_lines:
                il.hide()
            items_visible = False
        else:
            item_scatter_plot.show()
            for il in item_lines:
                il.show()
            items_visible = True

    def toggle_path_line_visible():
        global path_line_visible
        if path_line_visible:
            path_line.hide()
            click_scatter.hide()
            path_line_visible = False
        else:
            path_line.show()
            click_scatter.show()
            path_line_visible = True

    def toggle_reconstruction_item_lines_visible():
        global reconstruction_item_lines_visible
        if reconstruction_item_lines_visible:
            if reconstruction_item_scatter_plot is not None:
                reconstruction_item_scatter_plot.hide()
            for ril in reconstruction_item_lines:
                ril.hide()
            reconstruction_item_lines_visible = False
        else:
            if reconstruction_item_scatter_plot is not None:
                reconstruction_item_scatter_plot.show()
            for ril in reconstruction_item_lines:
                ril.show()
            reconstruction_item_lines_visible = True

    def toggle_billboard_item_labels_visible():
        global billboard_item_labels_visible
        if billboard_item_labels_visible:
            for il in billboard_item_labels:
                il.hide()
                billboard_item_labels_visible = False
        else:
            for il in billboard_item_labels:
                il.show()
                billboard_item_labels_visible = True

    # GUI Initialization
    sh = QtGui.QShortcut(QtGui.QKeySequence("+"), window, speed_up)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("-"), window, speed_down)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence(" "), window, pause)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("R"), window, reset)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("E"), window, go_to_end)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("Escape"), window, close_all)
    sh.setContext(QtCore.Qt.ApplicationShortcut)

    sh = QtGui.QShortcut(QtGui.QKeySequence("1"), window, toggle_grid_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("2"), window, toggle_base_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("3"), window, toggle_color_bars_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("4"), window, toggle_items_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("5"), window, toggle_path_line_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("6"), window, toggle_reconstruction_item_lines_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("7"), window, toggle_billboard_item_labels_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)

    ####################################################################################################################
    # Animation Loop
    ####################################################################################################################

    timer = QtCore.QTimer()
    # noinspection PyUnresolvedReferences
    timer.timeout.connect(update)
    timer.start(1)

    ####################################################################################################################
    # PyQtGraph Initialization
    ####################################################################################################################

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # noinspection PyArgumentList
        QtGui.QApplication.instance().exec_()


def update():
    """
    This function is the animation update function used internally in visualize_time_travel_data.

    """
    global path_line, idx, timer, iterations, click_scatter, click_pos, click_color, click_size, window, meta, \
        reconstruction_items, num_points_to_update, line_color, line_color_state, auto_rotate
    if auto_rotate:
        window.opts['elevation'] = math.fabs(math.cos(float(idx) / 800.) * 20.) + 10.
        window.opts['azimuth'] = math.sin(float(idx) / 800.) * 45 + 45
    for _ in range(0, abs(num_points_to_update)):
        if num_points_to_update > 0:
            line_color_state[idx] = line_color[idx]
            idx += 1
        else:
            line_color_state[idx] = (0, 0, 0, 0)
            idx -= 1
        if idx < 0:
            idx = 0
        elif idx >= len(line_color):
            idx = len(line_color) - 1
            break
    path_line.setData(color=line_color_state)
    if meta['phase'] == '2' or meta['phase'] == '5' or meta['phase'] == '8':
        xs = []
        zs = []
        ts = []
        for item in reconstruction_items:
            xs.append(item['pos'][0])
            zs.append(item['pos'][1])
            ts.append(item['pos'][2])
        # position = np.array([(xpos, zpos, tpos) for (xpos, zpos, tpos) in zip(iterations[idx]['itemsx'],
        #                                                                       iterations[idx]['itemsz'],
        #                                                                       iterations[idx]['itemstime'])])
        position = np.array([(xpos, zpos, tpos) for (xpos, zpos, tpos) in zip(xs, zs, ts)])
        active_colors = []
        for active_item in iterations[idx]['itemsactive']:
            if active_item:
                active_colors.append((255, 255, 255, 255))
            else:
                active_colors.append((0, 0, 0, 0))
        active_sizes = np.array([0.5] * len(position))
        click_scatter.setData(pos=position, size=active_sizes, color=np.array(active_colors))
    else:
        click_scatter.setData(pos=click_pos, size=click_size[idx], color=click_color, pxMode=False)


def get_rotation_matrix(i_v, unit=None):
    """
    This function gets a rotation matrix given a vector from a unit vector.

    :param i_v: the vector whose rotation should be calculated
    :param unit: the unit vector for reference

    :return: a rotation matrix
    """
    if unit is None:
        unit = [1.0, 0.0, 0.0]
    i_v = np.divide(i_v, np.sqrt(np.dot(i_v, i_v)))
    u, v, w = np.cross(i_v, unit)
    axis = np.array([u, v, w])
    u, v, w = np.divide(axis, np.sqrt(np.dot(axis, axis)))
    d = np.dot(i_v, unit)
    phi = np.arccos(d)
    rcos = np.cos(phi)
    rsin = np.sin(phi)
    matrix = np.zeros((3, 3))
    matrix[0][0] = rcos + u * u * (1.0 - rcos)
    matrix[1][0] = w * rsin + v * u * (1.0 - rcos)
    matrix[2][0] = -v * rsin + w * u * (1.0 - rcos)
    matrix[0][1] = -w * rsin + u * v * (1.0 - rcos)
    matrix[1][1] = rcos + v * v * (1.0 - rcos)
    matrix[2][1] = u * rsin + w * v * (1.0 - rcos)
    matrix[0][2] = v * rsin + u * w * (1.0 - rcos)
    matrix[1][2] = -u * rsin + v * w * (1.0 - rcos)
    matrix[2][2] = rcos + w * w * (1.0 - rcos)
    return matrix


def generate_normed_segments(path, __meta=None,
                             normalize_translation=True, normalize_length=True, normalize_rotation=True):
    """
    This function generates normalized line segments given a path, meta-data, and normalization flags.

    :param path: the path to the data to process.
    :param __meta: the meta information from the data filename (automatically detected if None)
    :param normalize_translation: if True, translation will be normalized
    :param normalize_length: if True, scaling will be normalized
    :param normalize_rotation: if True, rotation will be normalized
    :return:
    """
    _iterations = read_binary_file(path)
    logging.info("Plotting " + str(len(_iterations)) + " iterations.")
    if __meta is None:
        # noinspection PyBroadException
        try:
            __meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
        except:
            logging.error(
                'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
            exit()
    items, times, directions = get_items_solutions(__meta)
    # noinspection PyRedeclaration
    _click_pos, _click_idx, _, _ = get_click_locations_and_indicies(_iterations, items, __meta)
    _click_idx, _click_pos = [list(l) for l in zip(*sorted(zip(_click_idx, _click_pos)))]
    num_lines = len(_click_idx) - 1
    xs = []
    ys = []
    zs = []
    for line_idx in range(0, num_lines):
        start_idx = int(_click_idx[line_idx])
        end_idx = int(_click_idx[line_idx + 1])
        start_iter = _iterations[start_idx]
        end_iter = _iterations[end_idx - 1]
        start_pos = [float(start_iter['x']), float(start_iter['z']), float(start_iter['time'])]
        end_pos = [float(end_iter['x']), float(end_iter['z']), float(end_iter['time'])]

        original_vector = np.subtract(end_pos, start_pos)
        magnitude = np.sqrt(np.dot(original_vector, original_vector))

        R = get_rotation_matrix(original_vector)

        x = []
        y = []
        z = []
        sub_iterations = _iterations[start_idx:end_idx]
        for _, i in enumerate(sub_iterations):
            xtmp = float(i['x'])
            ytmp = float(i['z'])
            ztmp = float(i['time'])
            if normalize_translation:
                xtmp, ytmp, ztmp = np.subtract([xtmp, ytmp, ztmp], start_pos)
            if normalize_length:
                xtmp, ytmp, ztmp = np.divide([xtmp, ytmp, ztmp], magnitude)
            if normalize_rotation:
                xtmp, ytmp, ztmp = np.dot(np.array([xtmp, ytmp, ztmp]).T, R.T)
            x.append(xtmp)
            y.append(ytmp)
            z.append(ztmp)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return xs, ys, zs, num_lines


def subsetter(path):
    """
    This function is used as a helper to subset data via a particular criteria.

    :param path: the path to the data
    :return: a bool which, if True, suggests keeping the data, otherwise it suggests removing the data from the
             visualization
    """
    __meta = None
    # noinspection PyBroadException
    try:
        __meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
    except:
        logging.error(
            'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
        exit()
    return int(__meta["subID"]) == 29 and int(__meta["trial"]) >= 0


def item_path_visualization(search_directory=None, file_regex="\d\d\d_\d_1_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat"):
    """
    This function visualizes item-to-item paths given some input files.

    :param search_directory: a directory to search recursively for input files
    :param file_regex: a regular expression to search for in the files in search_directory

    :rtype: bool
    :return: True if files were successfully processed, False otherwise.
    """
    import mpl_toolkits.mplot3d

    ####################################################################################################################
    # Setup
    ####################################################################################################################

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Have a look at the colormaps here and decide which one you'd like:
    # http://matplotlib.org/1.2.1/examples/pylab_examples/show_colormaps.html
    colormap = plt.cm.Accent  # winter

    if search_directory is None:
        search_directory = easygui.diropenbox()
    if search_directory is '':
        logging.info('No directory selected, returning.')
        return False
    if not os.path.exists(search_directory):
        raise IOError('Specified search directory was not found.')

    files = find_data_files_in_directory(search_directory, file_regex=file_regex)

    if len(files) == 0:
        logging.info('No files found. returning.')
        return False

    num_subs = 0
    for path in files:
        if subsetter(path):
            num_subs += 1
    count = 0
    for path in files:
        _meta = None
        # noinspection PyBroadException
        try:
            _meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
        except:
            logging.error(
                'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
            exit()
        if subsetter(path):
            xs, ys, zs, num_lines = generate_normed_segments(path, _meta)
            # plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_lines)])
            base_colors = [colormap(i) for i in np.linspace(0, 0.9, num_subs)]
            col_space = [base_colors[count]] * num_lines
            col_space = [(x[0], x[1], x[2], a) for x, a in zip(col_space, np.linspace(0.5, 1, num_lines))]
            labels = [""] * num_lines
            labels[-1] = "{0} trial {1}".format(_meta["subID"], _meta["trial"])
            [ax.plot(x, y, z, label=labels[_idx], color=col_space[_idx]) for _idx, (x, y, z)
             in enumerate(zip(xs, ys, zs))]
            count += 1

    ax.scatter([0, 1], [0, 0], [0, 0], color='k')
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

    '''
    import sklearn.preprocessing as skpre
    from scipy import stats
    from mayavi import mlab

    xss = []
    yss = []
    zss = []
    for path in files:
        meta = None
        # noinspection PyBroadException
        try:
            meta = get_filename_meta_data(os.path.basename(path))  # The meta filename information for convenience
        except:
            logging.error(
                'There was an error reading the filename meta-information. Please confirm this is a valid log file.')
            exit()
        if subsetter(path):
            xs, ys, zs, num_lines = generate_normed_segments(path, meta)
            xss.extend(xs)
            yss.extend(ys)
            zss.extend(zs)

    xss = np.array([item for sublist in xss for item in sublist])
    yss = np.array([item for sublist in yss for item in sublist])
    zss = np.array([item for sublist in zss for item in sublist])

    xyz = np.vstack([xss, yss, zss])
    kde = stats.gaussian_kde(xyz)
    # Evaluate kde on a grid
    xmin, ymin, zmin = xss.min(), yss.min(), zss.min()
    xmax, ymax, zmax = xss.max(), yss.max(), zss.max()
    xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
    coords = np.vstack([item.ravel() for item in [xi, yi, zi]])
    density = kde(coords).reshape(xi.shape)

    # Plot scatter with mayavi
    figure = mlab.figure('DensityPlot')

    grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
    min = density.min()
    max=density.max()
    mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

    mlab.axes()
    mlab.show()
    '''

    return True

if __name__ == '__main__':
    visualize_time_travel_data()
    # The commented visualization below needs testing and does not currently work.
    # item_path_visualization(search_directory=r"C:\Users\Kevin\Desktop\Work\Time Travel Task\v2",
    #                         file_regex="021_\d_1_\d_\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d.dat")
