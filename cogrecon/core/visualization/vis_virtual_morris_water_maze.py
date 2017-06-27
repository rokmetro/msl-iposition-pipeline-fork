import os
import sys
import easygui
import logging

import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui

from scipy.misc import imread

if __name__ == "__main__":
    from cogrecon.core.data_flexing.virtual_morris_water_maze.virtual_morris_water_maze_parser import \
        get_filename_meta_data, read_binary_file, trial_num_to_str
else:
    from ..data_flexing.virtual_morris_water_maze.virtual_morris_water_maze_parser import get_filename_meta_data, \
        read_binary_file, trial_num_to_str


# TODO: Fix issue with pyqt globals (should not used globals)
def visualize(path=None):
    """
    This function visualizes data for the virtual morris water maze task.

    :param path: a path to a log file from the virtual morris water maze task

    :return: Nothing
    """
    # noinspection PyGlobalUndefined
    global path_line, idx, timer, iterations, num_points_to_update, line_color_state, line_color

    ####################################################################################################################
    # Setup
    ####################################################################################################################

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.WARNING)

    show_time = False

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
    w = gl.GLViewWidget()
    w.opts['center'] = pg.Qt.QtGui.QVector3D(0, 0, 0)
    w.opts['elevation'] = 90
    w.opts['azimuth'] = 0
    w.opts['distance'] = 200
    w.setWindowTitle('MSL Virtual Morris Water Maze Visualizer' + ' - Subject {0}, Trial {1}, iteration {2}'.format(
        meta['subid'],
        meta['trial'],
        trial_num_to_str(meta['iteration'])))

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

    radius = 60
    time_length = 60
    if show_time:
        g0 = make_grid_item((-radius, 0, time_length/2), (90, 0, 1, 0), (time_length/20, radius/10, radius/10))
        g1 = make_grid_item((-radius, 0, time_length*3/2), (90, 0, 1, 0), (time_length/20, radius/10, radius/10))
        g2 = make_grid_item((0, -radius, time_length/2), (90, 1, 0, 0), (radius/10, time_length/20, radius/10))
        g3 = make_grid_item((0, -radius, time_length*3/2), (90, 1, 0, 0), (radius/10, time_length/20, radius/10))
        grid_items.append(g0)
        grid_items.append(g1)
        grid_items.append(g2)
        grid_items.append(g3)
        w.addItem(g0)
        w.addItem(g1)
        w.addItem(g2)
        w.addItem(g3)
    gn = make_grid_item((0, 0, 0), (0, 0, 0, 0), (radius/10, radius/10, radius/10))
    grid_items.append(gn)
    w.addItem(gn)

    # Make Image Base

    # Determine the background image according to meta phase
    bg_path = 'maze.png'
    img = imread(os.path.join('./media/virtual_morris_water_maze', bg_path))

    image_scale = (radius * 2.0) / float(img.shape[0])
    tex1 = pg.makeRGBA(img)[0]
    base_image = gl.GLImageItem(tex1)
    base_image.translate(-radius, -radius, 0)
    base_image.rotate(270, 0, 0, 1)
    base_image.scale(image_scale, image_scale, image_scale)
    w.addItem(base_image)

    # Generate Path Line

    color = (255, 255, 255, 255)
    line_color = np.empty((len(iterations), 4))
    line_color_state = np.empty((len(iterations), 4))
    x = []
    y = []
    z = []
    for idx, i in enumerate(iterations):
        x.append(float(i['x']))
        y.append(float(i['z']))
        if show_time:
            z.append(float(i['time']))
        else:
            z.append(0.0)
        line_color[idx] = pg.glColor(color)
        line_color_state[idx] = pg.glColor((0, 0, 0, 0))

    pts = np.vstack([x, y, z]).transpose()
    path_line = gl.GLLinePlotItem(pos=pts, color=line_color_state, mode='line_strip', antialias=True)
    w.addItem(path_line)

    ####################################################################################################################
    # Show UI
    ####################################################################################################################

    w.show()
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
    path_line_visible = True

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

    def toggle_path_line_visible():
        global path_line_visible
        if path_line_visible:
            path_line.hide()
            path_line_visible = False
        else:
            path_line.show()
            path_line_visible = True

    # GUI Initialization
    sh = QtGui.QShortcut(QtGui.QKeySequence("+"), w, speed_up)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("-"), w, speed_down)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence(" "), w, pause)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("R"), w, reset)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("E"), w, go_to_end)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("Escape"), w, close_all)
    sh.setContext(QtCore.Qt.ApplicationShortcut)

    sh = QtGui.QShortcut(QtGui.QKeySequence("1"), w, toggle_grid_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("2"), w, toggle_base_visible)
    sh.setContext(QtCore.Qt.ApplicationShortcut)
    sh = QtGui.QShortcut(QtGui.QKeySequence("5"), w, toggle_path_line_visible)
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
    This function is the animation update function used internally in visualize.

    """
    global path_line, idx, num_points_to_update, line_color_state, line_color
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

if __name__ == "__main__":
    visualize()
