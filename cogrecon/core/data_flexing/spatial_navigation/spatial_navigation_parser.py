import datetime
import logging

import pytz
from tzlocal import get_localzone
import numpy as np
import copy


def get_filename_meta_data(fn, path):
    """
    This function makes a meta-data object given a file path from the spatial navigation task.

    Example input filename: RawLog_Sub999_Trial1_19_22_02_10-04-2017.csv

    :param fn: the filename (basename)
    :param path: the full file path
    :return: a dictionary with 'fileType', 'subID', 'trial', 'phase', and 'datetime' where 'fileType' is either RawLog
             or SummaryLog, 'phase' is either 'practice', 'study', or 'test' and datetime is a datetime object.
    """
    parts = fn.split('_')
    file_type = parts[0]
    sub_id = parts[1].replace('Sub', '').replace('sub', '')
    trial_num = int(parts[2].replace('Trial', '').replace('trial', ''))
    date_time_string = '_'.join(parts[3:]).replace('.csv', '')
    dt = datetime.datetime.strptime(date_time_string, '%H_%M_%S_%d-%m-%Y')
    phase = 'unknown'
    if 'Practice' in path:
        phase = 'practice'
    elif 'Study' in path:
        phase = 'study'
    elif 'Test' in path:
        phase = 'test'
    return {"fileType": file_type, "subID": sub_id, "trial": trial_num, "phase": phase, "datetime": dt}


def datetime_from_dot_net_binary(data):
    """
    This function returns a datetime from a .NET binary datetime representation.

    From http://stackoverflow.com/questions/15919598/serialize-datetime-as-binary

    :param data: the datetime data in .NET binary format
    :return: the Python datetime object corresponding to the input data
    """
    kind = (data % 2 ** 64) >> 62  # This says about UTC and stuff...
    ticks = data & 0x3FFFFFFFFFFFFFFF
    seconds = ticks / 10000000
    tz = pytz.utc
    if kind == 0:
        tz = get_localzone()
    return datetime.datetime(1, 1, 1, tzinfo=tz) + datetime.timedelta(seconds=seconds)


def get_object_info_from_string(info_string):
    """
    This function returns position, rotation and scaling given an object's info string.

    :param info_string: the object info string
    :return: three tuples of floats; position (3), rotation (4), and scaling (3)
    """
    vals = info_string.split(':')[1].split(',')
    pos = (float(vals[0].strip()), float(vals[1].strip()), float(vals[2].strip()))
    rot = (float(vals[3].strip()), float(vals[4].strip()), float(vals[5].strip()), float(vals[6].strip()))
    sca = (float(vals[7].strip()), float(vals[8].strip()), float(vals[9].strip()))
    return pos, rot, sca


def get_object_info_from_summary_string(summary_info_string):
    """
    This function gets name and position given an object's summary string.

    :param summary_info_string: the summary string line for an object
    :return: the tuple containing a string name and position (3)
    """
    split_line = summary_info_string.split(':')
    name = split_line[0].split(',')[1]
    pos_list = split_line[1].replace('(', '').replace(')', '').split(',')
    pos = (float(pos_list[0]), float(pos_list[1]), float(pos_list[2]))
    return name, pos


def read_summary_file(path):
    """
    This function takes in a summary file path and returns a list of the events in the summary file (in the form of a
    dictionary whose members are 'time', 'eventType', 'objectName', and 'location'.

    :param path: the input file path to a summary file

    :return: a list of event dictionaries
    """
    events = []
    with open(path, 'rb') as f:
        f.readline()  # Remove header
        file_string = f.readlines()
    current_dt = None
    for line in file_string:
        if line[0] == '-':
            current_dt = datetime_from_dot_net_binary(int(line.replace(',', '').strip()))
        if line.startswith("ChangeTextureEvent_ObjectClicked"):
            events.append({'time': current_dt, 'eventType': 'clicked', 'objectName': line.split(',')[1].strip()})
        if line.startswith("Object_Placed"):
            name, pos = get_object_info_from_summary_string(line)
            events.append({'time': current_dt, 'eventType': 'placed', 'objectName': name, "location": pos})
        if line.startswith("Object_Picked_Up"):
            name, pos = get_object_info_from_summary_string(line)
            events.append({'time': current_dt, 'eventType': 'picked', 'objectName': name, "location": pos})
        if line.startswith("Object_Identity_Set"):
            name, pos = get_object_info_from_summary_string(line)
            events.append({'time': current_dt, 'eventType': 'identified', 'objectName': name, "location": pos})
        if line.startswith("Object_Identity_Removed"):
            name, pos = get_object_info_from_summary_string(line)
            events.append({'time': current_dt, 'eventType': 'deidentified', 'objectName': name, "location": pos})
    return events


def read_raw_file(path):
    """
    This function reads a raw log file and returns a tuple containing a list of iterations and a list of events. The
    iterations are dictionaries containing 'time' and 'state' keys while the events contain 'time', 'eventType', and
    'objectName' keys.

    :param path: the raw input data file path
    :return: a tuple with a list of iterations and list of events
    """
    iterations = []
    with open(path, 'rb') as f:
        file_string = f.readlines()
    current_dt = None
    current_state = {"Main Camera": None, "First Person Controller": None}
    events = []
    for line in file_string:
        if line[0] == '-':
            if current_dt is not None:
                iterations.append({"time": current_dt, "state": current_state})
                current_state = copy.deepcopy(current_state)
            current_dt = datetime_from_dot_net_binary(int(line.strip()))
        if line.startswith('Main Camera'):
            pos, rot, sca = get_object_info_from_string(line.strip())
            current_state["Main Camera"] = {"position": pos, "rotation": rot, "scale": sca}
        if line.startswith('First Person Controller'):
            pos, rot, sca = get_object_info_from_string(line.strip())
            current_state["First Person Controller"] = {"position": pos, "rotation": rot, "scale": sca}
        if line.startswith("ChangeTextureEvent_ObjectClicked"):
            events.append({'time': current_dt, 'eventType': 'clicked', 'objectName': line.split(',')[1].strip()})
        if line.strip() == 'End of File':
            logging.debug('End of File')
    return iterations, events


def get_simple_path_from_raw_iterations(raw_iterations, make_2d=True):
    """
    This function gets a simple list of location points given raw data iterations.

    :param raw_iterations: the iterations list from read_raw_file
    :param make_2d: if True, only X and Z are returned (Y is omitted) as these are the only coordinates which change
                    in the task

    :return: a list of location points
    """
    points = []
    for i in raw_iterations:
        p = i['state']['First Person Controller']['position']
        if make_2d:
            points.append((p[0], p[2]))
        else:
            points.append(p)
    return np.array(points)


def quat2euler(q):
    """
    This function converts a quaternion into roll, pitch and yaw values

    :param q: a quaternion

    :return: a tuple containing roll, pitch and yaw
    """
    roll = np.arctan2(2*(q[1]*q[3] + q[0]*q[2]), 1-2*(q[1]*q[1]+q[2]*q[2]))
    pitch = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1-2*(q[0]*q[0]+q[2]*q[2]))
    yaw = np.arcsin(2 * (q[0] * q[1] - q[2] * q[3]))
    return roll, pitch, yaw


def get_simple_orientation_path_from_raw_iterations(raw_iterations):
    """
    This function returns a list of angles given the iterations from read_raw_file.

    :param raw_iterations: the iterations from read_raw_file

    :return: a list of angles
    """
    angles = []
    for i in raw_iterations:
        p = i['state']['First Person Controller']['rotation']
        x, y, z = quat2euler(p)
        angles.append(np.pi - x - np.pi/2.)
    return angles


def compress(pos, orient):
    """
    This function compresses an input of positions and orientations, removing repeats to make the data size smaller.

    :param pos: a list of positions
    :param orient:  a list of orientation angles

    :return: a tuple containing a list of positions and list of orientations with consecutive duplicates removed
    """
    new_pos = [pos[0]]
    new_orient = [orient[0]]
    for p, o in zip(pos, orient)[1:]:
        # noinspection PyTypeChecker
        if all(new_pos[-1] == p) and new_orient[-1] == o:
            continue
        else:
            new_pos.append(p)
            new_orient.append(o)
    return np.array(new_pos), new_orient


def get_final_state_from_summary_events():  # summary_events):
    """
    This function will eventually return a final item state given the summary events from read_summary_file.
    """
    raise NotImplemented


def validate_summary_events_are_complete():  # summary_events):
    """
    This function will eventually return a bool value which is True if the summary events from read_summary_file are
    complete.
    """
    raise NotImplemented


def compare_summary_and_raw_events():  # raw_events, summary_events):
    """
    This function will eventually return a bool value which is True if the summary events from read_summary_file and
    raw events from read_raw_file match.
    """
    raise NotImplemented
