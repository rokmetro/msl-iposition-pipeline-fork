import datetime
import logging
import copy
import os
import csv
import re
import time
import math

import numpy
import numpy as np

import pytz
from tzlocal import get_localzone

from enum import Enum
from scipy.spatial import distance


########################################################################################################################
# 2-Room Parser
########################################################################################################################

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

########################################################################################################################
# 4-Room Parser
########################################################################################################################

"""
These variables represent the configuration of the test items and their contexts. test_labels is the list
of all of the labels (in the 2d test log files) for each item. test_realX and test_realY comprise the 2d test image
space locations of the objects from test_labels (in order). test_context_boundaries is a list containing the
x, y, width, and height of each context (in order defined by context_labels) in the 2d test image space.
"""
test_labels = ["purse", "crown", "basketball", "boot", "emerald", "clover", "bandana", "guitar", "fire extinguisher",
               "hammer", "lemon", "ice cube", "bottle", "ketchup", "boxing gloves", "crab"]
test_realX = [-210, -163, -163, 211, -163, 352, 71, -117, 71, 258, -304, -304, -70, 305, 305, 305]
test_realY = [-211, -304, -70, -258, 164, -211, -211, 258, 258, 258, -117, 258, 70, 211, -70, 70]
test_context_boundries = [{'x': -375, 'y': 0, 'w': 375, 'h': 375}, {'x': 0, 'y': 0, 'w': 375, 'h': 375},
                          {'x': -375, 'y': -375, 'w': 375, 'h': 375}, {'x': 0, 'y': -375, 'w': 375, 'h': 375}]

"""
These variables represent the configuration of the study items and their contexts. study_labels is the list
of all of the labels (in the Unity log files) for each item. study_realX and study_realY comprise the Unity space
locations of the objects from study_labels (in order). study_context_boundaries is a list containing the
x, y, width, and height of each context (in order defined by context_labels) in the Unity space.
"""
study_labels = ["PurseCube", "CrownCube", "BasketballCube", "BootCube", "EmeraldCube", "CloverCube", "BandanaCube",
                "GuitarCube", "FireExtCube",
                "HammerCube", "LemonCube", "IceCubeCube", "BottleCube", "KetchupCube", "BoxingGloveCube", "CrabCube"]
study_realX = [5, 10, 10, 50, 10, 65, 35, 15, 35, 55, -5, -5, 20, 60, 60, 60]
study_realY = [15, 5, 30, 10, 55, 15, 15, 65, 65, 65, 25, 65, 45, 60, 30, 45]
study_context_boundries = [{'x': -12.5, 'y': 37.5, 'w': 40, 'h': 40}, {'x': 27.5, 'y': 37.5, 'w': 40, 'h': 40},
                           {'x': -12.5, 'y': -2.5, 'w': 40, 'h': 40}, {'x': 27.5, 'y': -2.5, 'w': 40, 'h': 40}]

"""
These values represent the context labels and the indicies (from the other variable lists) of the associated
items in the given contexts.
"""
context_labels = ["red", "green", "blue", "yellow"]
context_item_indicies = [[4, 7, 11, 12], [8, 9, 13, 15], [0, 1, 2, 10], [3, 5, 6, 14]]

"""
The following configuration variables have to do with file naming formats and internal saving representations.
These values should never be changed unless changes are being made to saving/naming formats. They're held as
constants at the top of this file for convenience in case such changes need to be made, but seriously, don't
change them unless you know when you're doing.
"""

"""
This variable represents the number of lines to be skipped in the parsing of the 2d test files. This number would
only likely need changing if additional objects are added (or objects are removed) to the 2d test.
"""
test_skip_lines = 124

"""
These values represent the test2d and navigation input file date/time tag format and associated parsing format string
(for finding and for reading file date/times)
"""
test2d_time_format_regex = "(\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d-[AP]M)"
test2d_time_format_string = "%Y-%m-%d_%I-%M-%S-%p"
nav_time_format_regex = "\d\d_\d\d_\d\d_\d\d-\d\d-\d\d\d\d"
nav_time_format_string = "%H_%M_%S_%d-%m-%Y"

"""
These values represent the regular expressions being used to find the three types of files of interest
"""
nav_raw_regex_search_key = 'C:.*\\\RawLog.*'
nav_summary_regex_search_key = 'C:.*\\\SummaryLog.*'
test2d_raw_regex_search_key = 'C:.*\\\GMDA.*_Raw\.csv'

"""
These values represent the search string (_search_key) to be used to determine what phase a file is while the
_store_key_ and _store_key values represent the internal dictionary representations which are used
to store the various file types in the parser dictionary. The search key may need to be adjusted if file naming
changes one day, but the store keys should almost never have a reason to be changed.
"""
practice_search_key = 'practice'
practice_store_key_raw = 'practice'
practice_store_key_summary = 'practice_summary'

study_search_key = 'study'
study_store_key_raw = 'study'
study_store_key_summary = 'study_summary'

test_search_key = 'test'
test_store_key_raw = 'test'
test_store_key_summary = 'test_summary'

test2d_store_key = 'test2d'


class Individual:
    """
    The Individual object stores and assists with processing of information pertaining to an individual subject.
    """

    def __init__(self):
        """
        The Individual object initializes with a subject_id of None and an empty list of trials.
        """
        self.subject_id = None  # string of the subject id
        self.trials = []  # list of the trials as Trial objects

    def meets_trial_number_requirement(self, required_number_of_trials, require_complete_trials):
        """
        This helper function is to confirm the trial requirements are met.

        :param required_number_of_trials: an integer number of trials which are required
        :param require_complete_trials: a bool which, if True, requires that all trials be complete

        :return: if True, all trials meet the requirements
        """
        # Count the number of complete trials
        complete_trials_count = 0
        for t in self.trials:
            if t.is_complete():
                complete_trials_count += 1
        # Use the correct trial count based on the completion requirement
        trials_count = len(self.trials)
        if require_complete_trials:
            trials_count = complete_trials_count

        # Return the requirement criteria
        return trials_count >= required_number_of_trials

    def get_full_file_list(self):
        """
        This helper function is to get the list of any files reference in this individual.

        :return: a list of all files across all trials in this individual (with None files excluded)
        """
        files = []
        for t in self.trials:
            files.extend(t.get_full_file_list())
        return filter(None, files)


class Trial:
    """
    This class contains information about a Trial, as well as Trial specific helper functions.
    """

    def __init__(self):
        """
        Initializing a trial defaults to all None and -1 data.
        """
        self.num = -1  # The trial number
        # The paths to the associated files
        self.study_path = None
        self.study_look = None
        self.study_summary = None
        self.test_path = None
        self.test_look = None
        self.test_summary = None  # This element will always be the same as test_vr
        self.practice_path = None
        self.practice_look = None
        self.practice_summary = None
        self.test_2d = None
        self.test_vr = None

    def is_complete(self):
        """
        This helper function determines if a Trial contains complete data.

        Note: Completion does not require a Practice phase.

        :return: True if data is complete.
        """
        return self.num >= 0 and self.study_path is not None and self.study_look is not None \
            and self.test_path is not None and self.test_look is not None \
            and self.test_2d is not None and self.test_vr is not None \
            and self.study_summary is not None and self.test_summary is not None

    def get_full_file_list(self):
        """
        This function returns a full list of the Trial files. None files are automatically removed.

        :return: a list of files in the following order: study_path, study_look, study_summary, test_path, test_look,
                 test_summary, practice_path, practice_look, practice_summary, test_2d, test_vr
        """
        files = [self.study_path, self.study_look, self.study_summary, self.test_path, self.test_look,
                 self.test_summary, self.practice_path,
                 self.practice_look, self.practice_summary, self.test_2d, self.test_vr]
        return filter(None, files)

    def all_trial_dates_match(self):
        """
        This function determines if the datetimes across the study, test and practice match.

        :return: True if file datetimes match, false otherwise.
        """
        files = [[self.study_path, self.study_look, self.study_summary],
                 [self.test_path, self.test_look, self.test_summary],
                 [self.practice_path, self.practice_look, self.practice_summary]]
        match = True
        for file_set in files:
            root_date = None
            for f in file_set:
                if f:
                    if not root_date:
                        root_date = extract_date_time_from_filename_custom(f)
                    else:
                        current_date = extract_date_time_from_filename_custom(f)
                        if not root_date == current_date:
                            match = False
                            break
        return match


def extract_date_time_from_filename_custom(filename):
    """
    Helper function which will extract a datetime object given two possible formats of datetime strings (the formats
    which are used by the Unity program and the 2D test program.

    :param filename: the filename (basename) from which to extract the datetime
    :return: the Python datetime object associated with the filename contents
    """
    # 15_04_56_20-01-2016 , raw log
    # 2016-01-20_03-16-31-PM , test
    # Use a regular expression to find the file format portion of the filename
    re_test_time = re.compile(test2d_time_format_regex)
    re_nav_time = re.compile(nav_time_format_regex)

    # Determine which format is being used and parse it
    datetime_val = None
    if re_test_time.search(filename):
        datetime_val = time.strptime(re_test_time.findall(filename)[0], test2d_time_format_string)
    if re_nav_time.search(filename):
        datetime_val = time.strptime(re_nav_time.findall(filename)[0], nav_time_format_string)

    # Convert the time struct to a datetime and return it
    return datetime.datetime.fromtimestamp(time.mktime(datetime_val))


def catalog_files(files, min_num_trials, exclude_incomplete_trials):
    """
    This function should produce an Individual list filled according to input restrictions.

    :param files: a list of files
    :param min_num_trials: a minimum criteria for number of trials
    :param exclude_incomplete_trials: if True, incomplete trials are excluded

    :return: a tuple containing a list of Individual objects, a list of excluded files, and a list of non matching files
    """
    # First, isolate each file type we care about
    raw_files = []
    summary_files = []
    test_2d_raw_files = []
    non_matching_files = []
    other_files = []

    # Generate regular expressions to determine if a particular file is of a particular type
    re_raw = re.compile(nav_raw_regex_search_key)
    re_summary = re.compile(nav_summary_regex_search_key)
    re_test_2d_raw = re.compile(test2d_raw_regex_search_key)

    # Search all the files and classify them by type
    for f in files:
        if re_raw.match(f):
            raw_files.append(f)
        elif re_summary.match(f):
            summary_files.append(f)
        elif re_test_2d_raw.match(f):
            test_2d_raw_files.append(f)
        else:
            non_matching_files.append(f)

    # Generate dictionary of individuals for various phases (practice, study, test)
    subject_dict = dict()
    search_subtypes = [practice_search_key, study_search_key, test_search_key]
    store_subtypes = [practice_store_key_raw, study_store_key_raw, test_store_key_raw]
    for f in raw_files:
        subtype_found = False
        for search, store in zip(search_subtypes, store_subtypes):
            # Extract the subject id from the basename for the file
            basename = os.path.basename(f)
            subject_id_split = basename.split('_')
            subject_id = subject_id_split[1][3:]
            # Search for the particular file subtype and store it in the dictionary if it is found
            search_lower = search.lower()
            store_lower = store.lower()
            if search_lower in f.lower():
                subtype_found = True
                if subject_id in subject_dict:
                    # If the subject is already in the dictionary
                    if store_lower in subject_dict[subject_id]:
                        # If the subtype is already in the dictionary, append the current file to that list
                        subject_dict[subject_id][store_lower].append(f)
                    else:
                        # If the subtype is not in the dictionary, add it and add the current file to the list
                        subject_dict[subject_id][store_lower] = [f]
                else:
                    # If the subject isn't in the dictionary, add them and add the subtype field
                    subject_dict[subject_id] = dict()
                    subject_dict[subject_id][store_lower] = [f]
        if not subtype_found:
            # If the file had been classified as a raw file type but wasn't added to any field, add it to other_files
            other_files.append(f)

    # Generate dictionary of summary files for individuals for various phases (practice, study, test)
    search_subtypes = [practice_search_key, study_search_key, test_search_key]
    store_subtypes = [practice_store_key_summary, study_store_key_summary, test_store_key_summary]
    for f in summary_files:
        subtype_found = False
        for search, store in zip(search_subtypes, store_subtypes):
            # Extract the subject id from the basename for the file
            basename = os.path.basename(f)
            subject_id_split = basename.split('_')
            subject_id = subject_id_split[1][3:]
            # Search for the particular file subtype and store it in the dictionary if it is found
            search_lower = search.lower()
            store_lower = store.lower()
            if search_lower in f.lower():
                subtype_found = True
                if subject_id in subject_dict:
                    # If the subject is already in the dictionary
                    if store_lower in subject_dict[subject_id]:
                        # If the subtype is already in the dictionary, append the current file to that list
                        subject_dict[subject_id][store_lower].append(f)
                    else:
                        # If the subtype is not in the dictionary, add it and add the current file to the list
                        subject_dict[subject_id][store_lower] = [f]
                else:
                    # If the subject isn't in the dictionary, add them and add the subtype field
                    subject_dict[subject_id] = dict()
                    subject_dict[subject_id][store_lower] = [f]
        if not subtype_found:
            # If the file had been classified as a raw file type but wasn't added to any field, add it to other_files
            other_files.append(f)

    # Search for the 2D files and add them to the data structure
    store_string = test2d_store_key
    for f in test_2d_raw_files:
        # Once again, extract subject id by location in basename string
        basename = os.path.basename(f)
        subject_id = basename[5:8]
        # Find subject in dictionary
        if subject_id in subject_dict:
            if store_string in subject_dict[subject_id]:
                # If the subject and test2d list are present, add the file to that list
                subject_dict[subject_id][store_string].append(f)
            else:
                # If the subject is there but the test2d list isn't, add the list and the file
                subject_dict[subject_id][store_string] = [f]
        else:
            # If subject isn't there, add them, then add the test2d list and file
            subject_dict[subject_id] = dict()
            subject_dict[subject_id][store_string] = [f]

    individuals = []

    # Sort files in each phase according to file date/time
    for subject_id in subject_dict:
        # Create a new individual and store their subject id (extracted earlier)
        new_individual = Individual()
        new_individual.subject_id = subject_id
        new_individual.trials = []
        # Create a temporary variable for determining the max number of trials in any given phase type
        max_trial_phase_count = 0
        # Iterate through the phase types, sort the files by date/time (to get order), and generate Trial objects
        for phase_type in subject_dict[subject_id]:
            dates = []
            # If the current phase type has more files than any other, set the max_trial_phase_count to that length
            if len(subject_dict[subject_id][phase_type]) > max_trial_phase_count:
                max_trial_phase_count = len(subject_dict[subject_id][phase_type])

            # For the files in this phase type, extract the file date/time as a datetime object and add it to dates list
            for f in subject_dict[subject_id][phase_type]:
                dates.append(extract_date_time_from_filename_custom(f))
            # Zip the datetimes and filenames
            zipped = zip(dates, subject_dict[subject_id][phase_type])
            # If there's more than one file in the list, sort it
            if len(zipped) >= 2:
                # Sort by Year, Month, Day, Hour, Minute, Second
                sorted_zipped = sorted(zipped, key=lambda x: datetime.datetime.strftime(x[0], "%Y %m %d %H %M %S"))
                # Extract the sorted data back into the structure
                dates, subject_dict[subject_id][phase_type] = zip(*sorted_zipped)
        # Iterate through each element in each phase type, creating trials that unify them all
        for i in range(0, max_trial_phase_count):
            # Create a new trial with the correct number
            new_trial = Trial()
            new_trial.num = i
            # Extract practice, study, test, test2d, and testvr files from the data structure (if they are available)
            # and add them to the trial object.
            if (practice_store_key_raw in subject_dict[subject_id]) and len(
                    subject_dict[subject_id][practice_store_key_raw]) > i:
                new_trial.practice_path = subject_dict[subject_id][practice_store_key_raw][i]
                new_trial.practice_look = subject_dict[subject_id][practice_store_key_raw][i]
            if (study_store_key_raw in subject_dict[subject_id]) and len(
                    subject_dict[subject_id][study_store_key_raw]) > i:
                new_trial.study_path = subject_dict[subject_id][study_store_key_raw][i]
                new_trial.study_look = subject_dict[subject_id][study_store_key_raw][i]
            if (test_store_key_raw in subject_dict[subject_id]) and len(
                    subject_dict[subject_id][test_store_key_raw]) > i:
                new_trial.test_path = subject_dict[subject_id][test_store_key_raw][i]
                new_trial.test_look = subject_dict[subject_id][test_store_key_raw][i]
            if (test2d_store_key in subject_dict[subject_id]) and len(subject_dict[subject_id][test2d_store_key]) > i:
                new_trial.test_2d = subject_dict[subject_id][test2d_store_key][i]
            if (test_store_key_summary in subject_dict[subject_id]) and len(
                    subject_dict[subject_id][test_store_key_summary]) > i:
                new_trial.test_vr = subject_dict[subject_id][test_store_key_summary][i]
                new_trial.test_summary = subject_dict[subject_id][test_store_key_summary][i]
            if (study_store_key_summary in subject_dict[subject_id]) and len(
                    subject_dict[subject_id][study_store_key_summary]) > i:
                new_trial.study_summary = subject_dict[subject_id][study_store_key_summary][i]
            if (practice_store_key_summary in subject_dict[subject_id]) and len(
                    subject_dict[subject_id][practice_store_key_summary]) > i:
                new_trial.practice_summary = subject_dict[subject_id][practice_store_key_summary][i]

            # Check if trial should be excluded based on the input criteria and trial status
            if exclude_incomplete_trials and not new_trial.is_complete():
                # If it should be excluded, add its files to the other_files list
                other_files.extend(new_trial.get_full_file_list())
                continue
            else:
                # If it should not be excluded, add it to the individual
                if new_trial.all_trial_dates_match():
                    new_individual.trials.append(new_trial)
                else:
                    logging.error(("Error: In cataloging trials a trial was found to have non-matching dates " +
                                   "(Subject ID: %s; Trial #: %d).") % (new_individual.subject_id, new_trial.num))

        # If the individual (now completely constructed) meets the minimum trial number requirement
        if new_individual.meets_trial_number_requirement(min_num_trials, exclude_incomplete_trials):
            # Add them to the output list
            individuals.append(new_individual)

    # Convert the other_files list into a set then back to a list to confirm uniqueness of items
    excluded_files = list(set(other_files))
    non_matching_files = list(set(non_matching_files))
    # Return the now constructed individuals objects and excluded files list
    return individuals, excluded_files, non_matching_files


def make_output_file(directory, filename, header):
    """
    Helper function which will make a csv writer reference to be passed around for file writing.

    :param directory: the directory to which the output file should be saved
    :param filename: the filename (basename) of the file to output
    :param header: the header to write before returning

    :return: a tuple containing the csv writer and file pointer to the output file
    """
    output = open(os.path.join(directory, filename), 'wb')
    writer = csv.writer(output)
    writer.writerow(header)
    return writer, output


class FileType(Enum):
    """
    This Enum represents the four possible file types.
    """
    path_file = 1
    look_file = 2
    test_file_2d = 3
    test_file_vr = 4


def parse_file_and_write(path, subject_id, trial_num, file_type, output_file_writer, summary_file_path):
    """
    Helper function which can be used externally to parse any given file type into the appropriate output format
    then write those rows to the appropriate output file.

    :param path: the path to the log file
    :param subject_id: the subject identifier
    :param trial_num: the trial number
    :param file_type: the type of file to be parsed
    :param output_file_writer: the output file writer pointer
    :param summary_file_path: the path to the summary file
    :return: Nothing or [] if path is empty
    """
    # Check for empty path
    if not path:
        return []

    try:
        rows = []
        if file_type == FileType.path_file:
            rows = parse_path_file(path, subject_id, trial_num, summary_file_path)
        elif file_type == FileType.look_file:
            rows = parse_look_file(path, subject_id, trial_num, summary_file_path)
        elif file_type == FileType.test_file_2d:
            rows = parse_test_2d_file(path, subject_id, trial_num, summary_file_path)
        elif file_type == FileType.test_file_vr:
            rows = parse_test_vr_file(path, subject_id, trial_num, summary_file_path)
        else:
            logging.error("Error: The parse_file function 'type' parameter was not a recognized type.")
        if rows:
            output_file_writer.writerows(rows)
    except LogParseError as e:
        logging.warning("Subject %s, Trial %d, (%s) did not parse successfully (Exception: %s). Skipping..."
                        % (subject_id, trial_num, path, e.message))


class LogParseError(Exception):
    """
    This Exception object is a specialized exception for use internally.
    """

    def __init__(self, *args):
        """
        Initializes an exception.

        :param args: The exception arguments.
        :param kwargs: The exception kwargs.
        """
        Exception.__init__(self, *args)


class SummaryType(Enum):
    """
    The summary type enum is for readability and convenience. It differentiates between when a test vs study/practice
    summary is present to know which schema to use (unfortunately, they use different syntax).
    """
    test = 1
    study_practice = 2
    unknown = 3


def parse_summary_file(path):
    """
    This helper function will parse a summary file of either type and return summary type, times, event types,
    object names (in the format of the summary type), and location (if test type, the placed location, if study/practice
    type, the location the object was when clicked).

    :param path: a string path to a summary file

    :return: a tuple containing summary_type, times, event_types, object_types, and locations
    """
    # Read the entire file into memory
    fp = open(path, 'rb')
    data = fp.readlines()
    fp.close()

    times = []
    event_types = []
    object_types = []
    locations = []

    summary_type = SummaryType.unknown
    # Detect which type of summary file this is (study/practice or test)
    if any(['ChangeTextureEvent_ObjectClicked' in line for line in data]):
        summary_type = SummaryType.study_practice
    elif any(['Object_Placed' in line for line in data]):
        summary_type = SummaryType.test

    # This line_count business just skips the first line of the data (or can be reconfigured to skip more if necessary
    line_count = 0
    for line in data:
        if line_count == 0:
            line_count += 1
            continue
        if line[0] == '-':
            # Otherwise, extract the current time only
            t_raw = int(line[0:20])
            times.append(t_raw)
        else:
            if summary_type == SummaryType.test:
                # Parse the line from the test summary type
                split_str = line.split(':')
                split_str_first = split_str[0].strip().split(',')
                event_types.append(split_str_first[0].strip())
                object_types.append(split_str_first[1].strip())
                location_strings = split_str[1].strip()[1:-1].split(',')
                locations.append((float(location_strings[0]), float(location_strings[1]), float(location_strings[2])))
            if summary_type == SummaryType.study_practice:
                # Parse the line from the study/practice summary type and capture the non-test location
                split_str = line.split(',')
                event_types.append(split_str[0].strip())
                object_types.append(split_str[1].strip())
                locations.append(get_location_by_name(split_str[1].strip(), test2d=False))

    expected_elements = len(study_labels)
    if practice_search_key in path.lower():
        expected_elements /= 2
    if len(times) < expected_elements:
        logging.warning(("Warning: The summary file %s contains an incomplete accounting of objects. " +
                         "This may impact parsing in unpredictable ways.") % path)
    return summary_type, times, event_types, object_types, locations


def parse_nav_file(path, subject_id, trial_number, summary_file_path, process_as_look=False):
    """
    Internal parser for navigation files which can handle either path (process_as_look=False)
    or look (process_as_look=True) elements. Formerly, this code was broken into two functions (parse_path_file and
    parse_look_file), but their code were almost identical copies, so they were combined for maintainability.

    :param path: the file path to the raw data file
    :param subject_id: the subject identifier
    :param trial_number: the trial number
    :param summary_file_path: the file path to the summary data file
    :param process_as_look: if True, the navigation file will be parsed via looking behavior rather than position

    :return: a list of rows with the navigation data contents
    """
    # Read the entire file into memory
    fp = open(path, 'rb')
    data = fp.readlines()
    fp.close()

    # Get data from associated summary file
    summary_type, summary_times, summary_event_types, summary_object_types, summary_locations = parse_summary_file(
        summary_file_path)
    summary_index_tracker = 0

    # Initialize time and vector variables
    t0 = 0
    prev_t = 0
    tn = 0
    prev_v = []
    count = 0

    # Initialize other output variables
    items_clicked = 0
    previous_context_color = None
    context_number = 0
    most_recent_item_interaction = "None"

    # Initialize buffer for storing output lines
    out_lines = []

    for line in data:
        # If this is the first time point
        if t0 == 0 and line[0] == '-':
            # Initialize the time values
            t0 = int(line[0:20])
        elif not t0 == 0 and line[0] == '-':
            # Otherwise, extract the current time only
            tn = int(line[0:20])
        # Extract by name the position vector
        if ((not process_as_look) and 'First Person Controller' in line) or (process_as_look and 'Main Camera' in line):
            offset = 24
            # Add exception for test renaming
            if 'First Person Controller Test' in line:
                offset = 29
            if process_as_look:
                offset = 12

            # Turn it into a float vector
            split_v = line[offset:].split(',')
            v = [float(split_v[0]), float(split_v[1]), float(split_v[2]), float(split_v[3]),
                 float(split_v[4]), float(split_v[5]), float(split_v[6]), float(split_v[7]),
                 float(split_v[8]), float(split_v[9])]

            # Calculate the relative times
            t = tn - t0
            if count == 0:
                t = 0
                prev_t = 0
                count += 1
            time_since_last_point = t - prev_t

            # If there's not a previous vector, just use the current (first) one
            if not prev_v:
                prev_v = v

            # Use summary file to determine how many items have been placed/clicked
            if summary_index_tracker < len(summary_times) and t > (summary_times[summary_index_tracker] - t0):
                if summary_type == SummaryType.test:
                    if 'placed' in summary_event_types[summary_index_tracker].lower():
                        items_clicked += 1
                    elif 'picked' in summary_event_types[summary_index_tracker].lower():
                        items_clicked -= 1
                elif summary_type == SummaryType.study_practice:
                    items_clicked += 1
                most_recent_item_interaction = summary_object_types[summary_index_tracker]
                summary_index_tracker += 1

            # Calculate room color in navigation space
            index, room = nav_get_room_by_location((v[0], v[2]))
            room_by_color = room

            # Determine if this is first iteration (set previous variable to current so no counting is done)
            if not previous_context_color:
                previous_context_color = room_by_color
            # If the previous room is different from the current room, iterate the context_number
            if not (previous_context_color == room_by_color):
                context_number += 1
            # Update the previous context state
            previous_context_color = room_by_color
            # Apply the order number
            room_by_order = context_number

            # Calculate distance on this tick
            distance_from_last_point = distance.euclidean(prev_v[0:3], v[0:3])

            if process_as_look:
                ex, ey, ez = calculate_euler_vector_from_quaternion(v[3], v[4], v[5], v[6])

                # Create the output line and write it to the output buffer
                out_line = [subject_id, trial_number, t, v[3], v[4], v[5], v[6], ex, ey, ez, room_by_order,
                            room_by_color, items_clicked, distance_from_last_point, time_since_last_point,
                            most_recent_item_interaction]
                out_lines.append(out_line)
            else:
                # Create the output line and write it to the output buffer
                out_line = [subject_id, trial_number, t, v[0], v[1], v[2], room_by_order, room_by_color, items_clicked,
                            distance_from_last_point, time_since_last_point, most_recent_item_interaction]
                out_lines.append(out_line)

            # Update prev variables
            prev_t = t
            prev_v = v

    return out_lines


def parse_path_file(path, subject_id, trial_number, summary_file_path):
    """
    A special parser for path files (Raw Unity). It should produce lines with following format:
    subject_id,trial_number,time,x,y,z,room_by_order,room_by_color,items_clicked,distance_from_last_point,
    time_since_last_point

    :param path: the file path to the raw data file
    :param subject_id: the subject identifier
    :param trial_number: the trial number
    :param summary_file_path: the file path to the summary data file

    :return: a list of rows with the following contents: subject_id,trial_number,time,x,y,z,room_by_order,room_by_color,
             items_clicked,distance_from_last_point,time_since_last_point
    """
    return parse_nav_file(path, subject_id, trial_number, summary_file_path, process_as_look=False)


def parse_look_file(path, subject_id, trial_number, summary_file_path):
    """
    A special parser for look files (Raw Unity). It should produce lines with following format:
    subject_id,trial_number,time,x,y,z,w,euler_x,euler_y,euler_z,room_by_order,room_by_color,items_clicked,
    distance_from_last_point,time_since_last_point

    :param path: the file path to the raw data file
    :param subject_id: the subject identifier
    :param trial_number: the trial number
    :param summary_file_path: the file path to the summary data file

    :return: a list of rows with the following contents: subject_id,trial_number,time,x,y,z,w,euler_x,euler_y,euler_z,
             room_by_order,room_by_color,items_clicked,distance_from_last_point,time_since_last_point
    """
    return parse_nav_file(path, subject_id, trial_number, summary_file_path, process_as_look=True)


def parse_test_2d_file(path, subject_id, trial_number, summary_file_path):
    """
    A special parser for 2d test files (2D test files). It should produce lines with following format:
    subject_id,trial_number,item_id,x_placed,y_placed,x_expected,y_expected,order_clicked_study,expected_room_by_order,
    expected_room_by_color,actual_room_by_order,actual_room_by_color

    :param path: the file path to the raw data file
    :param subject_id: the subject identifier
    :param trial_number: the trial number
    :param summary_file_path: the file path to the summary data file

    :return: a list of rows with the following contents: subject_id,trial_number,item_id,x_placed,y_placed,x_expected,
             y_expected,order_clicked_study,expected_room_by_order,expected_room_by_color,actual_room_by_order,
             actual_room_by_color
    """
    f = open(path, 'rb')

    # Get the summary file for the test data (study summary)
    summary_type, summary_times, summary_event_types, summary_object_types, summary_locations = parse_summary_file(
        summary_file_path)

    previous_context_color = None
    context_color_order_mapping = dict()
    context_order_counter = 0
    # Get from the summary file the context_color->context_order mapping via the order of item clicks
    for t, event_type, object_type_study, loc in zip(summary_times, summary_event_types, summary_object_types,
                                                     summary_locations):
        index, current_context_color = nav_get_room_by_location(get_location_by_name(object_type_study, test2d=False))
        if not previous_context_color:
            previous_context_color = current_context_color
            context_color_order_mapping[current_context_color] = context_order_counter
        if not (previous_context_color == current_context_color):
            context_order_counter += 1
            if not (current_context_color in context_color_order_mapping):
                context_color_order_mapping[current_context_color] = context_order_counter
        previous_context_color = current_context_color

    out_lines = []

    # Skip the unnecessary lines at the top of the file
    # noinspection PyRedeclaration
    for j in range(0, test_skip_lines):
        f.readline()
    # Read the lines for each object
    # noinspection PyRedeclaration
    for j in range(0, len(test_labels)):
        # Parse the line, extracting the x, y points
        line = f.readline()
        split_line = line.split(',')
        if not line:
            raise LogParseError("The test 2d file had a problem parsing. There were not enough lines.")
        x = int(split_line[3])
        y = int(split_line[4])

        # Get the object type and use it to determine (looking at the summary file) in what order the object was viewed
        # during study time
        object_type_test = split_line[0]
        object_type_study = study_labels[test_labels.index(object_type_test)]  # Swap label schema for study labels
        order_clicked_study = None
        for i, study_object in zip(range(0, len(summary_object_types)), summary_object_types):
            if study_object == object_type_study:
                order_clicked_study = i
                break

        # Using the object label, determine what color room the object should have been in
        expected_room_by_color = None
        object_label_index = test_labels.index(split_line[0])
        for i, l in zip(range(0, len(context_item_indicies)), context_item_indicies):
            if object_label_index in l:
                expected_room_by_color = context_labels[i]
                break

        # Using the object location, get the color of the room the object was actually placed in
        index, room = test2d_get_room_by_location((x, y))
        actual_room_by_color = room

        # Using the name of the object, find the expected location
        x_expected, y_expected = get_location_by_name(object_type_test, test2d=True)

        # Using the room colors and the previously generated color->order mapping, get the expected/actual room order
        try:
            expected_room_by_order = context_color_order_mapping[expected_room_by_color]
        except KeyError:
            expected_room_by_order = -1  # If the expectation isn't in the map, return -1
        try:
            actual_room_by_order = context_color_order_mapping[actual_room_by_color]
        except KeyError:
            actual_room_by_order = -1  # If the expectation isn't in the map, return -1

        # Generate and append the output line
        out_line = [subject_id, trial_number, test_labels[j], x, y, x_expected, y_expected, order_clicked_study,
                    expected_room_by_order, expected_room_by_color, actual_room_by_order, actual_room_by_color]
        out_lines.append(out_line)

    return out_lines


def parse_test_vr_file(path, subject_id, trial_number, summary_file_path):
    """
    This is a special parser for vr test files (Summary Unity Test). It should produce lines with following format:

    subject_id,trial_number,item_id,x_placed,y_placed,x_expected,y_expected,order_clicked_study,expected_room_by_order,
    expected_room_by_color,actual_room_by_order,actual_room_by_color,number_of_replacements,time_placed

    :param path: the file path to the raw data file
    :param subject_id: the subject identifier
    :param trial_number: the trial number
    :param summary_file_path: the file path to the summary data file
    :return: a list of rows with the following contents: subject_id,trial_number,item_id,x_placed,y_placed,x_expected,
             y_expected,order_clicked_study,expected_room_by_order,expected_room_by_color,actual_room_by_order,
             actual_room_by_color,number_of_replacements,time_placed
    """
    # Get data from associated summary files
    test_type, test_times, test_event_types, test_object_types, test_locations = parse_summary_file(path)
    summary_type, summary_times, summary_event_types, summary_object_types, summary_locations = parse_summary_file(
        summary_file_path)

    previous_context_color = None
    context_color_order_mapping = dict()
    context_order_counter = 0
    # Get from the summary file the context_color->context_order mapping via the order of item clicks
    for t, event_type, object_type, loc in zip(summary_times, summary_event_types, summary_object_types,
                                               summary_locations):
        index, current_context_color = nav_get_room_by_location(get_location_by_name(object_type, test2d=False))
        if not previous_context_color:
            previous_context_color = current_context_color
            context_color_order_mapping[current_context_color] = context_order_counter
        if not (previous_context_color == current_context_color):
            context_order_counter += 1
            if not (current_context_color in context_color_order_mapping):
                context_color_order_mapping[current_context_color] = context_order_counter
        previous_context_color = current_context_color

    already_processed_item_ids = []

    out_rows = []

    # Iterate through the test file
    count = 0
    for t, event_type, object_type, loc in zip(test_times, test_event_types, test_object_types, test_locations):
        # If the object has already been processed, skip it (we're going in order down the file, but we're processing
        # the entirety of each object as we come to it
        if object_type in already_processed_item_ids:
            count += 1
            continue
        placed_state = True
        last_placed_values = (t, event_type, object_type, loc)
        replacements = 0
        count += 1
        # Iterate through all lines after the current one and find the number of replacements and final position value
        for i in range(count, len(test_times)):
            if test_object_types[i] == object_type:
                placed_state = not placed_state
                last_placed_values = (test_times[i], test_event_types[i], test_object_types[i], test_locations[i])
                replacements += 0.5
        # Add the object to the already processed list so we don't process it again
        already_processed_item_ids.append(object_type)

        # Get the object type and use it to determine (looking at the summary file) in what order the object was viewed
        # during study time
        order_clicked_study = None
        for i, study_object in zip(range(0, len(summary_object_types)), summary_object_types):
            if study_object == object_type:
                order_clicked_study = i
                break

        # Using the object label, determine what color room the object should have been in
        expected_room_by_color = None
        object_label_index = study_labels.index(object_type)
        for i, l in zip(range(0, len(context_item_indicies)), context_item_indicies):
            if object_label_index in l:
                expected_room_by_color = context_labels[i]
                break

        if len(last_placed_values[3]) == 2:
            x_placed = loc[0]
            y_placed = loc[1]
        elif len(last_placed_values[3]) == 3:
            x_placed = loc[0]
            y_placed = loc[2]
        else:
            raise LogParseError("The test vr file had a problem parsing. A location had neither 2 or 3 points.")

        # Using the object location, get the color of the room the object was actually placed in
        index, room = nav_get_room_by_location((x_placed, y_placed))
        actual_room_by_color = room

        # Using the name of the object, find the expected location
        x_expected, y_expected = get_location_by_name(object_type, test2d=False)

        # Using the room colors and the previously generated color->order mapping, get the expected/actual room order
        try:
            expected_room_by_order = context_color_order_mapping[expected_room_by_color]
        except KeyError:
            expected_room_by_order = -1  # If the expectation isn't in the map, return -1
        try:
            actual_room_by_order = context_color_order_mapping[actual_room_by_color]
        except KeyError:
            actual_room_by_order = -1  # If the expectation isn't in the map, return -1

        # Generate and append the output line
        out_rows.append(
            [subject_id, trial_number, object_type, x_placed, y_placed, x_expected, y_expected, order_clicked_study,
             expected_room_by_order, expected_room_by_color, actual_room_by_order, actual_room_by_color,
             replacements, last_placed_values[0]])

    return out_rows


def close_writer(file_pointer):
    """
    Helper function for closing a csv writer (this, in combination with the open and parse functions, prevents the
    external application from needing an import csv reference.

    :param file_pointer: a file pointer to be closer
    :return: Nothing
    """
    try:
        file_pointer.close()
    except AttributeError:
        return


def calculate_euler_vector_from_quaternion(q0, q1, q2, q3):
    """
    This function takes in a quaternion and returns an x,y,z euler vector.

    :param q0: the first quaternion parameter
    :param q1: the second quaternion parameter
    :param q2: the third quaternion parameter
    :param q3: the fourth quaternion parameter

    :return: a tuple containing the x,y,z euler angles
    """
    x = math.atan2(2 * ((q0 * q1) + (q2 + q3)), 1 - (2 * ((q1 * q1) + (q2 * q2))))
    # noinspection PyTypeChecker
    y = math.asin(numpy.clip(float(2 * ((q0 * q2) - (q3 * q1))), float(-1), float(1)))
    z = math.atan2(2 * ((q0 * q3) + (q1 * q2)), 1 - (2 * ((q2 * q2) + (q3 * q3))))
    return x, y, z


def point_is_in_rectangle(point, rectangle):
    """
    This helper function will return true if the point given is within the rectangle given (point is (x,y), rectangle
    is (dictionary 'x', 'y', 'w', 'h').

    :param point: a tuple or list containing an (x,y) point
    :param rectangle: a dictionary containing 'x', 'y', 'w', and 'h' keys associated with the x,y top left corner of
                      a rectangle and the width and height of the rectangle

    :return: True if the point is within the rectangle, False otherwise
    """
    return (rectangle['x'] < point[0] < (rectangle['x'] + rectangle['w']) and
            rectangle['y'] < point[1] < (rectangle['y'] + rectangle['h']))


def test2d_get_room_by_location(location):  # Accepts (x,y)
    """
    This helper function will get the room in the 2d test image space in which an (x,y) point is contained.

    :param location: a tuple containing the x and y coordinates from the 2d test
    :return: a tuple containing a room index (in test_context_boundaries) and a context label (from context_labels)
    """
    index = -1
    for i, context_boundary in zip(range(0, len(test_context_boundries)), test_context_boundries):
        if point_is_in_rectangle(location, context_boundary):
            index = i
            break
    return index, context_labels[index]


def nav_get_room_by_location(location):
    """
    This helper function will get the room in the vr navigation environment in which an (x,y) point is contained.

    Accepts (x,y) coordinates (aka (x,z) in Unity space).

    :param location: a tuple containing the (x, y) coordinate whose room should be identified

    :return: a tuple containing a room index (in study_context_boundaries) and a context label (from context_labels)
    """
    index = -1
    for i, context_boundary in zip(range(0, len(study_context_boundries)), study_context_boundries):
        if point_is_in_rectangle(location, context_boundary):
            index = i
            break
    return index, context_labels[index]


def get_location_by_name(name, test2d=True):
    """
    This helper function will get the expected location in either test (2d test) or navigation space.

    :param name: the name of the item
    :param test2d: if True, the 2D test items are used

    :return: the correct location of the item
    """
    name = name.strip()
    if test2d:
        index = test_labels.index(name)
        return test_realX[index], test_realY[index]
    else:
        index = study_labels.index(name)
        return study_realX[index], study_realY[index]
