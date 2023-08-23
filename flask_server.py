#!/usr/bin/env python
# encoding: utf-8
from cogrecon.core.full_pipeline import full_pipeline
from cogrecon.core.data_structures import TrialData, ParticipantData, AnalysisConfiguration
from cogrecon.core.tools import generate_random_test_points
import math

import json, http
from flask import Flask, jsonify, request, make_response
app = Flask(__name__)
# expected json format from POST
# {
#     "trial_objects": [
#       {
#         "image_key": "trial-object-1",
#         "position": {
#           "x": -26.214187056929976,
#           "y": 53.626855920862305
#         },
#         "goal_position": {
#           "x": -31,
#           "y": 27
#         }
#       },
#       {
#         "image_key": "trial-object-2",
#         "position": {
#           "x": 45.84311026114002,
#           "y": 38.339301215277736
#         },
#         "goal_position": {
#           "x": 51,
#           "y": 45
#         }
#       }
#     ]
# }


@app.route('/', methods=['POST'])
def analysis():
    """
    this function checks if the POST body contains correct json input and invoke the 
    analysis pipline to return desired analysis result.
    """
    try:
        request_data = request.get_json()
        if not request_data:
            return jsonify({'response': 'empty json'}), 400
    except Exception as err:
        print(err)
        return jsonify({'response': 'missing body in request'}), 401
    try:
        trial_objects = request_data['trial_objects']
        if not trial_objects:
            return jsonify({'response': 'trial_objects is empty'}), 401
    except Exception as err:
        print(err)
        return jsonify({'response': 'trial_objects is missing'}), 401
    
    # Ensure the necessary fields are present
    if not all(key in obj for key in ('position', 'goal_position') for obj in trial_objects):
        return jsonify({'response': 'Required field(s) missing in trial_objects'}), 400

    positions, goal_positions = [],[]
    for obj in trial_objects:
        pos, goal_pos = obj['position'], obj['goal_position']
        positions.append([float(pos['x']),float(pos['y'])])
        goal_positions.append([float(goal_pos['x']),float(goal_pos['y'])])

    data = ParticipantData([TrialData(positions, goal_positions)])
    config = AnalysisConfiguration(debug_labels=['test', -1])
    dict_result = full_pipeline(data, config, visualize=False, single_trial=True, return_as_dict=True)    
    return jsonify(dict_result)
    
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
    

