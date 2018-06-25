Memory Systems Lab iPosition Data Pipeline
==========================================

This data pipeline is meant for the processing of iPosition data. It will output all the pre-2017 metrics as well as the newer metrics. It has a large number of options which will be enumerated in this readme. It can really run on any spatial/temporal reconstruction data whose outputs are formatted properly (TSV).

Install
-------

Install Git: https://git-scm.com/downloads if you don't already have it.

Install Anaconda Python: https://www.continuum.io/downloads if you don't already have it.

In a command prompt/terminal (you may need to run as administrator), navigate to/create an **empty directory**, then run:

    conda create -n iposition python=2.7 --yes
    
    activate iposition
    
    conda install scipy jupyter scikit-learn pandas --yes
    
    git clone https://github.com/kevroy314/msl-iposition-pipeline/ .
    
    pip install .
    

Updating
--------

To update the script to the latest version navigate to/create an **empty directory**, then run:

    activate iposition
    
    git clone https://github.com/kevroy314/msl-iposition-pipeline/ .
    
    pip install --upgrade .
    

If you'd like to update without changing the dependencies you can instead, from an **empty directory**, run:


    activate iposition
    
    git clone https://github.com/kevroy314/msl-iposition-pipeline/ .
    
    pip install --upgrade . --no-deps
    

Usage
-----

Note: this section is incomplete and will be updated as new features are added.

Although there are many ways to interface with this analysis software, the easiest is to use a Jupyter Notebook in your web browser. To begin, navigate to wherever you downloaded the github repository (from the installation steps), and open a command prompt/terminal window. Then run:

    activate iposition
    
    jupyter notebook
    

A window in your default web browser (preferrably Chrome) will open with a listing of the files and subdirectories in that github repository directory. Click on the 'examples' folder. This folder contains a variety of interactive scripts to perform various functions using the software package. To run a simple analysis on a directory of data, generating an output CSV, click Main.ipynb. This will open a new window in which there are cells containing code as well as additional documentation.

To run the simple analysis, scroll down to Batch Pipeline Test and select the first code cell. Press Shift+Enter and you will see the cell execute (the first cell imports the software). Click Shift+Enter again and wait. A popup asking you to select a folder will appear. Select your data folder and click OK. The data will be processed and a CSV file will be created locally in the 'examples' folder. 

If you want to run with different settings/paramters, see the documentation for the batch_pipeline function here: http://msl-iposition-pipeline.readthedocs.io/en/latest/source/cogrecon.core.html#module-cogrecon.core.batch_pipeline

In particular, note if you have trial_by_trial_accuracy set to True or False (to determine if accuracy is computer within or across trials) and if you need actual_coordinate_prefixes set to True or False (if you have an actual_coordinates.txt file for every participant).

Output
--------

| Column Name | Description |
| ---      | ---       |
| subID | The subject ID (i.e. text before the filename suffix) |
| trial | The trial number |
| Original Misplacement | The misplacement with no correction (i.e. euclidean distance between item placement and studied locations |
| Original Swap | The original swap metric from Watson et. al 2013 |
| Original Edge Resizing | The original edge resizing metric from Watson et. al 2013 |
| Original Edge Distortion | The original edge distortion metric from Watson et. al 2013 |
| Axis Swap Pairs | The pairs of items involved in the Original Swap column (note that if trials are collapsed, this list will contain pairs involved in any trial) |
| Pre-Processed Accurate Placements | The number of items placed within the accuracy circle with an accuracy circle computed based on the raw inputs (i.e. no identity or transformation correction) |
| Pre-Processed Inaccurate Placements | The number of items placed outside of the accuracy circle with an accuracy circle computed based on the raw inputs (i.e. no identity or transformation correction) |
| Pre-Processed Accuracy Threshold | The accuracy circle radius computed based on the raw inputs (i.e. no identity or transformation correction) |
| Deanonymized Accurate Placements | The number of items placed within the accuracy circle with an accuracy circle computed based on the deanonymized (i.e. identity-stripped) inputs (i.e. no transformation correction) |
| Deanonymized Inaccurate Placements | The number of items placed outside of the accuracy circle with an accuracy circle computed based on the deanonymized (i.e. identity-stripped) inputs (i.e. no transformation correction) |
| Deanonymized Accuracy Threshold | The accuracy circle radius computed based on the deanonymized (i.e. identity-stripped) inputs (i.e. no transformation correction) |
| Raw Deanonymized Misplacement | The misplacement after deanonymization (i.e. after identities have been removed) |
| Post-Deanonymized Misplacement | The misplacement after deanonymization (i.e. after identities have been removed) |
| Transformation Auto-Exclusion | True if any items were excluded due to failure to meet the transform accuracy threshold |
| Number of Points Excluded From Geometric Transform | The number of points excluded from the geometric transform due to poor accuracy |
| Rotation Theta | The rotation in degrees detected by the geometric transform (note that it will be nan if the transform fails to improve the misplacement) |
| Scaling | The scaling as a proportion (i.e. 1 means no scaling, 0 to 1 is shrinking, greater than 1 is stretching) detected by the geometric transform (note that it will be nan if the transform fails to improve the misplacement) |
| Translation Magnitude | The translation magnitude (in pixels) detected by the geometric transform (note that it will be nan if the transform fails to improve the misplacement) |
| Translation | A vector representing the translation along each axis found by the geometric transform (note that it will be nan if the transform fails to improve the misplacement) |
| TranslationX | The x component of the translation found by the geometric transform (note that it will be nan if the transform fails to improve the misplacement) |
| TranslationY | The y component of the translation found by the geometric transform (note that it will be nan if the transform fails to improve the misplacement) |
| Geometric Distance Threshold | The threshold used to determine if the transformation function will be applied to a given point (note that points which are sufficiently inaccurate are excluded to avoid the transform being biased by the extreme nature of the outlier) |
| Post-Transform Misplacement | The misplacement after both deanonymization and transformation has been performed on the points |
| Number of Components | The number of associative components found by the deanonymization/assignment process (i.e. the sum of the number of single item placements, swaps, and cycles with no accuracy conditions) |
| Accurate Single-Item Placements | The number of items which were associated with their studied location and within the accuracy circle |
| Inaccurate Single-Item Placements | The number of items which were associated with their studied location and not within the accuracy circle |
| True Swaps | The number of pairs of items which were associated with each other's studied location and within the accuracy circle (i.e. both items were within the accuracy circle) |
| Partial Swaps | The number of pairs of items which were associated with each other's studied location where at least one was not within the accuracy circle (i.e. one or both were not accurately placed, but they were associated with each other's locations) |
| Cycle Swaps | The number of cycles of items (i.e. groups with more than 2 members) which were associated with each other's studied locations where all items were within the accuracy circle |
| Partial Cycle Swaps | The number of cycles of items (i.e. groups with more than 2 members) which were associated with each other's studied locations where at least one item was not within the accuracy circle |
| Misassignment | The total number of items which were associated with another items location, regardless of accuracy (and disregarding any group-wise relations) |
| Accurate Misassignment | The number of items which were associated with another items location and within the accuracy circle |
| Inaccurate Misassignment | The number of items which were associated with another items location and not within the accuracy circle |
| Swap Distance Threshold | The distance threshold for misassignment accuracy |
| True Swap Data Distance | The average distance of true swap items from one another in the participant data |
| True Swap Actual Distance | The average distance of true swap items from one another in the studied data |
| Partial Swap Data Distance | The average distance of partial swap items from one another in the participant data |
| Partial Swap Actual Distance | The average distance of partial swap items from one another in the studied data |
| Cycle Swap Data Distance | The average distance of true cycle items from one another in the participant data |
| Cycle Swap Actual Distance | The average distance of true cycle items from one another in the studied data |
| Partial Cycle Swap Data Distance | The average distance of partial cycle items from one another in the participant data |
| Partial Cycle Swap Actual Distance | The average distance of partial cycle items from one another in the studied data |
| Unique Components | A list of the items in component groups without any duplicates (i.e. if items 0 and 1 were swapped in two separate trials, that pair will only appear once in this list) |
| Contains Category Data | True if the data was processed categorically (i.e. separated according to a categories.txt) |
| Category Label | The label of the category if the data was categorical |
| Accurate Misassignment Pairs | The pairs of items involved in accurate misassignments (note that this is useful if you want to know what items were involved in what associative errors) |
| Inaccurate Misassignment Pairs | The pairs of items involved in inaccurate misassignments (note that this is useful if you want to know what items were involved in what associative errors) |
| num_rows_with_nan| | The number of rows which contain nan data |
