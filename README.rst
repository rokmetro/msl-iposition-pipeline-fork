Memory Systems Lab iPosition Data Pipeline
========

This data pipeline is meant for the processing of iPosition data. It will output all the pre-2017 metrics as well as the newer metrics. It has a large number of options which will be enumerated in this readme. It can really run on any spatial/temporal reconstruction data whose outputs are formatted properly (TSV).

Install
--------

Install Git: https://git-scm.com/downloads if you don't already have it.

Install Anaconda Python: https://www.continuum.io/downloads if you don't already have it.

In a command prompt/terminal, navigate to/create an **empty directory**, then run:
::
    conda create -n iposition python=2.7 scipy --yes \
    activate iposition \
    git clone https://github.com/kevroy314/msl-iposition-pipeline/ . \
    pip install .

Updating
--------

To update the script to the latest version navigate to/create an **empty directory**, then run:

    activate iposition \
    git clone https://github.com/kevroy314/msl-iposition-pipeline/ . \
    pip install --upgrade .

If you'd like to update without changing the dependencies you can instead, from an **empty directory**, run:

    activate iposition \
    git clone https://github.com/kevroy314/msl-iposition-pipeline/ . \
    pip install --upgrade . --no-deps

Usage
--------

Note: this section is incomplete and will be updated as new features are added.

Command Line Options
--------

The easiest way to run the program is in batch mode via the command line. Running

    python batch_pipeline.py

runs the program in default mode (with a directory selection dialog popup).

Command Line Arguments
--------

* --search_directory - the root directory in which to search for the actual and data coordinate files (actual_coordinates.txt and ###position_data_coordinates.txt, respectively (will prompt with dialog if left empty)
* --num_trials - the number of trials in each file (will be detected automatically if left empty)
* --num_items - the number of items in each trial (will be detected automatically if left empty)
* --pipeline_mode - the mode in which the pipeline should process (default is 3); 
   * 0 for just accuracy+swaps, 
   * 1 for accuracy+deanonymization+swaps, 
   * 2 for accuracy+global transformations+swaps, 
   * 3 for accuracy+deanonymization+global transformations+swaps
* --accuracy_z_value - the z value to be used for accuracy exclusion (default is 1.96, corresponding to 95% confidence)
* --collapse_trials - if 0, one row per trial will be output, otherwise one row per participant will be output (default is 1)
* --dimension - the dimensionality of the data (default is 2)
* --trial_by_trial_accuracy - when not 0, z_value thresholds are used on a trial-by-trial basis for accuracy calculations, when 0, the thresholds are computed then collapsed across an individual\'s trials (default is 1)
* --prefix_length - the length of the subject ID prefix at the beginning of the data filenames (default is 3)

Advanced usage example
--------

In this example, the "C:\Users Folder\Data" folder and its subfolders will be searched for actual_coordinates.txt and files with length 5 participant IDs followed by position_data_coordinates.txt. Each file will be expected to have 15 trials and 6 items/trial with 3 dimensions each. The accuracy will be computed on a trial by trial basis using a 90% confidence interval. Each trial will be output independently (one per row).

    python batch_pipeline.py --search_directory="C:\User Folder\Data" --num_trials=15 --num_items=6 --accuracy_z_value=1.64 --collapse_trials=0 --dimension=3 --trial_by_trial_accuracy=1 --prefix_length=5

Visualization of Single Trials
--------

Individual trials can be visualized by calling the full_pipeline.py file with appropriate arguments. The required arguments are (in this order):

* actual_coordinates - the path to the file containing the actual coordinates
* data_coordinates - the path to the file containing the data coordinates
* num_trials
* num_items
* line_number

The optional arguments are:

* --pipeline_mode - the mode in which the pipeline should process (default is 3); 
   * 0 for just accuracy+swaps, 
   * 1 for accuracy+deanonymization+swaps, 
   * 2 for accuracy+global transformations+swaps, 
   * 3 for accuracy+deanonymization+global transformations+swaps
* --accuracy_z_value - the z value to be used for accuracy exclusion (default is 1.96, corresponding to 95% confidence)
* --dimension - the dimensionality of the data (default is 2)

Visualization Usage Example
--------

To visualize the second of participant 101's data (assuming 15 trials and 5 items), the command line should be:

    python full_pipeline.py "actual_coordinates.txt" "101position_data_coordinates.txt" 15 5 1

Scripted Usage
--------

Each program can be run from another python script. The easiest way to learn to do this is to look at the examples built into the buttom of each script (below the "# Test code" comment). 
