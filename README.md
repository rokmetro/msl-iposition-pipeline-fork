# msl-iposition-pipeline
Memory Systems Lab iPosition Data Pipeline

This data pipeline is meant for the processing of iPosition data. It will output all the pre-2017 metrics as well as the newer metrics.

# Install

## Windows

Download the repository via git or the zip file here: [Download](https://github.com/kevroy314/msl-iposition-pipeline/archive/master.zip).

Run First Time Setup (Windows).bat

To run, run either:
Run Batch (Collapsed Trials).bat
Run Batch (Non-Collapsed Trials).bat

Select a folder then hit OK.

## MAC

Run FirstTimeSetupMAC

In the therminal in the main directory, run either:

python batch_pipeline.py --collapse_trials=1

python batch_pipeline.py --collapse_trials=0
