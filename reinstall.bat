pip uninstall msl-iposition-pipeline --yes
pip install .
@echo off
python -c "import cogrecon; print(cogrecon.__version__)"