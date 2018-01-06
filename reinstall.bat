pip uninstall msl-iposition-pipeline --yes
pip install . --no-deps
@echo off
python -c "import cogrecon; print(cogrecon.__version__)"