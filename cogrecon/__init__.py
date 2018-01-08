from ._version import __version__

# __all__ = ["core", "misc"]

# Configure the output logger
import coloredlogs
import logging
logging.basicConfig(format="%(levelname)s (%(asctime)s): %(message)s", level=logging.INFO)
coloredlogs.install()
