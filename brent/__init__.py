# TODO: docstrings
# TODO: type hinting
# TODO: standardise datetime/timedelta type (probably NumPy)
# TODO: fix bug where THALASSA breaks following the use of Matplotlib

# Standard imports
import os
import sys

# Load paths
from . import paths

# Orekit imports
import orekit
from orekit.pyhelpers import setup_orekit_curdir

# Initialise Orekit. Suppress macOS OpenJDK stack-guard warnings emitted at VM startup.
stderr_fd = os.dup(sys.stderr.fileno())
try:
	with open(os.devnull, "w") as null:
		os.dup2(null.fileno(), sys.stderr.fileno())
		vm = orekit.initVM()
finally:
	os.dup2(stderr_fd, sys.stderr.fileno())
	os.close(stderr_fd)
setup_orekit_curdir(paths.DATA_OREKIT_DIR)

# Add PyTHALASSA to path
sys.path.append(paths.THALASSA_LIB_DIR)

# Load constants
from .constants import Constants

# Internal imports
from . import bias
from . import covariance
from . import filter
from . import frames
from . import io
from . import propagators
from . import util
from . import skyfield
