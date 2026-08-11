# Standard import
from argparse import ArgumentParser, Namespace

# Internal imports
from apps.application import Application, ApplicationFactory


# ---------------------------------------------------------------------------
# Application registration
# ---------------------------------------------------------------------------
# In this codebase, every "app" is a class that:
#   1. Inherits from Application (an abstract base class).
#   2. Is decorated with @ApplicationFactory.register(...), which adds it to a
#      global dictionary keyed by a short name (here "tle_fit").
#   3. Implements two class/static methods:
#        addArguments — declares the command-line flags for this app.
#        run          — is called by main.py after argument parsing.
#
# When the user runs:
#     python main.py tle_fit -i input/tle_fit.json -o output/
# main.py picks "tle_fit" from the registry, instantiates TLEFit, and calls run().
# ---------------------------------------------------------------------------

@ApplicationFactory.register("tle_fit", "Batch least-squares TLE fit using Thalassa")
class TLEFit(Application):

    @staticmethod
    def run(arguments: Namespace) -> None:
        # We import the heavy logic here (not at module level) so that the
        # import of orekit / pythalassa only happens when this app is actually run.
        from ._tle_fit import main

        # Delegate to the imperative main() function in _tle_fit.py
        main(arguments.input, arguments.output_dir)

    @classmethod
    def addArguments(cls, parser: ArgumentParser) -> None:
        parser.add_argument(
            "-i",
            "--input",
            type=str,
            default="./input/tle_fit.json",
            help="Path to the JSON config file",
        )
        parser.add_argument(
            "-o",
            "--output_dir",
            type=str,
            default="./output/",
            help="Directory in which to save results",
        )
