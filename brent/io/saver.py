# Standard imports
from datetime import datetime, timezone
from enum import Enum
import os
from typing import List

# Third-party imports
import pandas as pd
from tqdm import tqdm

# Internal imports
import brent.util


class Saver:

    def __init__(
        self,
        root: str,
        save_period: int = 50,
        retry_limit: int = 5,
        name_suffix: str = "",
    ) -> None:
        # Declare state
        self.state = SaverState.DEINITIALISED

        # Loop until directory initialised
        ntries = 0
        while self.state != SaverState.INITIALISED:
            # Save start time and name
            self.time = datetime.now(timezone.utc)
            self.name = self.time.strftime("%Y%m%d_%H%M%S") + name_suffix

            # Create directory
            initialised = self._create_directory(root, self.name)

            # Set state
            if initialised:
                self.state = SaverState.INITIALISED

            # Increment number of tries
            ntries += 1

            # Throw error if not initialised
            if ntries > retry_limit:
                raise ValueError("Failed to initialise")

        # Store save period
        self.save_period = save_period

        # Initialise state
        self.niter = 0
        self._increment_state()

        # Results
        self.results = []

        # Save initial metadata
        self.save_metadata()

        # Set state to running
        self.state = SaverState.RUNNING

    def update(self, result: dict) -> None:
        # Throw error if saver not running
        if self.state != SaverState.RUNNING:
            raise RuntimeError("Saver not running")

        # Append result to results
        self.results.append(result)

        # Trigger auto-save
        if len(self.results) % self.save_period == 0:
            self.save()

    def save(self, final: bool = False) -> None:
        # Return if already in saving state
        if (self.state == SaverState.FINAL_SAVE) or (self.state == SaverState.FINISHED):
            return

        # Set saving state
        if final:
            self.state = SaverState.FINAL_SAVE

        # Set filename suffix
        suffix = "" if final else f"_{self.checkpoint}"

        # Generate results filepath
        fname = os.path.join(self.directory, self.name + suffix + ".pkl")

        # Convert results to DataFrame
        df = pd.DataFrame(self.results)

        # Save results
        try:
            # Try to save the results
            with open(fname, "wb") as fp:
                df.to_pickle(fp)

            # Update state
            if not final:
                self._increment_state()
        except:
            # Print error message
            tqdm.write(f"Error saving file at: {fname}")

        # Save metadata
        self.save_metadata()

        # Set state to finished
        if self.state == SaverState.FINAL_SAVE:
            self.state = SaverState.FINISHED

    def save_metadata(self) -> None:
        # Generate metadata filepath
        fname = os.path.join(self.directory, self.name + "_metadata.txt")

        # Declare lines list
        lines: List[str] = []

        # Check run state
        if self.state == SaverState.INITIALISED:
            # Start metadata
            lines = self._initial_metadata()
        elif self.state == SaverState.RUNNING:
            # TODO: save iteration progress?
            pass
        elif self.state == SaverState.FINAL_SAVE:
            # Final metadata
            lines = self._final_metdata()
        else:
            # Raise error
            raise RuntimeError("Incompatible run state")

        # Ensure lines have linebreak
        lines = [line + "\n" for line in lines]

        # Save lines to metdata file
        # TODO: error handling?
        with open(fname, "a") as fp:
            fp.writelines(lines)

    def _initial_metadata(self) -> List[str]:
        # Declare lines list
        lines: List[str] = []

        # Git commit
        git_hash = brent.util.get_commit()
        lines.append(f"Git commit: {git_hash}")

        # Start time
        time = self.time
        lines.append(f"Start time: {time.isoformat()}")

        # Return lines
        return lines

    def _final_metdata(self) -> List[str]:
        # Declare lines list
        lines: List[str] = []

        # End time
        time = datetime.now(timezone.utc)
        lines.append(f"End time:   {time.isoformat()}")

        # Execution time
        exectime = (time - self.time).total_seconds()
        lines.append(f"Exec. time: {exectime:.2f} s")

        # Number of fits
        nfits = len(self.results)
        lines.append(f"No. fits:   {nfits}")

        # Fit period
        fitperiod = exectime / nfits
        lines.append(f"Fit rate:   {fitperiod:.2f} s/fit")

        # Return lines
        return lines

    def _create_directory(self, root: str, name: str) -> bool:
        # Return if already initialised
        if self.state != SaverState.DEINITIALISED:
            return False

        try:
            # Generate output directory
            directory = os.path.abspath(os.path.join(root, name))

            # Throw error if the directory already exists
            os.makedirs(directory, exist_ok=False)

            # Store directory
            self.directory = directory
            self.initialised = True

            # Return success
            return True
        except Exception as e:
            # Print exemption
            tqdm.write(str(e))

            # Return failure
            return False

    def save_input(self, input: str) -> None:
        # Generate output path
        output = os.path.join(self.directory, f"{self.name}_input.json")

        # Copy the input file to the output path
        with open(input, "rb") as src, open(output, "wb") as dest:
            dest.write(src.read())

    def _increment_state(self) -> None:
        # Increment number iteration number
        self.niter += 1

        # Update checkpoint state
        self.checkpoint = "checkpoint_b" if self.niter % 2 == 0 else "checkpoint_a"


class SaverState(Enum):
    DEINITIALISED = 1
    INITIALISED = 2
    RUNNING = 3
    FINAL_SAVE = 4
    FINISHED = 5
