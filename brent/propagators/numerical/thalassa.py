# Standard imports
from datetime import datetime, timedelta
import multiprocessing as mp
import os
import queue

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from org.orekit.frames import FramesFactory

# External imports
import pythalassa

# Internal imports
from .parameters import NumericalPropagatorParameters
from brent import Constants
import brent.paths
from brent.propagators import Propagator


def date_to_MJD(date: datetime | np.datetime64) -> float:
    # Ensure correct date type
    if not isinstance(date, datetime):
        date = pd.to_datetime(date)

    # Calculate delta from 1st January 2000
    delta_J2000 = date - datetime(2000, 1, 1, 0, 0, 0, 0)

    # Convert to float
    delta_J2000_days = delta_J2000 / timedelta(days=1)

    # Add MJD offset
    mjd = delta_J2000_days + 51544.0

    # Return MJD
    return mjd


dates_to_MJD = np.vectorize(date_to_MJD)


class ThalassaProcess(mp.Process):

    def __init__(
        self,
        dates,
        state,
        parameters: NumericalPropagatorParameters,
        *args,
        **kwargs,
    ):
        # Initialise parent class
        super().__init__(daemon=True, *args, **kwargs)

        # Store dates and state
        self.dates = dates
        self.state = state

        # Store parameters
        self.parameters = parameters

        # Results queue
        self.results = mp.Queue(maxsize=1)

    def run(self):
        # Get THALASSA settings
        model = self.getModel(self.parameters)
        paths = self.getPaths()
        settings = self.getSettings(self.parameters)
        spacecraft = self.getSpacecraft(self.parameters)

        # Change working directory
        # NOTE: this is to ensure that SPICE kernels are loaded from the data directory
        os.chdir(brent.paths.ROOT_DIR)

        # Create propagator
        propagator = pythalassa.Propagator(model, paths, settings, spacecraft)

        # Convert state vector to [km] and [km/s]
        state_ = self.state.ravel() / 1000.0

        # Convert dates to MJD floats
        dates_ = dates_to_MJD(self.dates)

        # Propagate
        states_: np.ndarray = propagator.propagate(dates_, state_)

        # Transpose to row vectors and convert to [m] and [m/s]
        states = states_.T * 1000.0

        # Put states onto results queue
        self.results.put(states)

    def propagate(self, timeout: float | None = None) -> np.ndarray:
        # TODO: ensure only called in main process?
        # Try to propagate
        try:
            # Start propagation
            self.start()

            # Wait for results
            states = self.results.get(timeout=timeout)
        except queue.Empty:
            # Raise error due to lack of results
            # TODO: switch to returning NaNs?
            raise RuntimeError(f"Propagation timed out after {timeout} seconds")
        finally:
            # Terminate process
            self.kill()

        # Return propagated states
        return states

    @classmethod
    def getModel(cls, parameters: NumericalPropagatorParameters) -> pythalassa.Model:
        # Declare model
        model = pythalassa.Model()

        # Enable/disable geopotential model
        if parameters.potential:
            model.insgrav = pythalassa.NONSPHERICAL
            model.gdeg = parameters.potential_degree
            model.gord = parameters.potential_order
        else:
            model.insgrav = pythalassa.SPHERICAL
            model.gdeg = 0
            model.gord = 0

        # Set ephemeris source
        model.iephem = pythalassa.EPHEM_DE431

        # Enable/disable Sun
        if parameters.sun:
            model.isun = pythalassa.SUN_ENABLED
        else:
            model.isun = pythalassa.SUN_DISABLED

        # Enable/disable Moon
        if parameters.moon:
            model.imoon = pythalassa.MOON_ENABLED
        else:
            model.imoon = pythalassa.MOON_DISABLED

        # Enable/disable drag model
        if parameters.drag:
            model.idrag = pythalassa.DRAG_NRLMSISE00
        else:
            model.idrag = pythalassa.DRAG_DISABLED

        # Set Solar flux model
        model.iF107 = pythalassa.FLUX_VARIABLE

        # Enable/disable SRP model
        if parameters.srp:
            model.iSRP = pythalassa.SRP_ENABLED_CONICAL
        else:
            model.iSRP = pythalassa.SRP_DISABLED

        # Return model
        return model

    @classmethod
    def getPaths(cls) -> pythalassa.Paths:
        # Declare paths
        paths = pythalassa.Paths()
        paths.phys_path = brent.paths.THALASSA_DATA_PHYSICAL
        paths.earth_path = brent.paths.THALASSA_DATA_EARTH
        paths.kernel_path = brent.paths.THALASSA_DATA_KERNEL
        paths.eop_path = brent.paths.THALASSA_DATA_EOP

        # Return paths
        return paths

    @classmethod
    def getSettings(
        cls,
        parameters: NumericalPropagatorParameters,
    ) -> pythalassa.Settings:
        # Declare settings
        # TODO: set
        settings = pythalassa.Settings()
        settings.eqs = pythalassa.EDROMO_C  # TODO: add options for different methods?
        settings.tol = parameters.tol

        # Return settings
        return settings

    @classmethod
    def getSpacecraft(
        cls,
        parameters: NumericalPropagatorParameters,
    ) -> pythalassa.Spacecraft:
        # Declare spacecraft
        spacecraft = pythalassa.Spacecraft()

        # Set mass
        spacecraft.mass = parameters.mass

        # Set drag properties
        spacecraft.area_drag = parameters.area_drag
        spacecraft.cd = parameters.cd

        # Set SRP properties
        spacecraft.area_srp = parameters.area_srp
        spacecraft.cr = parameters.cr

        # Return spacecraft
        return spacecraft


class ThalassaNumericalPropagator(Propagator):

    def __init__(
        self,
        date: datetime,
        state: np.ndarray,
        parameters: NumericalPropagatorParameters,
    ) -> None:
        # Store initial date and state
        self.date = date
        self.state = state

        # Store parameters
        self.parameters = parameters

    def _propagate(self, date, frame=Constants.DEFAULT_ECI) -> np.ndarray:
        raise RuntimeError("This method should not be called at any time")

    def propagate(
        self,
        dates,
        frame=Constants.DEFAULT_ECI,
        timeout: float | None = 120.0,
    ):
        # NOTE: the propagator is executed as a subprocess to avoid issues encountered when
        #       creating and destroying large numbers of propagators using SPICE kernels

        # Check requested frame is compatible with THALASSA
        if frame != FramesFactory.getGCRF():
            # TODO: check correct frame
            raise ValueError("THALASSA only supports GCRF")

        # Ensure dates increase/decrease monotonically
        diff = np.diff(dates)
        zero = np.timedelta64(0, "D")
        monotonic = np.all(diff >= zero) or np.all(diff <= zero)
        if not monotonic:
            raise ValueError("Dates must be monotonically increasing or decreasing")

        # Check if first date matches stored initial date
        if dates[0] != self.date:
            # Propagate state to initial date
            # TODO: need to test that this is behaving properly
            state = self.propagate(np.array([self.date, dates[0]]))[1, :]
        else:
            # Use stored initial state
            state = self.state

        # Create propagation process
        process = ThalassaProcess(dates, state, self.parameters)

        # Propagate
        states = process.propagate(timeout)

        # Return states
        return states
