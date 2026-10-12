# Future imports
from __future__ import annotations

# Standard imports
from datetime import datetime, timedelta
from enum import Enum
from glob import glob
from functools import lru_cache

# Third-party imports
import numpy as np
import pandas as pd

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate, absolutedate_to_datetime
from org.orekit.propagation.analytical.tle import TLE

# Internal imports
from brent import Constants
from brent.propagators import Propagator
from brent.propagators.vallado_sgp4 import ValladoSGP4Propagator


class TLEPropagatorMethod(Enum):
    FORWARD = 1
    BACKWARD = 2
    BLENDED = 3


class TLEPropagator(Propagator):

    def __init__(
        self,
        tle: TLE | list[TLE],
        method: TLEPropagatorMethod = TLEPropagatorMethod.FORWARD,
        expiry: None | timedelta | np.timedelta64 = None,
    ):
        # Ensure list of TLEs
        tles = tle if not isinstance(tle, TLE) else [tle]

        # Check for number of TLEs
        if len(tles) < 1:
            raise ValueError("Insufficent number of TLEs")

        # Generate epochs and propagators
        epochs = [absolutedate_to_datetime(itle.getDate()) for itle in tles]
        propagators = [ValladoSGP4Propagator(itle) for itle in tles]

        # Sort propagators
        indices = sorted(range(len(epochs)), key=lambda k: epochs[k])

        # Store sorted epochs and propagators
        self.epochs = np.array([epochs[idx] for idx in indices], dtype=np.datetime64)
        self.propagators = [propagators[idx] for idx in indices]

        # Store propagation method
        if method == TLEPropagatorMethod.FORWARD:
            self._propagate_method = self._propagate_forward
        elif method == TLEPropagatorMethod.BACKWARD:
            self._propagate_method = self._propagate_backward
        elif method == TLEPropagatorMethod.BLENDED:
            self._propagate_method = self._propagate_blended
        else:
            raise RuntimeError("Unknown propagation method")

        # Store expiry
        self.expiry = expiry

    def _is_expired(self, t1, t2) -> bool:
        # Return valid if expiry not set
        if self.expiry is None:
            return False

        # Return expired if difference between dates is larger than TLE expiry
        return np.abs(t1 - t2) > self.expiry

    def _propagate_forward(self, date, frame=Constants.DEFAULT_ECI):
        # Find preceeding index
        indices = self.epochs > date
        idx = np.argmax(indices) - 1
        idx = np.clip(idx, 0, len(self.epochs) - 1)

        # Extract epoch
        epoch = self.epochs[idx]

        # Check if TLE has expired
        if self._is_expired(epoch, date):
            raise RuntimeError("TLE has expired")

        # Extract propagator
        propagator = self.propagators[idx]

        # Return propagated state
        return propagator._propagate(date, frame)

    def _propagate_backward(self, date, frame=Constants.DEFAULT_ECI):
        # Find following index
        indices = self.epochs > date
        idx = np.argmax(indices)
        idx = np.clip(idx, 0, len(self.epochs) - 1)

        # Extract epoch
        epoch = self.epochs[idx]

        # Check if TLE has expired
        if self._is_expired(epoch, date):
            raise RuntimeError("TLE has expired")

        # Extract propagator
        propagator = self.propagators[idx]

        # Return propagated state
        return propagator._propagate(date, frame)

    def _propagate_blended(self, date, frame=Constants.DEFAULT_ECI):
        # Find preceeding index
        indices = self.epochs > date
        idx = np.argmax(indices) - 1

        # Extract epochs
        t1 = self.epochs[idx]
        t2 = self.epochs[idx + 1]

        # Check if TLEs have expired
        if self._is_expired(t1, date) or self._is_expired(t2, date):
            raise RuntimeError("TLE has expired")

        # Calculate time deltas in seconds
        dt = (date - t1) / np.timedelta64(1, "s")
        dt12 = (t2 - t1) / np.timedelta64(1, "s")

        # Calculate weighting
        w = 0.5 + 0.5 * np.cos(np.pi * dt / dt12)
        wp = -0.5 * np.pi / dt12 * np.sin(np.pi * dt / dt12)

        # Extract propagators
        propagator1 = self.propagators[idx]
        propagator2 = self.propagators[idx + 1]

        # Calculate states
        state1 = propagator1._propagate(date, frame)
        state2 = propagator2._propagate(date, frame)

        # Extract positions and velocities
        r1, v1 = state1[0:3], state1[3:6]
        r2, v2 = state2[0:3], state2[3:6]

        # Blend states
        r = w * r1 + (1 - w) * r2
        v = w * v1 + (1 - w) * v2 + wp * (r1 - r2)

        # Return blended state
        return np.concat((r, v))

    def _propagate(self, date, frame=Constants.DEFAULT_ECI):
        # Check if date is outside epoch bounds
        if date <= self.epochs[0]:
            # Return state propagated with first propagator
            return self._propagate_backward(date, frame)
        elif date >= self.epochs[-1]:
            # Return state propagated with last propagator
            return self._propagate_forward(date, frame)

        # Return state propagated with method
        return self._propagate_method(date, frame)

    @lru_cache(maxsize=256)
    @staticmethod
    def _parse(path: str) -> pd.DataFrame:
        # Return parsed TLE file
        return pd.read_json(path)

    @classmethod
    def _load(
        cls,
        path: str,
        start: datetime = datetime.min,
        end: datetime = datetime.max,
        norad: int | None = None,
        cospar: str | None = None,
    ) -> list[TLE]:
        # Read TLEs
        paths = sorted(glob(path, recursive=True))
        tles = pd.concat([cls._parse(path) for path in paths], ignore_index=True)

        # Filter TLEs
        if norad is not None:
            tles = tles[tles["NORAD_CAT_ID"] == norad]
        if cospar is not None:
            tles = tles[tles["OBJECT_ID"] == cospar]

        # Throw error if file contains zero or multiple different objects
        nunique = tles["OBJECT_ID"].nunique()
        if nunique != 1:
            raise ValueError(f"{nunique} object identifiers")

        # Sort by epoch and creation date
        tles.sort_values(["EPOCH", "CREATION_DATE"], inplace=True)

        # Drop duplicate TLEs, keeping the most recently issued instances
        # TODO: check for TLEs in close temporal proximity
        tles.drop_duplicates("EPOCH", keep="last", inplace=True)

        # Generate TLEs
        tles = [
            TLE(tle["TLE_LINE1"], tle["TLE_LINE2"])
            for tle in tles.to_dict(orient="records")
        ]

        # Extract TLE subset
        # TODO: move date filtering to DataFrame?
        tles = [
            tle
            for tle in tles
            if (tle.getDate().durationFrom(datetime_to_absolutedate(start)) >= 0)
            and (tle.getDate().durationFrom(datetime_to_absolutedate(end)) <= 0)
        ]

        # Return filtered TLEs
        return tles

    @classmethod
    def load(
        cls,
        path: str,
        start: datetime = datetime.min,
        end: datetime = datetime.max,
        norad: int | None = None,
        cospar: str | None = None,
        *args,
        **kwargs,
    ) -> TLEPropagator:
        # Load TLEs
        tles = cls._load(path, start, end, norad, cospar)

        # Return TLE propagator
        return TLEPropagator(tles, *args, **kwargs)
