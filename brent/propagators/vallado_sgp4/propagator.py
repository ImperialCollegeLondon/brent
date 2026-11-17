import inspect

import numpy as np
from orekit.pyhelpers import datetime_to_absolutedate
from org.hipparchus.geometry.euclidean.threed import Vector3D
from org.orekit.frames import FramesFactory
from org.orekit.time import TimeScalesFactory
from org.orekit.utils import PVCoordinates

from brent import Constants
from brent.propagators.propagator import Propagator
from .valladopy.astro.sgp4.sgp4 import SGP4
from .valladopy.astro.sgp4.utils import WGSModel
from .valladopy.mathtime.julian_date import jday


class ValladoSGP4Propagator(Propagator):
    """Propagate one Orekit TLE with Vallado SGP4 and return states in GCRF."""

    def __init__(self, tle):
        self.epoch = tle.getDate()
        epoch = self.epoch.getComponents(TimeScalesFactory.getUTC())
        epoch_date, epoch_time = epoch.getDate(), epoch.getTime()

        self.sgp4 = SGP4(wgs_model=WGSModel.WGS_72, use_afspc_mode=True)
        satrec = self.sgp4.satrec
        satrec.ecco = tle.getE()
        satrec.inclo = tle.getI()
        satrec.nodeo = tle.getRaan()
        satrec.argpo = tle.getPerigeeArgument()
        satrec.mo = tle.getMeanAnomaly()
        satrec.no_kozai = 60.0 * tle.getMeanMotion()
        satrec.bstar = tle.getBStar()
        satrec.jdsatepoch, satrec.jdsatepochf = jday(
            epoch_date.getYear(),
            epoch_date.getMonth(),
            epoch_date.getDay(),
            epoch_time.getHour(),
            epoch_time.getMinute(),
            epoch_time.getSecond(),
        )
        self.sgp4.sgp4init(satrec.jdsatepoch + satrec.jdsatepochf - 2433281.5)

        if self.sgp4.use_deep_space:
            # The upstream caller omits the current angles from this update.
            set_mean_elems = self.sgp4.ds.set_mean_elems

            def set_mean_elems_with_angles(**kwargs):
                elements = inspect.currentframe().f_back.f_locals
                nodem, argpm, mm = elements["nodem"], elements["argpm"], elements["mm"]
                set_mean_elems(
                    em=kwargs["em"],
                    inclm=kwargs["inclm"],
                    nodem=nodem,
                    argpm=argpm,
                    nm=kwargs["nm"],
                    mm=mm,
                )

            self.sgp4.ds.set_mean_elems = set_mean_elems_with_angles

    def _propagate(self, date, frame=Constants.DEFAULT_ECI):
        date = datetime_to_absolutedate(date)
        return self._propagate_absolute(date, frame)

    def _propagate_absolute(self, date, frame=Constants.DEFAULT_ECI):
        r_teme, v_teme = self.sgp4.propagate(date.durationFrom(self.epoch) / 60.0)
        pv_teme = PVCoordinates(
            Vector3D(float(r_teme[0]), float(r_teme[1]), float(r_teme[2])),
            Vector3D(float(v_teme[0]), float(v_teme[1]), float(v_teme[2])),
        )
        pv = FramesFactory.getTEME().getTransformTo(frame, date).transformPVCoordinates(
            pv_teme
        )
        # Vallado returns km and km/s; BRENT states use m and m/s.
        return 1000.0 * np.concatenate(
            (
                np.array(pv.getPosition().toArray()),
                np.array(pv.getVelocity().toArray()),
            )
        )
