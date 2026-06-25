# Standard imports
from datetime import datetime

# Third-party imports
import numpy as np

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.forces.drag import DragForce, DragSensitive, IsotropicDrag
from org.orekit.forces.gravity import (
    HolmesFeatherstoneAttractionModel,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.radiation import (
    SolarRadiationPressure,
    IsotropicRadiationSingleCoefficient,
    RadiationSensitive,
)
from org.orekit.models.earth.atmosphere import NRLMSISE00, NRLMSISE00InputParameters
from org.orekit.orbits import CartesianOrbit, PositionAngleType
from org.orekit.propagation import SpacecraftState
from org.orekit.propagation.conversion import (
    NumericalPropagatorBuilder,
    DormandPrince853IntegratorBuilder,
)
from org.orekit.utils import TimeStampedPVCoordinates
from org.hipparchus.geometry.euclidean.threed import Vector3D


# Internal imports
from .parameters import NumericalPropagatorParameters
from brent import Constants
from brent.propagators import WrappedPropagator


class OrekitNumericalPropagator(WrappedPropagator):

    def __init__(
        self,
        date: datetime,
        state: np.ndarray,
        model: NumericalPropagatorParameters,
    ):
        # Create propagator builder
        propagatorBuilder = OrekitNumericalPropagator.builder(date, state, model)

        # Calculate number of parameters
        n = len(propagatorBuilder.getSelectedNormalizedParameters())

        # Create propagator
        propagator = propagatorBuilder.buildPropagator(n * [1.0])

        # Convert date and state to Orekit format
        date_ = datetime_to_absolutedate(date)
        pos_ = Vector3D(*state[0:3].tolist())
        vel_ = Vector3D(*state[3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)
        spacecraftState = SpacecraftState(
            CartesianOrbit(
                state_,
                Constants.DEFAULT_ECI,
                Constants.DEFAULT_MU,
            )
        )

        # Ensure initial state is correct
        propagator.setInitialState(spacecraftState)

        # Initialise wrapped propagator
        super().__init__(propagator)

    @staticmethod
    def __builder(
        state: TimeStampedPVCoordinates,
        model: NumericalPropagatorParameters,
    ):
        # Create propagator builder
        propagatorBuilder = NumericalPropagatorBuilder(
            CartesianOrbit(state, Constants.DEFAULT_ECI, Constants.DEFAULT_MU),
            DormandPrince853IntegratorBuilder(0.1, 300.0, model.tol),
            PositionAngleType.MEAN,
            1.0,
        )

        # Set object mass
        propagatorBuilder.setMass(model.mass)

        # Create Earth body
        earthBody = OneAxisEllipsoid(
            Constants.DEFAULT_RADIUS,
            Constants.DEFAULT_FLATTENING,
            Constants.DEFAULT_ECEF,
        )

        # Get celestial bodies
        sun = CelestialBodyFactory.getSun()
        moon = CelestialBodyFactory.getMoon()

        # Create gravity model
        if model.potential:
            # Load geopotential model
            gravityFieldProvider = GravityFieldFactory.getNormalizedProvider(
                model.potential_degree,
                model.potential_order,
            )

            # Create geopotential force model
            gravityForceModel = HolmesFeatherstoneAttractionModel(
                Constants.DEFAULT_ECEF,
                gravityFieldProvider,
            )

            # Add geopotential force model to propagator
            propagatorBuilder.addForceModel(gravityForceModel)

        # Moon model
        if model.moon:
            # Create lunar perturbation
            moonAttraction = ThirdBodyAttraction(moon)

            # Add lunar perturbation to propagator
            propagatorBuilder.addForceModel(moonAttraction)

        # Sun model
        if model.sun:
            # Create solar perturbation
            sunAttraction = ThirdBodyAttraction(sun)

            # Add solar perturbation to propagator
            propagatorBuilder.addForceModel(sunAttraction)

        # SRP model
        if model.srp:
            # Create SRP force
            srp = SolarRadiationPressure(
                sun,
                earthBody,
                IsotropicRadiationSingleCoefficient(model.area_srp, model.cr),
            )

            # Enable CR estimation
            if model.srp_estimate:
                # Iterate through drivers
                for driver in srp.getParametersDrivers():
                    # Enable reflection coefficent driver
                    if driver.getName() == RadiationSensitive.REFLECTION_COEFFICIENT:
                        driver.setSelected(True)

            # Add SRP to force model
            propagatorBuilder.addForceModel(srp)

        # Drag model
        if model.drag:
            # Create drag force
            drag = DragForce(
                NRLMSISE00(NRLMSISE00InputParameters(), sun, earthBody),
                IsotropicDrag(model.area_drag, model.cd),
            )

            # Enable CD estimation
            if model.drag_estimate:
                # Iterate through drivers
                for driver in drag.getParametersDrivers():
                    # Enable drag driver
                    if driver.getName() == DragSensitive.DRAG_COEFFICIENT:
                        driver.setSelected(True)

            # Add drag to force model
            propagatorBuilder.addForceModel(drag)

        # Return propagator builder
        return propagatorBuilder

    @staticmethod
    def builder(
        date: datetime,
        state: np.ndarray,
        model: NumericalPropagatorParameters,
    ):
        # Convert date and state to Orekit format
        date_ = datetime_to_absolutedate(date)
        pos_ = Vector3D(*state[0:3].tolist())
        vel_ = Vector3D(*state[3:6].tolist())
        state_ = TimeStampedPVCoordinates(date_, pos_, vel_)

        # Return default propagator builder
        return OrekitNumericalPropagator.__builder(state_, model)
