# Standard imports
from copy import deepcopy
import multiprocessing as mp
from typing import List

# Third-party imports
import numpy as np
import pandas as pd
import scipy.optimize

# Internal imports
from .batchleastsquares import BatchLeastSquares
from brent.constants import Constants
from brent.covariance import Covariance
from brent.frames import Keplerian
from brent.propagators import ThalassaNumericalPropagator, NumericalPropagatorParameters


class ThalassaBatchLeastSquares(BatchLeastSquares):

    def __init__(
        self,
        dates: np.ndarray | pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
        covarianceProvider: Covariance,
    ) -> None:
        # Store observation dates and states
        self.dates = pd.to_datetime(np.copy(dates))
        self.states = np.copy(states)

        # Store model
        self.model = model
        self.srp_estimate = model.srp_estimate
        self.drag_estimate = model.drag_estimate

        # Throw error if SRP estimation enabled without SRP
        # TODO: force enable SRP?
        if model.srp_estimate and not model.srp:
            raise ValueError("Cannot estimate SRP if disabled in model")

        # Throw error if drag estimation enabled without drag
        # TODO: force enable drag?
        if model.drag_estimate and not model.drag:
            raise ValueError("Cannot estimate drag if disabled in model")

        # Store covariance provider
        self.covarianceProvider = covarianceProvider

    @staticmethod
    def _residuals(
        params: np.ndarray,
        dates: pd.DatetimeIndex,
        states: np.ndarray,
        model: NumericalPropagatorParameters,
    ) -> np.ndarray:
        # Calculate states
        states_ = ThalassaBatchLeastSquares._propagate(params, dates, model)

        # Calculate residuals
        residuals = (states_ - states).ravel()

        # Return residuals
        return residuals

    @staticmethod
    def _propagate(
        params: np.ndarray,
        dates: pd.DatetimeIndex,
        model: NumericalPropagatorParameters,
    ) -> np.ndarray:
        # Extract initial state vector
        state = params[0:6]

        # Make a copy of the model parameters
        model_ = deepcopy(model)

        # Adjust model
        if model_.srp_estimate and model_.drag_estimate:
            model_.cr = params[6]
            model_.drag = params[7]
        elif model_.srp_estimate:
            model_.cr = params[6]
        elif model_.drag_estimate:
            model_.drag = params[6]

        # Create propagator
        propagator = ThalassaNumericalPropagator(dates[0], state, model_)

        # Propagate states
        states = propagator.propagate(dates)

        # Return propagated states
        return states

    def estimate(self, guess: np.ndarray | None = None) -> ThalassaNumericalPropagator:
        # Extract the dates, states, and model
        dates = self.dates
        states = self.states
        model = self.model

        # Calculate scaling units
        keplerian = Keplerian.from_cartesian(dates[0:1], states[0:1, :])
        mu = Constants.DEFAULT_MU
        lu = keplerian[0, 0]
        vu = np.sqrt(mu / lu)

        # Calculate parameter scaling vector
        pscale = np.array(3 * [lu] + 3 * [vu])
        if self.srp_estimate:
            pscale = np.append(pscale, 1e-2)
        if self.drag_estimate:
            pscale = np.append(pscale, 1.0)  # TODO: update

        # Calculate residual scaling vector
        rscale = pscale[0:6]

        # Set initial guess
        # TODO: ratios (e.g. CR * A / m) instead of the coefficient alone
        if guess is None:
            p0 = np.copy(states[0, :])
        else:
            p0 = np.copy(guess)
        if self.srp_estimate:
            p0 = np.append(p0, model.cr)
        if self.drag_estimate:
            p0 = np.append(p0, model.cd)
        p0 /= pscale

        # Calculate observation covariance
        covariance = self.covarianceProvider.covariance(states)

        # Extract diagonal covariance matrix (and scale)
        # NOTE: only using the diagonal terms, matching generate_observations
        covarianceDiagonal = np.diag(np.diag(covariance))
        idx = np.diag_indices_from(covarianceDiagonal)
        covarianceDiagonal[idx] /= np.tile(rscale**2, len(dates))

        # Define predictor function
        def fun(_, *p):
            # Scale parameters (non-dimensional to dimensional)
            params = np.array(p) * pscale

            # Calculate states
            states_ = ThalassaBatchLeastSquares._propagate(
                params=params,
                dates=dates,
                model=model,
            )

            # Scale states (dimensional to non-dimensional)
            states_ /= rscale.reshape((1, -1))

            # Return column vector of states
            return states_.ravel()

        # Extract (and scale) observations
        x = dates
        y = states / rscale.reshape((1, -1))
        y = y.ravel()

        # Execute optimiser
        popt, pcov, infodict, mesg, ier = scipy.optimize.curve_fit(
            fun,
            x,  # NOTE: not used by function
            y,
            p0,
            sigma=covarianceDiagonal,
            absolute_sigma=True,
            method="lm",
            full_output=True,
            epsfcn=1e-8,
        )

        # Store optimisation scaling and results
        self.pscale = pscale
        self.rscale = rscale
        self.popt = popt
        self.pcov = pcov
        self.infodict = infodict
        self.mesg = mesg
        self.ier = ier

        # Extract estimated state
        stateEstimated = self.getEstimatedState()

        # Extract estimated model
        modelEstimated = self.getEstimatedModel()

        # Return solution
        return ThalassaNumericalPropagator(dates[0], stateEstimated, modelEstimated)

    def getEstimatedCovariance(self) -> np.ndarray:
        # Extract optimisation results
        pcov = self.pcov
        pscale = self.pscale

        # Generate scaling matrix
        pscaleMatrix = pscale.reshape((1, -1)) * pscale.reshape((-1, 1))

        # Return covariance matrix
        return pcov * pscaleMatrix

    def getEstimatedState(self) -> np.ndarray:
        # Extract optimisation results
        popt = self.popt
        pscale = self.pscale

        # Return estimated state
        return popt[0:6] * pscale[0:6]

    def getEstimatedModel(self) -> NumericalPropagatorParameters:
        # Extract optimisation results
        popt = self.popt
        pscale = self.pscale

        # Extract estimated model
        # TODO: cleaner solution
        model = deepcopy(self.model)
        if self.srp_estimate and self.drag_estimate:
            model.cr = popt[6] * pscale[6]
            model.cd = popt[7] * pscale[7]
        elif self.srp_estimate:
            model.cr = popt[6] * pscale[6]
        elif self.drag_estimate:
            model.drag = popt[6] * pscale[6]

        # Return estimated model
        return model
