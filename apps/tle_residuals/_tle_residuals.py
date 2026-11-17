from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from brent.propagators import ValladoSGP4Propagator

from apps.tle_fit._tle_fit import (
    _make_thalassa_model,
    load_measurements,
    propagate_from_reference,
)
from brent.constants import Constants
from brent.frames import RTN
from brent.propagators.numerical.thalassa import dates_to_MJD


def _propagate_tle(propagator, date):
    """Return one SGP4 state in GCRF at ``date`` [m, m/s]."""
    date = pd.Timestamp(date).to_pydatetime()
    return propagator._propagate(date, Constants.DEFAULT_ECI)


def giorgini_tle_states(tle_dates, tles, evaluation_dates):
    """Evaluate the Giorgini cosine-blended TLE solution in GCRF.

    For every evaluation time, the two bracketing TLEs are propagated with
    SGP4 and combined according to equations 28--31 in the paper.  The
    derivative of the cosine weight is included in the blended velocity.
    """
    tle_dates = pd.DatetimeIndex(tle_dates)
    evaluation_dates = pd.DatetimeIndex(evaluation_dates)
    propagators = [
        ValladoSGP4Propagator(tle) for tle in tles
    ]

    states = np.empty((len(evaluation_dates), 6))
    for j, date in enumerate(evaluation_dates):
        k = max(tle_dates.searchsorted(date) - 1, 0)
        dt = (tle_dates[k + 1] - tle_dates[k]).total_seconds()
        tau = (date - tle_dates[k]).total_seconds()
        phase = np.pi * tau / dt
        weight = 0.5 + 0.5 * np.cos(phase)
        weight_rate = -0.5 * np.pi / dt * np.sin(phase)

        state_k = _propagate_tle(propagators[k], date)
        state_k1 = _propagate_tle(propagators[k + 1], date)
        states[j, :3] = weight * state_k[:3] + (1.0 - weight) * state_k1[:3]
        states[j, 3:] = (
            weight * state_k[3:]
            + (1.0 - weight) * state_k1[3:]
            + weight_rate * (state_k[:3] - state_k1[:3])
        )

    return states


def _residual_table(dates, fitted_states, tle_states):
    """Build the RTN residual table for fitted minus TLE states."""
    residuals = RTN.transform(
        RTN.getTransform(tle_states), fitted_states - tle_states
    )
    table = pd.DataFrame(residuals, columns=[
        "delta_r_R_m", "delta_r_I_m", "delta_r_C_m",
        "delta_v_R_m_s", "delta_v_I_m_s", "delta_v_C_m_s",
    ])
    table["delta_r_norm_m"] = np.linalg.norm(residuals[:, :3], axis=1)
    table["delta_v_norm_m_s"] = np.linalg.norm(residuals[:, 3:], axis=1)
    table.insert(0, "MJD_UTC", dates_to_MJD(dates))
    table.insert(0, "EPOCH_UTC", dates.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    return table


def main(input_dir, output=None, step_hours=1.0):
    # Read the single-object fit result.
    fit_path = next(Path(input_dir).glob("*.pkl"))
    fit = pd.read_pickle(fit_path).iloc[0]

    # Use the same at-epoch SGP4 -> GCRF conversion as tle_fit [m, m/s].
    _, dates, Y, tles = load_measurements(
        fit["input_tle_path"], fit["window_start"], fit["window_end"],
        return_tles=True,
    )
    dates = pd.DatetimeIndex(dates)

    step = pd.to_timedelta(step_hours, unit="h")
    dense_dates = pd.date_range(dates[0], dates[-1], freq=step)

    # model_cfg is saved as a Python dictionary string. Restore its tolerance
    # and physical parameters, with cr = beta_fit * mass / area_srp.
    model = _make_thalassa_model(literal_eval(fit["model_cfg"]), fit["beta_fit"])
    X = propagate_from_reference(
        fit["ref_epoch"], fit["state_gcrf_fit"], model, dates
    )

    # Preserve the existing residual file at the original TLE epochs.
    table = _residual_table(dates, X, Y)
    output = (
        Path(output)
        if output
        else fit_path.with_name(f"{fit_path.stem}_residuals.csv")
    )
    table.to_csv(output, index=False)
    print(f"Saved {len(table)} residuals to {output}")

    # Build the dense fitted and Giorgini-blended TLE solutions.  The grid starts
    # at the first TLE epoch and stops at the last full step before the final TLE,
    # so every epoch is bracketed by two TLEs and the spacing remains uniform.
    X_dense = propagate_from_reference(
        fit["ref_epoch"], fit["state_gcrf_fit"], model, dense_dates
    )
    Y_dense = giorgini_tle_states(dates, tles, dense_dates)
    dense_table = _residual_table(dense_dates, X_dense, Y_dense)
    dense_output = output.with_name(f"{output.stem}_dense.txt")
    dense_table.to_csv(dense_output, index=False)
    print(
        f"Saved {len(dense_table)} dense residuals "
        f"({step_hours:g} h timestep) to {dense_output}"
    )
