from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd

from apps.tle_fit._tle_fit import (
    _make_thalassa_model,
    load_measurements,
    propagate_from_reference,
)
from brent.frames import RTN
from brent.propagators.numerical.thalassa import dates_to_MJD


def main(input_dir, output=None):
    # Read the single-object fit result.
    fit_path = next(Path(input_dir).glob("*.pkl"))
    fit = pd.read_pickle(fit_path).iloc[0]

    # Use the same at-epoch SGP4 -> GCRF conversion as tle_fit [m, m/s].
    _, dates, Y = load_measurements(
        fit["input_tle_path"], fit["window_start"], fit["window_end"]
    )

    # model_cfg is saved as a Python dictionary string. Restore its tolerance
    # and physical parameters, with cr = beta_fit * mass / area_srp.
    model = _make_thalassa_model(literal_eval(fit["model_cfg"]), fit["beta_fit"])
    X = propagate_from_reference(
        fit["ref_epoch"], fit["state_gcrf_fit"], model, dates
    )

    # R = r/|r|, C = (r x v)/|r x v|, I = C x R, using each TLE state.
    # Rotate position and velocity differences separately (no frame-rate term).
    # The requested sign is THALASSA - TLE, opposite to tle_fit's residuals.
    residuals = RTN.transform(RTN.getTransform(Y), X - Y)
    table = pd.DataFrame(residuals, columns=[
        "delta_r_R_m", "delta_r_I_m", "delta_r_C_m",
        "delta_v_R_m_s", "delta_v_I_m_s", "delta_v_C_m_s",
    ])
    table.insert(0, "MJD_UTC", dates_to_MJD(dates))
    table.insert(0, "EPOCH_UTC", dates.strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    output = Path(output) if output else fit_path.with_name(f"{fit_path.stem}_residuals.csv")
    table.to_csv(output, index=False)
    print(f"Saved {len(table)} residuals to {output}")
