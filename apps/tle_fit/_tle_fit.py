# Standard imports
from copy import deepcopy
from datetime import datetime, timedelta
import json
import time

# Third-party imports
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.optimize

# Orekit imports
import orekit
from orekit.pyhelpers import datetime_to_absolutedate
from org.orekit.propagation.analytical.tle import (
    TLE,
    TLEPropagator as OrekitTLEPropagator,
)

# Internal imports
from brent.constants import Constants
from brent.frames import RTN, Keplerian
from brent.io import Saver
from brent.propagators import NumericalPropagatorParameters, ThalassaNumericalPropagator
from brent.util import get_commit, has_uncommitted_changes


# ---------------------------------------------------------------------------
# CSV loader: read a tle_history_NNNNN.csv and reconstruct Orekit TLE objects
# ---------------------------------------------------------------------------

def _load_csv(path: str) -> tuple[str, pd.DataFrame]:
    """Return (norad_id_str, DataFrame with parsed mean elements).

    The CSV files in GEO_cat_2025 look like:
        # NORAD_CAT_ID: 00634
        # INTL_DESIGNATOR: 63031A
        # Starting Epoch (UTC): ...
        # Final Epoch (UTC): ...
        #
        EPOCH_DATETIME,MJD,MEAN_MOTION_REV_PER_DAY,...
        2025-01-01 16:14:11,...

    We skip all lines starting with '#' and parse the rest as a standard CSV.
    """
    norad_id = "UNKNOWN"
    with open(path, "r") as fid:
        for line in fid:
            if line.startswith("# NORAD_CAT_ID:"):
                norad_id = line.split(":", 1)[1].strip()
                break

    df = pd.read_csv(path, comment="#", parse_dates=["EPOCH_DATETIME"])
    return norad_id, df


def _build_tle(epoch_dt: datetime,
               n_rev_per_day: float,
               e: float,
               i_deg: float,
               raan_deg: float,
               aop_deg: float,
               ma_deg: float,
               bstar: float) -> TLE:
    """Construct an Orekit TLE from mean Kozai elements.

    The CSV stores:
        MEAN_MOTION_REV_PER_DAY   Kozai mean motion [rev/day]
        ECCENTRICITY              dimensionless
        INCLINATION_DEG           [deg]
        RAAN_DEG                  [deg]
        ARG_PERIGEE_DEG           [deg]
        MEAN_ANOMALY_DEG          [deg]
        BSTAR                     [1/Earth-radii] drag-like term used by SGP4

    Orekit's TLE constructor expects angles in radians and mean motion in rad/s.

    We use satelliteNumber=0 and blank classification / launch fields because the
    CSV does not preserve those; SGP4 evaluation (at the TLE's own epoch) does not
    need them.
    """
    # Convert units
    two_pi = 2.0 * np.pi
    n_rad_per_s = n_rev_per_day * two_pi / 86400.0   # rev/day → rad/s
    i_rad   = np.deg2rad(i_deg)
    raan_rad = np.deg2rad(raan_deg)
    aop_rad  = np.deg2rad(aop_deg)
    ma_rad   = np.deg2rad(ma_deg)

    epoch_abs = datetime_to_absolutedate(epoch_dt)

    return TLE(
        0,           # satelliteNumber
        "U",         # classification
        0,           # launchYear
        0,           # launchNumber
        "   ",       # launchPiece
        0,           # ephemerisType
        0,           # elementNumber
        epoch_abs,
        n_rad_per_s,
        0.0,         # meanMotionFirstDerivative (irrelevant at-epoch)
        0.0,         # meanMotionSecondDerivative
        float(e),
        float(i_rad),
        float(aop_rad),
        float(raan_rad),
        float(ma_rad),
        0,           # revolutionNumberAtEpoch
        float(bstar),
    )


def _tle_to_gcrf(tle: TLE, epoch_dt=None) -> tuple[datetime, np.ndarray]:
    """Evaluate SGP4 at ``epoch_dt`` and rotate TEME → GCRF.

    Orekit's OrekitTLEPropagator handles the TEME→GCRF rotation internally when
    we request PV coordinates in the GCRF frame via Constants.DEFAULT_ECI.

    If ``epoch_dt`` is omitted, the TLE's own epoch is used.

    Returns (epoch_datetime, state_gcrf) where state_gcrf is shape (6,) in [m, m/s].
    """
    from orekit.pyhelpers import absolutedate_to_datetime

    propagator = OrekitTLEPropagator.selectExtrapolator(tle)
    epoch_abs = (tle.getDate() if epoch_dt is None
                 else datetime_to_absolutedate(epoch_dt))
    pv = propagator.getPVCoordinates(epoch_abs, Constants.DEFAULT_ECI)
    pos = np.array(pv.getPosition().toArray())   # m
    vel = np.array(pv.getVelocity().toArray())   # m/s
    epoch_dt = absolutedate_to_datetime(epoch_abs)
    return epoch_dt, np.concatenate([pos, vel])


def load_measurements(tle_path: str,
                      start: datetime,
                      end: datetime,
                      return_tles: bool = False) -> tuple:
    """Load TLE CSV, filter to the fit window, and evaluate SGP4 at each epoch.

    Returns:
        norad_id  : NORAD catalog ID string
        dates     : list of datetime objects (measurement epochs), length N
        Y         : (N, 6) array of GCRF states [m, m/s]
        tles      : optional list of reconstructed TLEs when return_tles=True
    """
    norad_id, df = _load_csv(tle_path)

    # Filter to [start, end] window
    epochs_raw = pd.to_datetime(df["EPOCH_DATETIME"])
    mask = (epochs_raw >= pd.Timestamp(start)) & (epochs_raw <= pd.Timestamp(end))
    df = df[mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"No TLEs found between {start} and {end} in {tle_path}")

    # Reconstruct TLEs and evaluate SGP4 at each epoch
    dates = []
    states = []
    tles = []
    for _, row in df.iterrows():
        tle = _build_tle(
            epoch_dt=row["EPOCH_DATETIME"].to_pydatetime()
                if hasattr(row["EPOCH_DATETIME"], "to_pydatetime")
                else row["EPOCH_DATETIME"],
            n_rev_per_day=row["MEAN_MOTION_REV_PER_DAY"],
            e=row["ECCENTRICITY"],
            i_deg=row["INCLINATION_DEG"],
            raan_deg=row["RAAN_DEG"],
            aop_deg=row["ARG_PERIGEE_DEG"],
            ma_deg=row["MEAN_ANOMALY_DEG"],
            bstar=row["BSTAR"],
        )
        epoch_dt, state = _tle_to_gcrf(tle)
        tles.append(tle)
        dates.append(epoch_dt)
        states.append(state)

    Y = np.array(states)   # (N, 6)
    if return_tles:
        return norad_id, dates, Y, tles
    return norad_id, dates, Y


# ---------------------------------------------------------------------------
# Thalassa propagation helper: bidirectional from a reference epoch
# ---------------------------------------------------------------------------

def _make_thalassa_model(params_cfg: dict, beta: float) -> NumericalPropagatorParameters:
    """Build a NumericalPropagatorParameters from the config dict.

    β = C_R * A_srp / mass  [m²/kg]

    Thalassa's SRP model parameterises the acceleration as:
        a_SRP = (P_SRP / mass) * cr * area_srp
    so  a_SRP ∝ cr * area_srp / mass = β.

    We keep area_srp and mass fixed at their config values and derive:
        cr = β * mass / area_srp

    When area_srp = mass = 1 (the defaults), cr = β directly.
    """
    cfg = dict(params_cfg)   # shallow copy

    # Derive cr from β, mass, area_srp (handle missing keys with defaults)
    mass     = cfg.get("mass",     1.0)
    area_srp = cfg.get("area_srp", 1.0)
    cr       = beta * mass / area_srp

    # Force srp on, drag off (GEO default); override cr
    cfg.setdefault("potential",       True)
    cfg.setdefault("potential_degree", 8)
    cfg.setdefault("potential_order",  8)
    cfg.setdefault("sun",  True)
    cfg.setdefault("moon", True)
    cfg.setdefault("srp",  True)
    cfg.setdefault("drag", False)
    cfg.setdefault("tol",  1e-12)
    cfg.setdefault("area_drag", 1.0)
    cfg.setdefault("cd",        2.2)
    cfg["cr"] = cr

    # srp_estimate / drag_estimate must be False (we handle estimation ourselves)
    cfg["srp_estimate"]  = False
    cfg["drag_estimate"] = False

    return NumericalPropagatorParameters(**cfg)


def propagate_from_reference(t0: datetime,
                             state0: np.ndarray,
                             model: NumericalPropagatorParameters,
                             dates: list[datetime]) -> np.ndarray:
    """Propagate from reference epoch t0 to an arbitrary set of dates.

    Thalassa requires monotonically increasing or decreasing dates.
    We split the date list into three groups:
        at_t0    : dates exactly equal to t0  → return state0 directly
        forward  : dates > t0  (ascending from t0)
        backward : dates < t0  (descending from t0)
    Each branch that is propagated is prepended with t0, so Thalassa starts from
    the known state.  Results are reassembled in the original order.

    Returns X : (N, 6) array of GCRF states [m, m/s]
    """
    dates_pd = pd.to_datetime(dates)
    t0_pd    = pd.Timestamp(t0)

    idx_at   = np.where(dates_pd == t0_pd)[0]
    idx_fwd  = np.where(dates_pd  > t0_pd)[0]
    idx_back = np.where(dates_pd  < t0_pd)[0]

    X = np.full((len(dates), 6), np.nan)

    prop = ThalassaNumericalPropagator(t0, state0, model)

    # Dates exactly at t0: return the reference state directly
    if len(idx_at) > 0:
        X[idx_at] = state0

    # Forward branch: propagate from t0 → ascending dates
    if len(idx_fwd) > 0:
        fwd_dates  = pd.DatetimeIndex([t0_pd] + list(dates_pd[idx_fwd]))
        states_fwd = prop.propagate(fwd_dates)
        # states_fwd[0] is t0 (not needed); rows 1.. map to idx_fwd
        X[idx_fwd] = states_fwd[1:]

    # Backward branch: propagate from t0 → descending dates
    if len(idx_back) > 0:
        # Sort descending (closest to t0 first) so Thalassa gets monotone input
        order        = np.argsort(dates_pd[idx_back])[::-1]
        back_sorted  = list(dates_pd[idx_back][order])
        back_dates   = pd.DatetimeIndex([t0_pd] + back_sorted)
        states_back  = prop.propagate(back_dates)
        # states_back[0] is t0 (not needed); rows 1.. correspond to order
        for rank, orig_rank in enumerate(order):
            X[idx_back[orig_rank]] = states_back[rank + 1]

    return X


# ---------------------------------------------------------------------------
# Whitened residual function (§3 eq. whitened_residual)
# ---------------------------------------------------------------------------

def build_residual_fn(t0: datetime,
                      Y: np.ndarray,
                      dates: list[datetime],
                      model_cfg: dict,
                      sigma6: np.ndarray,
                      pscale: np.ndarray,
                      rtn_matrices: np.ndarray):
    """Return a closure `residuals(p_s)` for scipy.optimize.least_squares.

    The whitened residual is:
        z_i = T_i @ ε_i / σ   (component-wise division by σ)
    where  ε_i = Y_i - X_i(X₀)  and  T_i rotates GCRF → RIC (RTN convention).

    T_i is fixed to the *measurements* Y (not the nominal trajectory), per §3 of the
    supplementary notes.  This means T_i does not depend on the current iterate and
    does not enter the Jacobian — which is what allows scipy to finite-difference it
    cleanly.

    sigma6 : (6,) array of RIC std-devs [σ_rR, σ_rI, σ_rC, σ_vR, σ_vI, σ_vC]
    pscale : (7,) array [lu,lu,lu,vu,vu,vu,β_scale] — converts scaled → physical params
    rtn_matrices : (N, 6, 6) inertial→RIC rotation matrices, one per measurement epoch
    """
    def residuals(p_s: np.ndarray) -> np.ndarray:
        # Un-scale the parameter vector to physical units
        p = p_s * pscale
        state0 = p[0:6]   # [m, m/s]
        beta   = p[6]     # [m²/kg]

        # Build Thalassa model with this β
        model = _make_thalassa_model(model_cfg, beta)

        # Propagate from t0 to all measurement epochs
        X = propagate_from_reference(t0, state0, model, dates)   # (N, 6)

        # Measurement residual ε_i = Y_i - X_i  (GCRF)
        eps = Y - X   # (N, 6)

        # Rotate ε to RIC frame: z_i = T_i @ ε_i  (eq. 444)
        z_ric = np.einsum("nij,nj->ni", rtn_matrices, eps)   # (N, 6)

        # Whiten: divide by RIC std-devs  (eq. whitened_residual)
        z = z_ric / sigma6[np.newaxis, :]   # (N, 6), dimensionless

        return z.ravel()

    return residuals


# ---------------------------------------------------------------------------
# Reference-index helpers
# ---------------------------------------------------------------------------

def find_ref_index(dates: list, ref_epoch) -> int:
    """Return the index of the TLE epoch closest to ref_epoch."""
    ref_ts = pd.Timestamp(ref_epoch)
    diffs  = [abs((pd.Timestamp(d) - ref_ts).total_seconds()) for d in dates]
    return int(np.argmin(diffs))


def save_text_report(result: dict, path: str) -> None:
    """Save the concise, human-readable single-object fit report."""
    def utc(value) -> str:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")
        return timestamp.isoformat(sep=" ")

    def state_km(state) -> np.ndarray:
        return np.asarray(state) / 1000.0

    def write_section(fid, title: str, lines: list[str]) -> None:
        fid.write(f"[{title}]\n")
        fid.write("\n".join(lines))
        fid.write("\n\n")

    cov_ric_diag = np.diag(result["covariance_ric"]).copy()
    cov_ric_diag[:6] /= 1000.0**2  # m² and (m/s)² -> km² and (km/s)²
    with open(path, "w") as fid:
        fid.write("TLE FIT REPORT\n")
        fid.write("Units: state and RMS values are km and km/s; beta is m^2/kg.\n")
        fid.write("The covariance reported in the RIC frame is diagonal.\n\n")

        write_section(fid, "Input", [
            f"input_tle_path: {result['input_tle_path']}",
            f"window_start_utc: {utc(result['window_start'])}",
            f"window_end_utc: {utc(result['window_end'])}",
            f"ref_tle_index: {result['ref_tle_index']}",
            f"ref_epoch_utc: {utc(result['ref_epoch'])}",
            f"beta_guess_m2kg: {result['beta_guess']}",
        ])

        write_section(fid, "Git", [
            f"commit: {result['git_commit']}",
            f"uncommitted_changes: {result['git_uncommitted_changes']}",
        ])

        write_section(fid, "State at fit epoch (GCRF)", [
            "state_tle_gcrf: [Rx, Ry, Rz, Vx, Vy, Vz]",
            np.array2string(state_km(result["state_tle_gcrf"]), precision=12,
                            max_line_width=120),
            "state_gcrf_fit: [Rx, Ry, Rz, Vx, Vy, Vz]",
            np.array2string(state_km(result["state_gcrf_fit"]), precision=12,
                            max_line_width=120),
            f"beta_fit_m2kg: {result['beta_fit']}",
        ])

        write_section(fid, "Covariance (RIC diagonal)", [
            "[R, I, C, vR, vI, vC, beta]",
            np.array2string(cov_ric_diag, precision=12, max_line_width=120),
        ])

        eps_rms = result["eps_rms_ric"] / 1000.0
        write_section(fid, "Residuals", [
            f"dimensionless_rms: {result['wrms']:.12g}",
            f"RIC_pos_RMS_km: R={eps_rms[0]:.12g} I={eps_rms[1]:.12g} C={eps_rms[2]:.12g}",
            f"RIC_vel_RMS_km_s: R={eps_rms[3]:.12g} I={eps_rms[4]:.12g} C={eps_rms[5]:.12g}",
        ])

        write_section(fid, "Solver (scipy.optimize.least_squares, LM)", [
            f"status: {result['solver_message']}",
        ])


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and validate the JSON configuration file."""
    with open(path, "r") as fid:
        cfg = json.load(fid)

    dateformat = "%Y-%m-%d"

    # Parse window
    window = cfg["window"]
    start = datetime.strptime(window["start"], dateformat)
    if "end" in window:
        end = datetime.strptime(window["end"], dateformat)
    else:
        duration_days = window.get("duration", 90)
        end = start + timedelta(days=duration_days)
    cfg["window"]["_start"] = start
    cfg["window"]["_end"]   = end

    # Optional arbitrary reference epoch. "auto" selects the window midpoint.
    reference = cfg.get("reference", {})
    if "epoch" in reference:
        ref_spec = reference["epoch"]
        if ref_spec == "auto":
            reference["_epoch"] = start + (end - start) / 2
        else:
            reference["_epoch"] = pd.Timestamp(ref_spec).to_pydatetime()

    return cfg


# ---------------------------------------------------------------------------
# Core fit routine (reusable by both single-object and catalog scripts)
# ---------------------------------------------------------------------------

def fit_object(
    tle_path: str,
    cfg: dict,
    ref_epoch=None,      # datetime: exact fit epoch; default is window midpoint
    verbose: bool = True,
) -> dict:
    """Fit [R₀, V₀, β] to a single object's TLE sequence.

    Parameters
    ----------
    tle_path : path to tle_history_NNNNN.csv
    cfg      : parsed config dict (from load_config or equivalent)
    ref_epoch: exact reference epoch for the fitted state. The closest TLE is
               propagated with SGP4 to this epoch to form the initial guess.
               If omitted, the fit-window midpoint is used, unless the legacy
               cfg['reference']['index'] is explicitly provided.
    verbose  : print progress to stdout

    Returns
    -------
    result_dict containing fitted state, covariances, residuals, and metadata.
    Raises on any failure (caller should catch for batch use).
    """
    t_start = time.time()

    start     = cfg["window"]["_start"]
    end       = cfg["window"]["_end"]
    beta0     = float(cfg.get("beta", 0.02))
    sigma6    = np.array(cfg["noise"]["sigma"])
    model_cfg = cfg["model"]
    ls_cfg    = cfg.get("least_squares", {})

    # ---- 1. Load TLE measurements -------------------------------------------
    if verbose:
        print(f"Loading TLEs from {tle_path} ...")
    norad_id, dates, Y, tles = load_measurements(
        tle_path, start, end, return_tles=True
    )
    N = len(dates)
    if verbose:
        print(f"  {N} TLEs loaded for NORAD {norad_id}")

    if N < 2:
        raise ValueError(f"Need at least 2 TLEs (got {N}) to estimate 7 parameters")
    
    # ---- 2. Reference epoch and initial guess --------------------------------
    reference_cfg = cfg.get("reference", {})
    if ref_epoch is not None:
        ref_index = find_ref_index(dates, ref_epoch)
        t0 = pd.Timestamp(ref_epoch).to_pydatetime()
        if not (pd.Timestamp(start) <= pd.Timestamp(t0) <= pd.Timestamp(end)):
            raise ValueError(
                f"reference epoch {t0} outside fit window [{start}, {end}]"
            )
    elif "index" in reference_cfg:
        # Backward-compatible explicit selection of a TLE epoch.
        ref_index = reference_cfg["index"]
        t0 = dates[ref_index] if 0 <= ref_index < N else None
    else:
        t0 = start + (end - start) / 2
        ref_index = find_ref_index(dates, t0)

    if not (0 <= ref_index < N):
        raise ValueError(f"reference index {ref_index} out of range [0, {N-1}]")

    # Evaluate the closest/selected TLE at the exact fit reference epoch.
    _, state0_guess = _tle_to_gcrf(tles[ref_index], t0)
    if verbose:
        print(f"  Reference epoch t0 = {t0}")
        print(
            f"  Initial guess from TLE index {ref_index} "
            f"(epoch {dates[ref_index]}), propagated to t0 with SGP4"
        )

    # ---- 3. Parameter scaling ------------------------------------------------
    kep = Keplerian.from_cartesian([t0], state0_guess.reshape(1, 6))
    lu  = kep[0, 0]
    vu  = np.sqrt(Constants.DEFAULT_MU / lu)
    beta_scale = max(abs(beta0), 1e-3)
    pscale = np.array([lu, lu, lu, vu, vu, vu, beta_scale])

    p0 = np.concatenate([state0_guess, [beta0]]) / pscale

    # ---- 4. Pre-compute RIC rotation matrices from measurements (fixed) ------
    rtn_matrices = RTN.getTransform(Y)   # (N, 6, 6)

    # ---- 5. Build residual function and run LM fit ---------------------------
    residuals_fn = build_residual_fn(
        t0=t0,
        Y=Y,
        dates=dates,
        model_cfg=model_cfg,
        sigma6=sigma6,
        pscale=pscale,
        rtn_matrices=rtn_matrices,
    )

    ls_kwargs = {
        "jac":      ls_cfg.get("jac", "3-point"),
        "method":   ls_cfg.get("method",   "lm"),
        "ftol":     ls_cfg.get("ftol",     1e-8),
        "xtol":     ls_cfg.get("xtol",     1e-8),
        "gtol":     ls_cfg.get("gtol",     1e-8),
        "max_nfev": ls_cfg.get("max_nfev", None),
        "x_scale":  ls_cfg.get("x_scale",  "jac"),
        "verbose":  ls_cfg.get("verbose",   1) if verbose else 0,
    }
    if ls_cfg.get("diff_step") is not None:
        ls_kwargs["diff_step"] = ls_cfg["diff_step"]

    if verbose:
        print("Running Levenberg-Marquardt fit ...")
    result = scipy.optimize.least_squares(residuals_fn, p0, **ls_kwargs)

    # ---- 6. Extract fitted state and β ---------------------------------------
    p_fit     = result.x * pscale
    state_fit = p_fit[0:6]
    beta_fit  = p_fit[6]

    # ---- 7. Covariance from the Jacobian (whitened → physical) ---------------
    J = result.jac
    if J is None or J.size == 0:
        if verbose:
            print("Warning: Jacobian not available; covariance set to NaN")
        P_scaled_7 = np.full((7, 7), np.nan)
    else:
        JtJ        = J.T @ J
        P_scaled_7 = np.linalg.pinv(JtJ)

    D      = np.diag(pscale)
    P_gcrf = D @ P_scaled_7 @ D

    T0_66  = RTN.getTransform(state_fit.reshape(1, 6))[0, :, :]
    M      = scipy.linalg.block_diag(T0_66, np.array([[1.0]]))
    P_ric  = M @ P_gcrf @ M.T

    # ---- 8. Residual statistics ----------------------------------------------
    z_final      = residuals_fn(result.x).reshape(N, 6)
    wrms         = np.sqrt(np.mean(z_final**2))
    eps_ric_phys = z_final * sigma6[np.newaxis, :]
    eps_rms      = np.sqrt(np.mean(eps_ric_phys**2, axis=0))

    wallclock_s = time.time() - t_start

    if verbose:
        print(f"  Solver status : {result.message}")
        print(f"  Iterations    : {result.nfev} function evaluations")
        print(f"  Dimensionless RMS : {wrms:.4f}  (should be ~1 for well-calibrated noise)")
        print(f"  RIC pos RMS   : R={eps_rms[0]:.2f} I={eps_rms[1]:.2f} C={eps_rms[2]:.2f} m")
        print(f"  RIC vel RMS   : R={eps_rms[3]:.4f} I={eps_rms[4]:.4f} C={eps_rms[5]:.4f} m/s")
        print(f"  β_fit = {beta_fit:.6f} m²/kg")
        print(f"  Wallclock     : {wallclock_s:.1f} s")

    # ---- 9. Assemble result dict ---------------------------------------------
    return {
        # Inputs (reproducibility)
        "input_tle_path":   tle_path,
        "norad_id":         norad_id,
        "window_start":     start,
        "window_end":       end,
        "n_measurements":   N,
        "ref_tle_index":    ref_index,
        "ref_epoch":        t0,
        "beta_guess":       beta0,
        "sigma_ric":        sigma6,
        "model_cfg":        str(model_cfg),
        "ls_cfg":           str(ls_cfg),
        "git_commit":       get_commit(),
        "git_uncommitted_changes": has_uncommitted_changes(),

        # Fitted state (GCRF at ref_epoch)
        # States: [Rx, Ry, Rz, Vx, Vy, Vz] in [m, m/s]
        "state_gcrf_fit":   state_fit,
        "state_tle_gcrf":   state0_guess,
        "beta_fit":         beta_fit,

        # Covariances
        # covariance_gcrf : 7×7 cov of [Rx,Ry,Rz,Vx,Vy,Vz,β] in [m²,(m/s)²,(m²/kg)²]
        "covariance_gcrf":  P_gcrf,
        # covariance_ric  : same rotated to RIC at ref_epoch
        "covariance_ric":   P_ric,
        # ric_transform   : 6×6 GCRF→RIC rotation at fitted state
        "ric_transform":    T0_66,

        # Residual statistics
        # wrms        : dimensionless weighted RMS ‖z‖/√(N·6)
        "wrms":             wrms,
        # eps_rms_ric : (6,) per-component physical RMS [m, m/s] in RIC
        "eps_rms_ric":      eps_rms,
        # z_final     : (N,6) whitened residuals (dimensionless)
        "z_final":          z_final,
        # dates       : list of N measurement epoch datetimes
        "dates":            dates,

        # Solver info
        "solver_status":    result.status,
        "solver_message":   result.message,
        "solver_nfev":      result.nfev,
        "solver_cost":      result.cost,

        # Timing
        "wallclock_s":      wallclock_s,

    }


# ---------------------------------------------------------------------------
# Main entry point (single-object)
# ---------------------------------------------------------------------------

def main(input_path: str, output_dir: str) -> None:
    """Run the TLE batch least-squares fit for a single object and save results."""

    cfg = load_config(input_path)

    saver = Saver(output_dir)
    ref_epoch = cfg.get("reference", {}).get("_epoch")
    result = fit_object(cfg["tle"], cfg, ref_epoch=ref_epoch, verbose=True)

    saver.save_input(input_path)
    saver.update(result)
    saver.save(final=True)
    save_text_report(result, f"{saver.directory}/{saver.name}_report.txt")

    print(f"Results saved to {saver.directory}")
