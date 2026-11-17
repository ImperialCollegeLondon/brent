# Vendored Vallado SGP4

Copied from `https://github.com/CelesTrak/fundamentals-of-astrodynamics` at commit
`49df0479c950917c0a2c5ac31f2db6477bba57f6`.

The Python source files under `valladopy/astro/` and `valladopy/mathtime/` match
the upstream files byte for byte. Pydantic is retained as an upstream runtime
dependency for `SatRec`.

The upstream Python SGP4 caller omits the current node, argument of perigee, and
mean anomaly when preparing a deep-space resonance update. BRENT supplies those
values in `propagator.py`, outside the copied Vallado files. This keeps the
vendored source unchanged and preserves the correction checked against Vallado's
C++ SGP4 and Orekit's deep-space TLE propagator.

The Vallado TEME-to-ECI conversion routines are not included: BRENT converts
Vallado TEME PV coordinates with Orekit's TEME-to-GCRF frame transform.
