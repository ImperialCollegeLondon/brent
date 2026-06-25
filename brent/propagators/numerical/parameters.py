# Standard imports
from dataclasses import dataclass


@dataclass
class NumericalPropagatorParameters:
    # Physical properties
    mass: float

    # Geopotential perturbations
    potential: bool
    potential_degree: int
    potential_order: int

    # Third-body perturbations
    moon: bool
    sun: bool

    # SRP
    srp: bool
    srp_estimate: bool
    cr: float
    area_srp: float

    # Drag
    drag: bool
    drag_estimate: bool
    cd: float
    area_drag: float

    # Tolerance
    tol: float = 1e-12
