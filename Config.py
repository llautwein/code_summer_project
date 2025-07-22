from dataclasses import dataclass, field
from typing import List, Tuple, Union
from dolfin import Point, Expression
import ProblemDefinition as problem_def
import numpy as np

@dataclass
class BaseConfig:
    """
    A base config holding parameters for all analyses.
    """
    problem_1: problem_def.ProblemDefinition
    g_1: Expression
    problem_2: problem_def.ProblemDefinition
    g_2: Expression
    tol: float = 1e-6
    max_iter: int = 1000

@dataclass
class ConformingMeshAnalysisConfig(BaseConfig):
    """
    The config for the analysis using conforming meshes in the overlap.
    """
    # Geometry
    left_bottom_corner: Point=field(default_factory=lambda:Point(0, -0.25))
    length: float = 1
    height: float = 2
    mid_intersection: float = 0.75

    # Analysis lists
    polynomial_degrees: List[int] = field(default_factory=lambda:[1])
    interface_widths: Union[List[int], np.ndarray] = field(default_factory=lambda:np.logspace(-1, -2.5, 10))

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_conforming.csv"

@dataclass
class IndependentMeshAnalysisConfig(BaseConfig):
    """
    The config for the analysis using independent meshes.
    """
    # Geometry
    left_bottom_corner: Point = field(default_factory=lambda:Point(0, -0.25))
    length: float = 1
    height: float = 2
    mid_intersection: float = 0.75
    mesh_option: str = "gmsh"
    gmsh_paramters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                          "refinement_factor": 100.0,
                                                          "transition_ratio": 0.1})

    # Analysis lists
    polynomial_degrees: List[int] = field(default_factory=lambda:[1])
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda:np.logspace(-1, -4, 15))
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda:[0.1])

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_independent.csv"



