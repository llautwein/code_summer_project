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
    # Geometry
    left_bottom_corner: Point = field(default_factory=lambda: Point(0, -0.25))
    length: float = 1
    height: float = 2
    mid_intersection: float = 0.75

@dataclass
class ConformingMeshAnalysisConfig(BaseConfig):
    """
    The config for the analysis using conforming meshes in the overlap.
    """
    # Mesh
    mesh_option: str = "gmsh"
    gmsh_parameters: dict = field(default_factory=lambda: {"lc_coarse": 0.25})

    # Analysis lists
    polynomial_degrees: List[int] = field(default_factory=lambda:[1])
    interface_widths: Union[List[int], np.ndarray] = field(default_factory=lambda:np.logspace(-1, -4, 12))

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_conforming.csv"

@dataclass
class IndependentMeshAnalysisConfig(BaseConfig):
    """
    The config for the analysis using independent meshes.
    """
    # Mesh
    mesh_option: str = "gmsh"
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                          "refinement_factor": 150.0,
                                                          "transition_ratio": 0.1})

    # Analysis lists
    polynomial_degrees: List[int] = field(default_factory=lambda:[1])
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda:np.logspace(-1, -4, 12))
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda:[0.1])

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_independent.csv"

@dataclass
class OffsetMeshAnalysisConfig(BaseConfig):
    """
    The config for the analysis using offset meshes.
    """
    # Mesh
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                           "refinement_factor": 150.0,
                                                           "transition_ratio": 0.1})

    # Analysis lists
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.1])
    polynomial_degrees: List[int] = field(default_factory=lambda: [1])
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.1, 0.01, 0.001])
    offset_pctg: Union[List[float], np.ndarray] = field(default_factory=lambda: np.linspace(0, 0.9, 50))

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_offset.csv"

@dataclass
class DDMComparisonConfig(BaseConfig):

    # Mesh
    mesh_option: str = "built-in"
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": False,
                                                           "refinement_factor": 10.0,
                                                           "transition_ratio": 0.1})
    # Analysis lists
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.1])
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: np.logspace(-0.5, -2, 10))

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/ddm_comparison.csv"



