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
    mesh_option = "gmsh"
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                           "refinement_factor": 150.0,
                                                           "transition_ratio": 0.1})

    # Analysis lists
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.1])
    polynomial_degrees: List[int] = field(default_factory=lambda: [1])
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda: np.logspace(-1, -4, 12))
    offset_pctg: Union[List[float], np.ndarray] = field(default_factory=lambda: [0, 0.5, 0.9])

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_offset.csv"

@dataclass
class ScalabilityAnalysisConfig(BaseConfig):
    """
    The config for investigating the influence of increasing the DoFs.
    """
    # Mesh
    mesh_option: str = "gmsh"
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                           "refinement_factor": 10.0,
                                                           "transition_ratio": 0.1})

    # Analysis lists
    polynomial_degrees: List[int] = field(default_factory=lambda: [1, 2])
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.01])
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: np.logspace(-0.5, -2, 6))

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_scalability.csv"

@dataclass
class InterpolationError(BaseConfig):
    mesh_option: str = "gmsh"
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                           "refinement_factor": 10.0,
                                                           "transition_ratio": 0.1})
    polynomial_degree: List[int] = field(default_factory=lambda: 1)
    delta: Union[List[float], np.ndarray] = field(default_factory=lambda: 0.1)
    N_overlaps: List[int] = field(default_factory=lambda: [1, 2, 8, 20, 60])
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: np.logspace(-0.5, -2, 5))
    results_path = "output_files/interpolation_error.csv"

    use_lu_solver: bool = True

@dataclass
class MeshAnalysis3d(BaseConfig):
    """
    The config for the analysis using independent meshes.
    """
    # Mesh
    left_bottom_corner: Point = field(default_factory=lambda: Point(0, 0, -0.25))
    length: float = 1
    width: float = 1
    height: float = 2
    mid_intersection: float = 0.75
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": True,
                                                           "refinement_factor": 75.0,
                                                           "transition_ratio": 0.1})

    # Analysis lists
    polynomial_degrees: List[int] = field(default_factory=lambda: [1])
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda: np.logspace(-1, -3, 8))
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.25])

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/algebraic_schwarz_analysis_3d.csv"

@dataclass
class DDMComparisonConfig(BaseConfig):

    # Mesh
    mesh_option: str = "built-in"
    gmsh_parameters: dict = field(default_factory=lambda: {"refine_at_interface": False,
                                                           "refinement_factor": 10.0,
                                                           "transition_ratio": 0.1})
    # Analysis lists
    interface_widths: Union[List[float], np.ndarray] = field(default_factory=lambda: [0.1])
    mesh_resolutions: Union[List[float], np.ndarray] = field(default_factory=lambda: np.logspace(-1, -2.5, 10))

    # Solver
    use_lu_solver: bool = True

    # I/O
    results_path: str = "output_files/ddm_comparison.csv"



