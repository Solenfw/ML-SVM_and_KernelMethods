"""svmlib: From-scratch SVM and Kernel Methods library (Topic 4 — P2)."""

from .dcd import LinearSVMDCD
from .kernels import (
    compute_kernel_column,
    compute_kernel_diagonal,
    compute_kernel_matrix,
    is_psd,
    linear_kernel,
    polynomial_kernel,
    rbf_kernel,
    resolve_gamma,
)
from .krr import KernelRidge
from .multiclass import OneVsOneClassifier, OneVsRestClassifier
from .platt import (
    PlattCalibrator,
    brier_score_multiclass,
    fit_platt_sigmoid,
    pairwise_coupling,
    reliability_diagram_data,
)
from .smo import BinarySMO, SVC

__all__ = [
    "SVC",
    "BinarySMO",
    "LinearSVMDCD",
    "KernelRidge",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "PlattCalibrator",
    "fit_platt_sigmoid",
    "pairwise_coupling",
    "brier_score_multiclass",
    "reliability_diagram_data",
    "linear_kernel",
    "polynomial_kernel",
    "rbf_kernel",
    "compute_kernel_matrix",
    "compute_kernel_column",
    "compute_kernel_diagonal",
    "resolve_gamma",
    "is_psd",
]
