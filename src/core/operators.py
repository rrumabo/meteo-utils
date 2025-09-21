import numpy as np
from scipy.sparse import spmatrix

def apply_op(op, u):
    """
    Apply a linear operator to a vector u.
    Supports numpy arrays, SciPy sparse matrices, or callables.
    """
    if callable(op):
        return op(u)
    if isinstance(op, (np.ndarray,)):
        return op @ u
    if isinstance(op, spmatrix):
        return op @ u
    raise TypeError(f"Unsupported operator type: {type(op)}")