import numpy as np
import scipy.sparse as sp
from typing import Tuple, Union

# 2D Laplacian via Kronecker sums; periodic or dirichlet BC.
# Accepts either: (Nx, Ny, dx, dy) or ((Nx,Ny), (dx,dy)).

def make_laplacian_2d(
    Nx: Union[int, Tuple[int, int]],
    Ny: Union[int, None] = None,
    dx: Union[float, Tuple[float, float], None] = None,
    dy: Union[float, None] = None,
    *,
    bc: str = "periodic",
    fmt: str = "csr",
) -> sp.spmatrix:
    """
    Build a 2D finite‑difference Laplacian on a uniform grid using a Kronecker‑sum
    construction. Supports periodic or Dirichlet boundary conditions.

    You can call it either as:
        make_laplacian_2d((Nx, Ny), (dx, dy), bc="periodic")
    or:
        make_laplacian_2d(Nx, Ny, dx, dy, bc="periodic")

    Args:
        Nx, Ny: grid sizes in x and y (>=2).
        dx, dy: grid spacings (>0).
        bc: "periodic" or "dirichlet".
        fmt: scipy sparse format for the output (e.g., "csr").

    Returns:
        sp.spmatrix in the requested format representing the 2D Laplacian.
    """
    # basic input guard
    assert Nx is not None, "Nx is required"
    # Unpack flexible args
    if Ny is None:
        # assume Nx=(Nx,Ny), dx=(dx,dy)
        assert isinstance(Nx, (tuple, list)) and dx is not None and isinstance(dx, (tuple, list)), "use (Nx,Ny),(dx,dy) or Nx,Ny,dx,dy"
        Nx, Ny = int(Nx[0]), int(Nx[1])
        dx, dy = float(dx[0]), float(dx[1])
    else:
        Nx, Ny = int(Nx), int(Ny)  # type: ignore[arg-type]
        assert dx is not None and dy is not None, "dx and dy required"
        # in this branch dx, dy must be scalars; guard against tuple input
        if isinstance(dx, (tuple, list)) or isinstance(dy, (tuple, list)):
            raise TypeError("Use make_laplacian_2d((Nx,Ny), (dx,dy), ...) for tuple inputs")
    dx, dy = float(dx), float(dy)

    assert Nx >= 2 and Ny >= 2 and dx > 0.0 and dy > 0.0, "Require Nx,Ny>=2 and dx,dy>0"

    bc = bc.lower()
    fmt = fmt.lower()

    Ix = sp.eye(Nx, format=fmt)
    Iy = sp.eye(Ny, format=fmt)

    # 1D Laplacians
    e_x = np.ones(Nx, dtype=float)
    diagonals_x = [-2.0 * e_x, np.ones(Nx - 1, dtype=float), np.ones(Nx - 1, dtype=float)]
    offsets_x = (0, -1, 1)
    Dx = sp.diags(diagonals_x, offsets=offsets_x, shape=(Nx, Nx), format=fmt) # type: ignore
    if bc == "periodic":
        Dx = Dx.tolil()
        Dx[0, -1] = 1.0
        Dx[-1, 0] = 1.0
        Dx = Dx.asformat(fmt)
    elif bc != "dirichlet":
        raise ValueError(f"Unsupported bc: {bc}")
    Dx = Dx * (1.0 / (dx*dx))

    e_y = np.ones(Ny, dtype=float)
    diagonals_y = [-2.0 * e_y, np.ones(Ny - 1, dtype=float), np.ones(Ny - 1, dtype=float)]
    offsets_y = (0, -1, 1)
    Dy = sp.diags(diagonals_y, offsets=offsets_y, shape=(Ny, Ny), format=fmt) # type: ignore
    if bc == "periodic":
        Dy = Dy.tolil()
        Dy[0, -1] = 1.0
        Dy[-1, 0] = 1.0
        Dy = Dy.asformat(fmt)
    Dy = Dy * (1.0 / (dy*dy))

    # Kronecker sum
    L = sp.kron(Iy, Dx, format=fmt) + sp.kron(Dy, Ix, format=fmt)
    return L