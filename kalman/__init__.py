"""Provides utiltiy functions for generating Kalman processes and running
different variants of Kalman smoothing.
"""
from typing import Any
from typing import cast
from typing import Tuple
from typing import TypeVar

import numpy as np
from numpy.typing import NBitBase
from scipy import sparse

NpFlt = np.dtype[np.floating[NBitBase]]
Float1D = np.ndarray[int, NpFlt]
Float2D = np.ndarray[tuple[int, int], NpFlt]
FloatND = np.ndarray[Any, NpFlt]
FloatOrArray = TypeVar("FloatOrArray", float, FloatND)


def _dt_nt(stop: float, dt: float | None, nt: int | None) -> Tuple[Float1D, float, int]:
    """Handle creation of times, dt, and nt and guard ValueErrors"""
    if nt is not None:
        times = cast(Float1D, np.linspace(0, stop, nt))
        dt = stop / (nt - 1)
    elif dt is not None:
        times = cast(Float1D, np.arange(0, stop, dt))
        nt = len(times)
    else:
        raise ValueError("Either dt or nt must be provided")
    return times, dt, nt


def gen_sine(seed, *, stop=1, dt=None, nt=None, meas_var=0.1):
    """Generate (deterministic) sine trajectory and (random) measurements"""
    times, dt, nt = _dt_nt(stop, dt, nt)
    rng = np.random.default_rng(seed)
    x_true = np.sin(times)
    x_dot_true = np.cos(times)
    measurements = rng.normal(x_true, meas_var)
    H = sparse.lil_matrix((nt, 2 * nt))
    H[:, 1::2] = sparse.eye(nt)
    return measurements, x_true, x_dot_true, H, times


def gen_data(
    seed: int,
    *,
    stop: float = 1,
    dt: float | None = None,
    nt: int | None = None,
    meas_var: float = 0.1,
    process_var: float = 1,
) -> tuple[Float1D, Float1D, Float1D, sparse.lil_array, Float1D]:
    """Generate trajectory and measurements for a Kalman process.

    The standard Kalman Smoother is the maximum likelihood estimator for
    a trajectory whose velocity is a Brownian Motion.

    Args:
        seed: Numpy random seed
        stop: stop time
        dt: timestep.  Default is to infer from nt
        nt: number of timepoints.  Default is to infer from dt
        meas_var: measurement variance
        process_var: process variance
    """
    rng = np.random.default_rng(seed)
    times, dt, nt = _dt_nt(stop, dt, nt)
    Qs = [
        np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]]) for _ in range(nt - 1)
    ]
    Q = process_var * sparse.block_diag(Qs)
    x = rng.multivariate_normal(np.zeros(2 * nt - 2), Q.toarray())
    H = sparse.lil_array((nt, 2 * nt))
    H[:, 1::2] = sparse.eye(nt)
    dx_dot = H[:-1, 1:-1] @ x  # type: ignore
    x_dot_true = cast(Float1D, np.concatenate((np.zeros(1), dx_dot.cumsum())))
    dx = H[:-1, :-2] @ x + dt * x_dot_true[:-1]  # type: ignore
    x_true = cast(Float1D, np.concatenate((np.zeros(1), dx.cumsum())))
    meas_stdev = np.sqrt(meas_var)
    measurements = cast(Float1D, rng.normal(x_true, meas_stdev))

    return measurements, x_true, x_dot_true, H, times


def initialize_values(
    measurements: Float1D, times: Float1D, meas_var: float | np.ndarray
) -> tuple[
    Float1D,
    int,
    Float2D | sparse.dia_matrix,
    Float2D | sparse.dia_matrix,
    sparse.csr_array,
]:
    delta_times = times[1:] - times[:-1]
    T = len(times)
    R = meas_var
    if isinstance(R, float) or isinstance(R, int):
        R = R * sparse.eye(len(measurements))
        Rinv = cast(sparse.dia_matrix, 1 / meas_var * sparse.eye(len(measurements)))
    elif isinstance(R, np.ndarray):
        Rinv = cast(Float2D, np.linalg.inv(R))
    else:
        raise ValueError(
            "measurement variance sigma_z must either be a number or array."
        )
    G = gen_G(delta_times)
    return cast(Float1D, delta_times), T, R, Rinv, G


def gen_G(delta_times: Float1D) -> sparse.csr_array:
    T = len(delta_times) + 1
    G_left = sparse.block_diag([-np.array([[1, 0], [dt, 1]]) for dt in delta_times])
    G_right = sparse.eye(2 * (T - 1))
    align_cols = sparse.csc_array((2 * (T - 1), 2))
    G = sparse.hstack((G_left, align_cols)) + sparse.hstack((align_cols, G_right))
    return cast(sparse.csr_array, G)


def gen_Qi(dt: float):
    return np.array([[dt, dt**2 / 2], [dt**2 / 2, dt**3 / 3]])


def gen_Qinv(delta_times: Float1D, process_var: float = 1) -> sparse.spmatrix:
    Qs = [gen_Qi(dt) for dt in delta_times]
    Qinv = 1 / process_var * sparse.block_diag([np.linalg.inv(Q) for Q in Qs])
    return (Qinv + Qinv.T) / 2  # ensure symmetry


def solve(
    measurements: Float1D,
    obs_operator: Float2D | sparse.lil_matrix,
    times: Float1D,
    meas_var: float,
    process_var: float,
) -> tuple[Float1D, Float1D, sparse.csr_array, sparse.spmatrix, float]:
    H = obs_operator
    z = measurements.reshape((-1, 1))
    delta_times, T, _, Rinv, G = initialize_values(measurements, times, meas_var)
    Qinv = gen_Qinv(delta_times, process_var)

    rhs = H.T @ Rinv @ z.reshape((-1, 1))
    lhs = H.T @ Rinv @ H + G.T @ Qinv @ G
    sol = cast(Float1D, np.linalg.solve(lhs.toarray(), rhs))
    x_hat = cast(Float1D, (H @ sol).flatten())
    x_dot_hat = cast(Float1D, (H[:, list(range(1, 2 * T)) + [0]] @ sol).flatten())
    state_vec = cast(Float1D, restack(x_hat, x_dot_hat).reshape((-1, 1)))
    meas_err_hat = H @ state_vec - z
    loss = (
        meas_err_hat.T @ Rinv @ meas_err_hat + state_vec.T @ G.T @ Qinv @ G @ state_vec
    )
    return x_hat, x_dot_hat, G, Qinv, cast(float, loss)


def restack(x, x_dot):
    """Interleave x and x_dot to get vector represented by Kalman eqns

    Assumes first axis is time.
    """
    output_shape = (x.shape[0] + x_dot.shape[0], *x.shape[1:])
    c = np.empty(output_shape)
    c[0::2] = x_dot
    c[1::2] = x
    return c


def unstack(x):
    """unstack x vector represented by Kalman eqns to get x_dot and x

    Assumes first axis is time.
    """
    return x[1::2].flatten(), x[::2].flatten()


def gradient_test(f, g, x0):
    """Verifies that analytic function f matches gradient function g"""
    if isinstance(x0, np.ndarray):
        h = np.ones_like(x0) / np.linalg.norm(x0) / 1e2
    else:  # x0 is float or int
        h = x0 / 1e2
    return (f(x0 + h) - f(x0 - h)) / 2, np.dot(h, g(x0))


def complex_step_test(f, g, x0):
    """Verifies that analytic function f matches gradient function g"""
    if isinstance(x0, np.ndarray):
        h = np.ones_like(x0) / np.linalg.norm(x0) / 1e2
    else:  # x0 is float or int
        h = x0 / 1e2
    return f(x0 + h * 1j).imag, np.dot(h, g(x0))
