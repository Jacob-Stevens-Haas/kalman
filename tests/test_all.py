import warnings

import numpy as np
import pytest
import scipy
from scipy import stats

import kalman


# constants
@pytest.fixture
def sigma_z():
    return 0.1


@pytest.fixture
def sigma_x():
    return 1


@pytest.fixture
def seed():
    return 52


def test_restack():
    x = np.array([[1, 2], [3, 4]])
    x_dot = np.array([[5, 6], [7, 8]])
    expected = np.array([
        [5, 6],
        [1, 2],
        [7, 8],
        [3, 4],
    ])
    result = kalman.restack(x, x_dot)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
def sample_data(sigma_z, sigma_x, seed):
    return kalman.gen_data(seed, nt=100, meas_var=sigma_z, process_var=sigma_x)


def test_gen_data_marginal_dists(sample_data, sigma_x, sigma_z):
    """Test marginal distributions of x and x_dot

    Uses a Kolmogorov-Smirnoff test of each variable, divided by it's
    known standard deviation.
    """
    _, x_true, x_dot_true, _, times = sample_data
    delta_times = times[1:] - times[:-1]

    x_innovation = x_true[1:] - x_true[:-1] - delta_times * x_dot_true[:-1]
    x_stdev = np.sqrt(sigma_x * delta_times**3 / 3)
    p_x = stats.kstest(x_innovation / x_stdev, stats.norm.cdf).pvalue
    assert p_x > 0.05

    x_dot_innovation = x_dot_true[1:] - x_dot_true[:-1]
    x_dot_stdev = np.sqrt(sigma_x * delta_times)
    p_xdot = stats.kstest(x_dot_innovation / x_dot_stdev, stats.norm.cdf).pvalue
    assert p_xdot > 0.05


def test_gen_data_joint_dists(sample_data, sigma_x):
    """Test joint distribution of x and x_dot.

    Uses a Kolmogorov-Smirnoff test of each variable, multiplied by the
    root-inverse of its variance.
    """
    _, x_true, x_dot_true, _, times = sample_data
    true_state = kalman.restack(x_true, x_dot_true)
    delta_times = times[1:] - times[:-1]
    G = kalman.gen_G(delta_times)
    Q = np.linalg.inv(kalman.gen_Qinv(delta_times, sigma_x).todense())
    G_dagger = np.linalg.pinv(G.toarray())
    var_state = G_dagger @ Q @ G_dagger.T
    prematrix = root_pinv(var_state)
    true_state_normalized = prematrix @ true_state
    p_state = stats.kstest(true_state_normalized, stats.norm.cdf).pvalue
    assert p_state > 0.05


def test_gen_data_noise_dist(sample_data, sigma_z):
    """Test distribution of generated measurement errors

    Uses a Kolmogorov-Smirnoff test of each variable, divided by it's
    known standard deviation.
    """
    measurements, x_true, _, _, _ = sample_data
    meas_error = measurements - x_true
    meas_stdev = np.sqrt(sigma_z)
    p_z = stats.kstest(meas_error / meas_stdev, stats.norm.cdf).pvalue
    assert p_z > 0.05


def test_kalman_solution(sample_data, sigma_z, sigma_x):
    """Test solution of Kalman smoothing as well as its error.

    Uses a Kolmogorov-Smirnoff test of each variable, multiplied by the
    root-inverse of its variance.
    """
    measurements, x_true, x_dot_true, H, times = sample_data
    x_hat, x_dot_hat, G, Qinv = kalman.solve(measurements, H, times, sigma_z, sigma_x)
    nt = len(times)
    true_state = kalman.restack(x_true, x_dot_true)
    sol_state = kalman.restack(x_hat, x_dot_hat)
    err = sol_state - true_state

    Rinv = sigma_z * scipy.sparse.eye(nt)
    nt = len(x_true)
    G_dagger = np.linalg.pinv(G.toarray())
    Rinv = 1 / sigma_z * np.eye(nt)
    R = sigma_z * np.eye(nt)
    Q = np.linalg.inv(Qinv.toarray())

    var_x = G_dagger @ Q @ G_dagger.T
    var_z = R + H @ var_x @ H.T
    hessian = G.T @ Qinv @ G + H.T @ Rinv @ H
    hess_inv = np.linalg.inv(hessian)  # at 30 timepoints, condition:1.2e6
    var_x_hat = hess_inv @ H.T @ Rinv @ var_z @ Rinv @ H @ hess_inv

    p_sol = stats.kstest(root_pinv(var_x_hat) @ sol_state, stats.norm.cdf).pvalue
    assert p_sol > 0.05
    p_err = stats.kstest(root_pinv(var_x_hat + var_x) @ err, stats.norm.cdf).pvalue
    assert p_err > 0.05


def root_pinv(Q, threshold=1e-15):
    r"""Calculate the root pseudoinverse of matrix Q in R m x m using the SVD

    If :math:`x \sim \mathcal N(0, Q)`, then
    :math:`\tilde U^T\Sigma^{-1/2}x\sim \mathcal N(0, I)` if
    :math:`Q=\tilde U\tilde\sigma\tilde U^T.

    Q must be symmetric, and to make sense as a (singular) covariance, it must be
    positive (semi) definite)

    Arguments:
        Q: The matrix to calculate
        threshold: for determining rank of Q

    Returns:
        Qp_root, the k x m matrix where k is the rank of Q and the above equation holds.
    """
    U, s, _ = scipy.linalg.svd(Q)
    s_diag_nonzero = s / s.max() > threshold
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
        S_root_pinv = np.diag(s[s_diag_nonzero] ** -0.5)
    return S_root_pinv @ U.T[s_diag_nonzero]
