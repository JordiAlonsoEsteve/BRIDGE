import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bridge.model.bsr import BSR

"""
Checks that gradient and Hessian are OK!
"""


@pytest.fixture(scope="module", autouse=True)
def enable_x64():
    """Enable 64-bit precision for JAX during pytest execution."""
    jax.config.update("jax_enable_x64", True)


@pytest.fixture
def bsr_test_setup():
    """Sets up deterministic synthetic data and instantiates the BSR model."""
    np.random.seed(42)

    n_samples = 20
    m_targets = 30
    p_c = 4  # C features (5 with intercept)
    q_x = 5  # X features (6 with intercept)

    Y = np.random.randn(n_samples, m_targets)
    C = np.random.randn(n_samples, p_c)
    X = np.random.randn(q_x, m_targets)

    # Mock Laplacians (symmetric positive semi-definite)
    lambd, alpha = 0.5, 0.0

    model = BSR(
        X=X,
        Y=Y,
        C=C,
        lambd=lambd,
        alpha=alpha,
        dtype="float64",
    )

    # Dense Theta parameter matrix
    p_pad, q_pad = model.C.shape[1], model.X.shape[0]
    model.Theta = jnp.array(np.random.randn(p_pad, q_pad), dtype=jnp.float64)

    # Pre-calculated matrices for analytic gradient calculations
    A = (model.C.T @ model.C) / model.N
    B = model.X @ model.X.T
    D = (model.C.T @ model.Y @ model.X.T) / model.N

    return model, A, B, D, p_pad, q_pad


def test_analytic_gradient(bsr_test_setup):
    """Test analytic gradient against JAX autodiff gradient."""
    model, A, B, D, _, _ = bsr_test_setup

    grad_analytic = model._grad(model.Theta, A, B, D)

    grad_autodiff_fn = jax.grad(model.loss)
    grad_autodiff = grad_autodiff_fn(model.Theta)

    np.testing.assert_allclose(
        grad_analytic,
        grad_autodiff,
        rtol=1e-7,
        atol=1e-7,
        err_msg="Analytic gradient does not match JAX autodiff gradient.",
    )


def test_hessian(bsr_test_setup):
    """Test diagonal and full analytic submatrix Hessian against JAX autodiff Hessian."""
    model, _, _, _, p_pad, q_pad = bsr_test_setup

    # Identify active (non-zero) parameter indices in Fortran column-major order
    theta_flat = model.Theta.flatten(order="F")
    active = jnp.flatnonzero(theta_flat)

    # 1. Compute analytic Hessians
    H_diag_analytic = model._get_diagonal_hessian()
    H_full_analytic = model._get_full_hessian()

    # 2. Compute full JAX autodiff Hessian
    def flat_loss_fn(t_flat):
        Theta_reshaped = t_flat.reshape((p_pad, q_pad), order="F")
        return model.loss(Theta_reshaped)

    hessian_autodiff_fn = jax.hessian(flat_loss_fn)
    H_autodiff = hessian_autodiff_fn(theta_flat)

    # 3. Subset autodiff Hessian to match active set
    H_diag_autodiff = jnp.diag(H_autodiff)[active]
    H_full_autodiff = H_autodiff[active[:, None], active]

    # 4. Verify diagonal Hessian match
    np.testing.assert_allclose(
        H_diag_analytic,
        H_diag_autodiff,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Diagonal analytic Hessian does not match JAX autodiff diagonal Hessian.",
    )

    # 5. Verify full submatrix Hessian match
    np.testing.assert_allclose(
        H_full_analytic,
        H_full_autodiff,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Full analytic submatrix Hessian does not match JAX autodiff submatrix Hessian.",
    )
