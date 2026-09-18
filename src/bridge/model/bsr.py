import time
import warnings
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# allow jax to use 64-bit floats for higher numerical precision
jax.config.update("jax_enable_x64", True)


@jax.jit
def prox_exclusive_lasso_cols(V, lmbda):
    """
    JAX-optimized column-wise proximal operator for Exclusive Lasso.
    https://arxiv.org/pdf/2306.14196
    """

    def apply_prox(V_in, lmbda_in):
        abs_V = jnp.abs(V_in)
        num_rows = V_in.shape[0]

        sorted_V = jnp.sort(abs_V, axis=0)[::-1, :]
        cumsum_V = jnp.cumsum(sorted_V, axis=0)

        k_array = jnp.arange(1, num_rows + 1).reshape(-1, 1)
        candidate_taus = (lmbda_in * cumsum_V) / (1.0 + 1 * k_array * lmbda_in)

        greater_mask = sorted_V > candidate_taus
        valid_taus = jnp.where(greater_mask, candidate_taus, 0.0)
        final_taus = jnp.max(valid_taus, axis=0, keepdims=True)

        return jnp.sign(V_in) * jnp.maximum(abs_V - final_taus, 0.0)

    return jnp.where(lmbda == 0, V, apply_prox(V, lmbda))


def grad_fn_fast(Theta, A, B, D, lambd, alpha):
    ridge_penalty = lambd * (1.0 - alpha) * Theta
    ridge_penalty = ridge_penalty.at[0, 0].set(0.0)

    # Base gradient calculation
    grad = A @ Theta @ B - D + ridge_penalty
    return grad


def loss_fn(Theta, Y, C, X, N, lambd, alpha):
    # FOR THE ELASTIC NET PENALTY
    reconstruction_error = (1 / (2 * N)) * jnp.linalg.norm(
        Y - C @ Theta @ X, "fro"
    ) ** 2

    theta_penalty = Theta.at[0, 0].set(0.0)  # Do not penalize the intercept

    penalty = lambd * (
        alpha * jnp.sum(jnp.abs(theta_penalty))
        + ((1 - alpha) / 2) * jnp.linalg.norm(theta_penalty, "fro") ** 2
    )

    return reconstruction_error + penalty


class BSR:
    def __init__(
        self,
        X: np.array,
        Y: np.array,
        C: np.array,
        lambd: float,
        alpha: float,
        max_iter=50_000,
        tol=5e-5,
        dtype=jnp.float32,
        main_effects_C=True,
        main_effects_X=True,
        main_effects_only=False,
    ):
        self.dtype = dtype
        # Transform to jax arrays
        self.X = jnp.array(X)
        self.Y = jnp.array(Y)
        self.C = jnp.array(C)
        self.lambd = jnp.float32(lambd)
        self.alpha = jnp.float32(alpha)

        self.max_iter = max_iter
        self.tol = tol
        self.main_effects_only = main_effects_only

        # Including intercepts:
        n = self.C.shape[0]  # number of samples
        p = self.X.shape[1]  # number of features / TFs

        # Add main-effect column to C
        if main_effects_C:
            self.C = jnp.hstack([jnp.ones((n, 1), dtype=dtype), self.C])
        # Add main-effect row to X
        if main_effects_X:
            self.X = jnp.vstack([jnp.ones((1, p), dtype=dtype), self.X])

        self.Theta = jnp.zeros((self.C.shape[1], self.X.shape[0]), dtype=dtype)
        self.N = self.Y.size

        # Enforce type
        self.C = self.C.astype(dtype)
        self.X = self.X.astype(dtype)
        self.Y = self.Y.astype(dtype)
        self.lambd = self.lambd.astype(dtype)
        self.alpha = self.alpha.astype(dtype)

        self.L_init = (1 / self.N) * (
            jnp.linalg.norm(self.C, 2) ** 2 * jnp.linalg.norm(self.X, 2) ** 2
        ) + self.lambd * (1 - self.alpha)

        self._grad = jax.jit(
            partial(
                grad_fn_fast,
                lambd=self.lambd,
                alpha=self.alpha,
            )
        )

        self._loss = jax.jit(
            partial(
                loss_fn,
                Y=self.Y,
                C=self.C,
                X=self.X,
                N=self.N,
                lambd=self.lambd,
                alpha=self.alpha,
            )
        )

    def loss(self, Theta):
        return self._loss(Theta)

    def _soft_thresholding(self, Z, threshold):
        return jnp.sign(Z) * jnp.maximum(jnp.abs(Z) - threshold, 0)
    
    def _prox_step(self, G, J, thresh):
        """Shared proximal-step logic, factored out so both the monitored
        and unmonitored loop bodies stay in sync."""
       
        Theta_new = self._soft_thresholding(J, thresh)
        # Do not penalize intercept
        Theta_new = Theta_new.at[0, 0].set(G[0, 0])
        if self.main_effects_only:
            Theta_new = Theta_new.at[1:, 1:].set(0.0)
        return Theta_new
    
    def _fista_loop(self, Theta0, step, thresh, A, B, D, monitor=False, monitor_every=100):

        def cond_fn(state):
            k, err = state[3], state[4]
            return jnp.logical_and(k < self.max_iter, err > self.tol)

        def step_core(Theta, Q, t):
            grad_Q = self._grad(Q, A, B, D)
            G = Q - step * grad_Q
            J = G.at[0, 0].set(0.0)
            Theta_new = self._prox_step(G, J, thresh)

            t_new = (1.0 + jnp.sqrt(1.0 + 4.0 * t * t)) / 2.0
            Q_new = Theta_new + ((t - 1.0) / t_new) * (Theta_new - Theta)

            err_new = jnp.max(jnp.abs(Theta_new - Theta)) / jnp.max(
                jnp.abs(Theta) + 1e-8
            )
            return Theta_new, Q_new, t_new, err_new

        if not monitor:
            def body_fn(state):
                Theta, Q, t, k, err = state
                Theta_new, Q_new, t_new, err_new = step_core(Theta, Q, t)
                return Theta_new, Q_new, t_new, k + 1, err_new

            init_state = (Theta0, Theta0, 1.0, 0, jnp.inf)
            Theta, Q, t, k, err = jax.lax.while_loop(cond_fn, body_fn, init_state)
            return Theta, k, err, None, None, 0

        # ---- Monitored loop: extra history arrays carried in the state ----
        max_history = self.max_iter // monitor_every + 2

        def body_fn(state):
            Theta, Q, t, k, err, err_hist, loss_hist, hist_idx = state
            Theta_new, Q_new, t_new, err_new = step_core(Theta, Q, t)

            should_log = (k % monitor_every) == 0

            def do_log(carry):
                err_hist, loss_hist, hist_idx = carry
                loss_val = self._loss(Theta_new)  # only computed on logged iters
                err_hist = jax.lax.dynamic_update_slice_in_dim(
                    err_hist, err_new[None], hist_idx, axis=0
                )
                loss_hist = jax.lax.dynamic_update_slice_in_dim(
                    loss_hist, loss_val[None], hist_idx, axis=0
                )
                return err_hist, loss_hist, hist_idx + 1

            def skip_log(carry):
                return carry

            err_hist, loss_hist, hist_idx = jax.lax.cond(
                should_log, do_log, skip_log, (err_hist, loss_hist, hist_idx)
            )

            return Theta_new, Q_new, t_new, k + 1, err_new, err_hist, loss_hist, hist_idx

        init_state = (
            Theta0,
            Theta0,
            1.0,
            0,
            jnp.inf,
            jnp.full((max_history,), jnp.nan, dtype=self.dtype),
            jnp.full((max_history,), jnp.nan, dtype=self.dtype),
            0,
        )
        Theta, Q, t, k, err, err_hist, loss_hist, hist_idx = jax.lax.while_loop(
            cond_fn, body_fn, init_state
        )
        return Theta, k, err, err_hist, loss_hist, hist_idx

    def fit(self, monitor=False, monitor_every=100):
        step = 1.0 / self.L_init
        thresh = self.lambd * self.alpha * step

        print("Pre-calculating arrays...")
        A = (self.C.T @ self.C) / self.N
        B = self.X @ self.X.T
        D = (self.C.T @ self.Y @ self.X.T) / self.N
        # assert that the types are correct
        assert self.Theta.dtype == self.dtype, f"Theta dtype {self.Theta.dtype} does not match expected dtype {self.dtype}"
        assert A.dtype == self.Theta.dtype, f"A dtype {A.dtype} does not match Theta dtype {self.Theta.dtype}"
        assert B.dtype == self.Theta.dtype, f"B dtype {B.dtype} does not match Theta dtype {self.Theta.dtype}"
        assert D.dtype == self.Theta.dtype, f"D dtype {D.dtype} does not match Theta dtype {self.Theta.dtype}"

        print("Starting FISTA...")
        init_time = time.time()
        Theta, n_iter, err, err_hist, loss_hist, hist_len = self._fista_loop(
            self.Theta, step, thresh, A=A, B=B, D=D,
            monitor=monitor, monitor_every=monitor_every,
        )
        Theta = jnp.where(jnp.abs(Theta) < 1e-6, 0.0, Theta)
        elapsed = time.time() - init_time
        self.Theta = Theta

        if monitor:
            hist_len = int(hist_len)
            self.err_history = np.asarray(err_hist[:hist_len])
            self.loss_history = np.asarray(loss_hist[:hist_len])

        sparsity = jnp.mean(Theta == 0)
        msg = (
            f"Finished in {n_iter} iterations and {elapsed:.2f} seconds ({n_iter/elapsed:.2f} iter per sec), "
            f"final normalized norm change = {err:.3e}, sparsity = {sparsity:.5%}"
        )
        print(msg)

        return Theta

    def _get_diagonal_hessian(self):
        Theta_flat = self.Theta.flatten(order="F")
        active = jnp.flatnonzero(Theta_flat)
        k = len(active)

        if k == 0:
            return None

        p, _q = self.Theta.shape
        rows = active % p
        cols = active // p

        ridge_val = self.lambd * (1 - self.alpha)

        C_col_norms = jnp.sum(self.C**2, axis=0) / self.N
        X_row_norms = jnp.sum(self.X**2, axis=1)

        diag_h = (C_col_norms[rows] * X_row_norms[cols]) + ridge_val

        # Remove ridge penalty from the intercept (0,0) entry
        is_intercept = (rows == 0) & (cols == 0)
        diag_h = jnp.where(is_intercept, diag_h - ridge_val, diag_h)

        return diag_h

    def _get_full_hessian(self):
        Theta_flat = self.Theta.flatten(order="F")
        active = jnp.flatnonzero(Theta_flat)
        k = len(active)
        if k == 0:
            return None

        p, _q = self.Theta.shape
        rows = active % p
        cols = active // p

        # Pre-calculate components for both cases
        ridge_val = self.lambd * (1 - self.alpha)

        C_T_C = (self.C.T @ self.C) / self.N
        X_X_T = self.X @ self.X.T

        H = C_T_C[rows[:, None], rows] * X_X_T[cols[:, None], cols]

        # Add Ridge
        H = H + jnp.eye(k) * ridge_val
        is_intercept = (rows == 0) & (cols == 0)

        if jnp.any(is_intercept):
            idx = jnp.argmax(is_intercept)
            H = H.at[idx, idx].add(-ridge_val)

        return H

    def get_hessian(self):
        nonzero = jnp.sum(self.Theta != 0)
        if nonzero > 50_000:
            warnings.warn(
                "Theta is too large for full Hessian computation. Returning diagonal Hessian instead.",
                UserWarning,
            )
            return self._get_diagonal_hessian()
        else:
            return self._get_full_hessian()

    def predict(self, C, X, apdapt_shape=True):
        if jnp.sum(self.Theta != 0) == 0:
            warnings.warn(
                "Model has not been fit yet or some problem arised and parameters are at initialization state. Predictions may be meaningless.",
                UserWarning,
            )

        C = jnp.array(C, dtype=self.Theta.dtype)
        X = jnp.array(X, dtype=self.Theta.dtype)

        if apdapt_shape:
            n = C.shape[0]
            p = X.shape[1]

            C = jnp.hstack([jnp.ones((n, 1), dtype=C.dtype), C])
            X = jnp.vstack([jnp.ones((1, p), dtype=X.dtype), X])

        Y_hat = C @ self.Theta @ X
        return Y_hat
