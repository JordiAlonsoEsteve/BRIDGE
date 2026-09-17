import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def compute_metrics(Y_true, Y_pred):
    """
    Compute regression metrics between true and predicted values.
    Parameters:
    Y_true: array-like of shape (n_genes, n_samples)
        True gene expression values.
    Y_pred: array-like of shape (n_genes, n_samples)
        Predicted gene expression values.
    Returns:
    dict
        Dictionary containing overall metrics and metrics averaged over genes and samples.
        Keys: "MSE", "RMSE", "R2", "Pearson", "Spearman",
              "MSE_per_gene", "RMSE_per_gene", "R2_per_gene", "Pearson_per_gene", "Spearman_per_gene",
              "MSE_per_sample", "RMSE_per_sample", "R2_per_sample", "Pearson_per_sample", "Spearman_per_sample"
    """
    # assert dimensions are the same, both are 2D arrays
    assert Y_true.shape == Y_pred.shape, "Y_true and Y_pred must have the same shape."

    # Overall metrics (flattened)
    Y_true_flat = Y_true.flatten()
    Y_pred_flat = Y_pred.flatten()

    mse = np.mean((Y_true_flat - Y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (
        np.sum((Y_true_flat - Y_pred_flat) ** 2)
        / np.sum((Y_true_flat - np.mean(Y_true_flat)) ** 2)
    )
    pearson_corr, _ = pearsonr(Y_true_flat, Y_pred_flat)
    spearman_corr, _ = spearmanr(Y_true_flat, Y_pred_flat)

    # Per-gene metrics (axis=1, average over samples)
    mse_per_gene = np.mean((Y_true - Y_pred) ** 2, axis=1)
    rmse_per_gene = np.sqrt(mse_per_gene)
    r2_per_gene = np.maximum(
        0,
        1
        - (
            np.sum((Y_true - Y_pred) ** 2, axis=1)
            / np.sum((Y_true - np.mean(Y_true, axis=1, keepdims=True)) ** 2, axis=1)
        ),
    )
    pearson_per_gene = np.array(
        [pearsonr(Y_true[i], Y_pred[i])[0] for i in range(Y_true.shape[0])]
    )
    spearman_per_gene = np.array(
        [spearmanr(Y_true[i], Y_pred[i])[0] for i in range(Y_true.shape[0])]
    )

    # Average per-gene metrics to scalars
    mse_per_gene = np.nanmean(mse_per_gene)
    rmse_per_gene = np.nanmean(rmse_per_gene)
    r2_per_gene = np.nanmean(r2_per_gene)
    pearson_per_gene = np.nanmean(pearson_per_gene)
    spearman_per_gene = np.nanmean(spearman_per_gene)

    # Per-sample metrics (axis=0, average over genes)
    mse_per_sample = np.mean((Y_true - Y_pred) ** 2, axis=0)
    rmse_per_sample = np.sqrt(mse_per_sample)
    r2_per_sample = np.maximum(
        0,
        1
        - (
            np.sum((Y_true - Y_pred) ** 2, axis=0)
            / np.sum((Y_true - np.mean(Y_true, axis=0, keepdims=True)) ** 2, axis=0)
        ),
    )
    pearson_per_sample = np.array(
        [pearsonr(Y_true[:, i], Y_pred[:, i])[0] for i in range(Y_true.shape[1])]
    )
    spearman_per_sample = np.array(
        [spearmanr(Y_true[:, i], Y_pred[:, i])[0] for i in range(Y_true.shape[1])]
    )

    # Average per-sample metrics to scalars
    mse_per_sample = np.nanmean(mse_per_sample)
    rmse_per_sample = np.nanmean(rmse_per_sample)
    r2_per_sample = np.nanmean(r2_per_sample)
    pearson_per_sample = np.nanmean(pearson_per_sample)
    spearman_per_sample = np.nanmean(spearman_per_sample)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "average_MSE_per_gene": mse_per_gene,
        "average_RMSE_per_gene": rmse_per_gene,
        "average_R2_per_gene": r2_per_gene,
        "average_Pearson_per_gene": pearson_per_gene,
        "average_Spearman_per_gene": spearman_per_gene,
        "average_MSE_per_sample": mse_per_sample,
        "average_RMSE_per_sample": rmse_per_sample,
        "average_R2_per_sample": r2_per_sample,
        "average_Pearson_per_sample": pearson_per_sample,
        "average_Spearman_per_sample": spearman_per_sample,
    }


def get_theta_interpretation(Theta, C, X):
    Theta_interaction = Theta[1:, 1:]
    D = Theta_interaction @ X.values
    A = C.values @ Theta_interaction
    kmers = C.columns
    tf_names = X.index
    gene_names = C.index
    Theta_df = pd.DataFrame(Theta_interaction, index=kmers, columns=tf_names)
    A_df = pd.DataFrame(A, index=gene_names, columns=tf_names)
    D_df = pd.DataFrame(D, index=kmers, columns=X.columns)
    All_theta_df = pd.DataFrame(
        Theta, index=["intercept"] + list(kmers), columns=["intercept"] + list(tf_names)
    )
    return Theta_df, A_df, D_df, All_theta_df


def handle_theta_names(Theta: np.ndarray, TF_names: list, kmers: list):
    """Decompose nonzero Theta entries into four lists matching Fortran order

    used by the Hessian.

    Parameters
    ----------
    Theta : np.ndarray
        Shape can be (len(kmers)+1, len(TF_names)+1) or (len(kmers),
        len(TF_names))
    TF_names : list[str]
        Names of TFs (columns)
    kmers : list[str]
        Names of kmers (rows)

    Returns
    -------
    names: list[str]
    """
    n_rows, n_cols = Theta.shape
    has_main_effects = (n_rows == len(kmers) + 1) and (n_cols == len(TF_names) + 1)

    # 1. Match the Hessian code: flatten in Fortran order and get active indices
    Theta_flat = Theta.flatten(order="F")
    active = np.flatnonzero(Theta_flat)

    # 2. Reconstruct coordinates based on Fortran unrolling (column-major)
    rows = active % n_rows
    cols = active // n_rows

    names = []

    for i, j in zip(rows, cols):
        if has_main_effects:
            if i == 0 and j == 0:
                names.append("intercept")
            elif i == 0:
                # First row, column j -> Main TF effect
                names.append(TF_names[j - 1])
            elif j == 0:
                # First column, row i -> Main kmer effect
                names.append(kmers[i - 1])
            else:
                # Interaction terms
                names.append(f"{TF_names[j-1]}-{kmers[i-1]}")
        else:
            # Everything is interaction (no intercept or main effect offsets)
            names.append(f"{TF_names[j]}-{kmers[i]}")

    return names


def get_parameter_importance(Hessian, Theta, names):
    """
    Compute parameter importance scores and return:

    1. importance_df:
        DataFrame with nonzero coefficients and their importances

    2. importance_matrix:
        Matrix with same shape as Theta where each nonzero entry
        is replaced by its importance score.
    """

    # ---------------------------------------------------------
    # Flatten Theta in Fortran order
    # ---------------------------------------------------------
    theta_flat = Theta.flatten(order="F")

    # Indices of active coefficients
    nonzero_idx = np.flatnonzero(theta_flat)

    # Active coefficients
    betas = theta_flat[nonzero_idx]

    # ---------------------------------------------------------
    # Compute standard errors
    # ---------------------------------------------------------
    if Hessian.shape == betas.shape:
        se = jnp.sqrt(1 / Hessian)

    else:
        L = jsp.linalg.cholesky(Hessian, lower=True)

        # Solve L @ X = I
        L_inv = jsp.linalg.solve_triangular(
            L,
            jnp.eye(Hessian.shape[0]),
            lower=True,
        )

        # diag(H^{-1})
        var_diag = jnp.sum(jnp.square(L_inv), axis=0)

        se = jnp.sqrt(var_diag)

    # ---------------------------------------------------------
    # Importance
    # ---------------------------------------------------------
    importance = jnp.abs(betas / se)

    # ---------------------------------------------------------
    # DataFrame
    # ---------------------------------------------------------
    importance_df = pd.DataFrame(
        {
            "Parameter": names,
            "Beta": betas,
            "SE": se,
            "Importance": importance,
        }
    ).sort_values("Importance", ascending=False)

    # ---------------------------------------------------------
    # Reconstruct full importance matrix
    # ---------------------------------------------------------
    importance_flat = np.zeros_like(theta_flat, dtype=float)

    importance_flat[nonzero_idx] = np.asarray(importance)

    importance_matrix = importance_flat.reshape(
        Theta.shape,
        order="F",
    )

    return importance_df, importance_matrix


def get_scale_distribution(Y, plot_path=None):
    """
    Gets Y and calculates the std per gene (row) and column (sample), then it constructs
    a plot with the distributions and returns the averge std per gene and per sample.

    """

    std_per_gene = Y.std(axis=1)
    std_per_sample = Y.std(axis=0)

    if plot_path is not None:

        _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(std_per_gene, bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Standard Deviation per Gene")
        axes[0].set_xlabel("Standard Deviation")
        axes[0].set_ylabel("Frequency")

        sns.histplot(std_per_sample, bins="auto", kde=True, ax=axes[1])
        axes[1].set_title("Standard Deviation per Sample")
        axes[1].set_xlabel("Standard Deviation")
        axes[1].set_ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

    return std_per_gene.mean(), std_per_sample.mean()
