import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    train_test_split,
)


def structured_train_test_split(
    X: pd.DataFrame,
    C: pd.DataFrame,
    Y: pd.DataFrame,
    row_groups: np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int = 1234,
):

    assert Y.index.equals(C.index), "C must have the same row index as Y (genes)."

    # Y columns must match X columns (samples)
    assert Y.columns.equals(X.columns), "X must have the same columns as Y (samples)."

    assert set(X.index).isdisjoint(
        set(Y.index)
    ), "X index (TFs) and Y index (genes) must be disjoint."

    n_genes, n_samples = Y.shape

    # -----------------------
    # Row split
    # -----------------------
    if row_groups is None or len(np.unique(row_groups)) == 1:
        train_row_idx, test_row_idx = train_test_split(
            np.arange(n_genes), test_size=test_size, random_state=random_state
        )
    else:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_row_idx, test_row_idx = next(
            splitter.split(np.arange(n_genes), groups=row_groups)
        )

    # -----------------------
    # Column split
    # -----------------------
    train_col_idx, test_col_idx = train_test_split(
        np.arange(n_samples), test_size=test_size, random_state=random_state
    )

    # Deterministic ordering
    train_row_idx = np.sort(train_row_idx)
    test_row_idx = np.sort(test_row_idx)
    train_col_idx = np.sort(train_col_idx)
    test_col_idx = np.sort(test_col_idx)

    # -----------------------
    # Subset DataFrames
    # -----------------------

    # Y is genes x samples
    Y_train = Y.iloc[train_row_idx, train_col_idx]
    Y_test = Y.iloc[test_row_idx, test_col_idx]

    # X is features x samples
    X_train = X.iloc[:, train_col_idx]
    X_test = X.iloc[:, test_col_idx]

    # C is genes x features (or gene covariates)
    C_train = C.iloc[train_row_idx]
    C_test = C.iloc[test_row_idx]

    # -----------------------
    # Subset groups
    # -----------------------
    row_groups_train = row_groups[train_row_idx] if row_groups is not None else None
    row_groups_test = row_groups[test_row_idx] if row_groups is not None else None

    return (
        X_train,
        X_test,
        C_train,
        C_test,
        Y_train,
        Y_test,
        row_groups_train,
        row_groups_test,
    )


def structured_cv_split(
    Y,
    n_splits=5,
    row_groups=None,
    random_state=1234,
):

    n_genes, n_samples = Y.shape
    folds = []
    shuffle = True  # Always shuffle for CV to ensure randomness
    # --- Row splitter ---
    if row_groups is None or len(np.unique(row_groups)) == 1:
        row_kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        row_splits = list(row_kf.split(np.arange(n_genes)))
    else:
        row_gkf = GroupKFold(n_splits=n_splits)
        row_splits = list(row_gkf.split(np.arange(n_genes), groups=row_groups))

    # --- Column splitter ---
    col_kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    col_splits = list(col_kf.split(np.arange(n_samples)))

    # --- Combine row and column splits ---
    for fold_idx in range(n_splits):
        train_row_idx, val_row_idx = row_splits[fold_idx]
        train_col_idx, val_col_idx = col_splits[fold_idx]

        folds.append(((train_row_idx, train_col_idx), (val_row_idx, val_col_idx)))

    return folds
