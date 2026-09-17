import gc
import os

import jax
import numpy as np
import optuna

from bridge.model.bsr import BSR
from bridge.split.split import structured_cv_split
from bridge.utils.utils import compute_metrics


def cross_validation_pass(
    hyperparams: dict,
    X_train: np.array,
    Y_train: np.array,
    C_train: np.array,
    gene_families=None,
    n_splits=5,
):
    """Perform structured cross-validation."""
    # Generate structured CV folds
    folds = structured_cv_split(
        Y=Y_train,
        n_splits=n_splits,
        row_groups=gene_families,
        random_state=1234,
    )

    # Initialize fold-wise metric storage
    fold_metrics = []

    for fold_idx, ((train_rows, train_cols), (val_rows, val_cols)) in enumerate(folds):
        # --- Subset training and validation data ---
        X_tr_cv = X_train[:, train_cols].copy()
        Y_tr_cv = Y_train[np.ix_(train_rows, train_cols)].copy()
        C_tr_cv = C_train[train_rows, :].copy()

        X_val = X_train[:, val_cols].copy()
        Y_val = Y_train[np.ix_(val_rows, val_cols)].copy()
        C_val = C_train[val_rows, :].copy()

        # Initialize model with sampled hyperparameters
        model = BSR(X=X_tr_cv, Y=Y_tr_cv, C=C_tr_cv, **hyperparams)

        # --- Fit model ---
        theta = model.fit()

        # If all 0s in theta, skip prediction and assign worst metrics
        if np.all(theta == 0):
            fallback_metrics = {
                "MSE": np.inf,
                "RMSE": np.inf,
                "R2": -np.inf,
                "Pearson": -1.0,
                "Spearman": -1.0,
                "Sparsity": 1.0,  # Fully sparse if all zeros
            }
            # Instead of returning immediately and breaking the expected dict shape,
            # we fill the remaining folds or return the fallback format.
            return fallback_metrics

        # --- Predict and evaluate ---
        Y_val_pred = model.predict(C=C_val, X=X_val)
        metrics = compute_metrics(Y_val, Y_val_pred)
        sparsity = np.mean(theta == 0)
        metrics["Sparsity"] = sparsity
        fold_metrics.append(metrics)

        del model
        gc.collect()
        jax.clear_caches()

    # Compute the average metrics across folds
    avg_metrics = {
        metric: np.mean([fold[metric] for fold in fold_metrics])
        for metric in fold_metrics[0]
    }

    return avg_metrics


def find_hyperparameters(
    study_name,
    trials,
    X_train,
    Y_train,
    C_train,
    gene_families,
    max_iter=1000,
    main_effects_only=False,
    tol=5e-5,
    ll=1e-4,
    lu=2,
    la=0,
    ua=1.0,
):
    def objective(trial):
        # Sample hyperparameters
        lambd = trial.suggest_float("lambd", ll, lu, log=True)
        alpha = trial.suggest_float("alpha", la, ua)

        # Perform cross-validation and return the average metric
        avg_metrics = cross_validation_pass(
            hyperparams={
                "lambd": lambd,
                "alpha": alpha,
                "max_iter": max_iter,
                "tol": tol,
                "main_effects_only": main_effects_only,
            },
            X_train=X_train.values,
            Y_train=Y_train.values,
            C_train=C_train.values,
            gene_families=gene_families,
            n_splits=3,
        )

        trial.set_user_attr("Sparsity", float(avg_metrics.get("Sparsity", 0.0)))

        return avg_metrics["Pearson"]

    sampler = optuna.samplers.TPESampler(n_startup_trials=5, multivariate=True)
    os.makedirs("./results/hyper_tuning", exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///./results/hyper_tuning/{study_name}.db",
        load_if_exists=True,
        sampler=sampler,
    )

    completed_trials = len(study.trials)
    remaining_trials = trials - completed_trials
    if remaining_trials > 0:
        study.optimize(objective, n_trials=remaining_trials)
        print(f"Completed {trials} trials for study '{study_name}'.")
    else:
        print(
            f"Study '{study_name}' already has {completed_trials} completed trials. No new trials will be run."
        )
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not complete_trials:
        raise ValueError("No completed trials found to process.")

    gc.collect()
    jax.clear_caches()
    return study.best_params
