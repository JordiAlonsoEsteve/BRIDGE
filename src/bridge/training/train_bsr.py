import os

import jax.numpy as jnp
import numpy as np
import pandas as pd

from bridge.data_loading.generate_data import generate_data
from bridge.hyperparam_tuning.hyperparameter_tuning import find_hyperparameters
from bridge.model.bsr import BSR
from bridge.split.split import structured_train_test_split
from bridge.utils.utils import (
    compute_metrics,
    get_parameter_importance,
    get_scale_distribution,
    get_theta_interpretation,
    handle_theta_names,
)

# Configuration for all pipeline runs
DATASET_CONFIGS = [
    {
        # Arabidopsis Datasets
        "names": [
            "PTI_TPM",
            "leaf_CPTT",
            "all_hormone_TPM",
            "PTI_log2FC",
            "all_hormone_log2FC",
        ],
        "dna_folder": "data/raw/dna_tair10",
        "results_dir": "results/ath",
    },
    # Another arabidopsis dataset but that is mapped using Araport11 ...
    {
        "names": ["PB_snRNA_hormone_log2FC", "PB_snRNA_hormone_CPTT"],
        "dna_folder": "data/raw/dna_araport11",
        "results_dir": "results/ath",
    },
    # Human Datasets
    {
        "names": ["PMBC_CPTT", "TabulaSapiens", "GTX", "TCGA"],
        "dna_folder": "data/raw/dna_human38",
        "results_dir": "results/human",
    },
]

DNA_PARAMS = {"k": 6, "bp": (500, 500), "region": "TSS"}


def run_pipeline(config: dict, main_effects_only: bool = False) -> None:
    """Executes data loading, tuning, model fitting, and evaluation for a given configuration.

    Parameters
    ----------
    config : dict
        Dataset collection configuration parameters.
    main_effects_only : bool, default=False
        If True, restricts BSR to fitting main effects without interaction terms.
    """
    for treatment in config["names"]:
        output_dir = (
            os.path.join(config["results_dir"], treatment)
            if not main_effects_only
            else os.path.join(config["results_dir"] + "_main_only", treatment)
        )
        # if train_metrics.csv exists, skip this treatment (we already ran it)
        if os.path.exists(os.path.join(output_dir, "train_metrics.csv")):
            print(
                f"Skipping {treatment} (main_effects_only={main_effects_only}) as it was already ran. To re-run, delete the directory: {output_dir}"
            )
            continue
        os.makedirs(output_dir, exist_ok=True)
        print("Output directory created:", output_dir)
        print(f"\n{'='*20} Processing dataset: {treatment} {'='*20}")
        print(f"Main effects only mode: {main_effects_only}")

        # Load Data
        X, Y, C, row_groups = generate_data(
            dataset_name=treatment,
            DNA_folder=config["dna_folder"],
            DNA_params=DNA_PARAMS,
        )
        get_scale_distribution(
            Y, plot_path=os.path.join(output_dir, "scale_distribution.png")
        )

        # Train / Test Split
        (X_train, X_test, C_train, C_test, Y_train, Y_test, row_groups_train, _) = (
            structured_train_test_split(
                X=X,
                C=C,
                Y=Y,
                row_groups=row_groups,
                test_size=0.1,
                random_state=1234,
            )
        )

        # Integrity Checks
        assert Y_train.index.equals(
            C_train.index
        ), "Y_train and C_train indices are not in the same order"

        assert Y_train.columns.equals(
            X_train.columns
        ), "X_test and Y_test columns are not in the same order"

        # assert no same genes in train and test
        assert set(Y_train.index).isdisjoint(
            set(Y_test.index)
        ), "Train and test sets have overlapping genes"
        # 3. Hyperparameter Tuning
        if "filtered" not in config["dna_folder"]:
            study_name = f"{treatment}_study" + (
                "_main_only" if main_effects_only else ""
            )
        else:
            study_name = f"{treatment}_study_filtered" + (
                "_main_only" if main_effects_only else ""
            )

        best_hyper = find_hyperparameters(
            study_name=study_name,
            trials=30,
            X_train=X_train,
            Y_train=Y_train,
            C_train=C_train,
            gene_families=row_groups_train,
            max_iter=5000,
            main_effects_only=main_effects_only,
            tol=1e-3,
        )

        print(f"Best hyperparameters for {treatment}: {best_hyper}")
        with open(os.path.join(output_dir, "best_hyperparameters.txt"), "w") as f:
            f.writelines(f"{key}: {value}\n" for key, value in best_hyper.items())

        # Train Model on Split & Evaluate
        # We change the dtype to float64
        # to improve the reproducibility of the results
        BSR_model = BSR(
            X=X_train.values,
            Y=Y_train.values,
            C=C_train.values,
            max_iter=200_000,
            tol=1e-8,
            main_effects_only=main_effects_only,
            **best_hyper,
            dtype=jnp.float64,
        )
        _ = BSR_model.fit()

        # Save test predictions and metrics
        Y_hat = BSR_model.predict(C_test.values, X_test.values)
        metrics = compute_metrics(Y_test.values, Y_hat)
        print(f"Test metrics: {metrics}")

        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, "test_metrics.csv"), index=False
        )
        np.savez(
            os.path.join(output_dir, "test_predictions.npz"),
            Y_hat=Y_hat,
            Y_test=Y_test.values,
        )
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"))

        # Re-fit on Full Dataset
        BSR_full_model = BSR(
            X=X.values,
            Y=Y.values,
            C=C.values,
            max_iter=200_000,
            tol=1e-8,
            main_effects_only=main_effects_only,
            **best_hyper,
            dtype=jnp.float64,
        )
        Theta = BSR_full_model.fit()

        # Compute and Store Training Performance on the Full Dataset
        Y_hat_full = BSR_full_model.predict(C.values, X.values)
        train_metrics = compute_metrics(Y.values, Y_hat_full)
        print(f"Full dataset training metrics: {train_metrics}")

        pd.DataFrame([train_metrics]).to_csv(
            os.path.join(output_dir, "train_metrics.csv"), index=False
        )
        
        #np.savez(
        #    os.path.join(output_dir, "train_predictions.npz"),
        #    Y_hat=Y_hat_full,
        #    Y_true=Y.values,
        #)

        # Save active count stats
        num_active_motifs = (Theta[1:, :] != 0).any(axis=1).sum()
        num_active_tfs = (Theta[:, 1:] != 0).any(axis=0).sum()
        with open(os.path.join(output_dir, "num_active.txt"), "w") as f:
            f.write(f"Number of active TFs: {num_active_tfs}\n")
            f.write(f"Number of active motifs: {num_active_motifs}\n")

        # Interpretation & Feature Importance
        Theta_df, A_df, D_df, All_theta_df = get_theta_interpretation(
            Theta, C_train, X_train
        )
        Theta_df.to_csv(os.path.join(output_dir, "Theta.csv"))
        A_df.to_csv(os.path.join(output_dir, "A.csv"))
        D_df.to_csv(os.path.join(output_dir, "D.csv"))
        All_theta_df.to_csv(os.path.join(output_dir, "All_theta.csv"), index=True)

        names = handle_theta_names(
            Theta, X_train.index.to_list(), C_train.columns.to_list()
        )
        hessian = BSR_full_model.get_hessian()
        importance_df, importance_matrix = get_parameter_importance(
            hessian, Theta, names
        )

        importance_matrix_df = pd.DataFrame(
            importance_matrix[1:, 1:],
            columns=X_train.index.to_list(),
            index=C_train.columns.to_list(),
        )
        importance_matrix_df.to_csv(os.path.join(output_dir, "importance_matrix.csv"))
        importance_df.to_csv(
            os.path.join(output_dir, "parameter_importance.csv"), index=False
        )


if __name__ == "__main__":
    for main_effects_only in [False, True]:
        for config in DATASET_CONFIGS:
            run_pipeline(config, main_effects_only=main_effects_only)
