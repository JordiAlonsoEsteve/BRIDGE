import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_r2_metrics_comparison_1x3(
    comparison_df: pd.DataFrame,
    save_path: str = "results/r2_metrics_comparison_plot_1x3.png",
):
    """Generates a 1x3 horizontal grouped bar plot comparing Full Model vs. Main-Effects-Only
    
    specifically for the R^2 metric across Overall, Per-Gene, and Per-Sample levels.
    """
    if comparison_df.empty:
        print("Dataframe is empty. Skipping plot generation.")
        return None, None

    # Check if the required baseline R2 column exists
    if "R2_Full" not in comparison_df.columns:
        print("The required metric 'R2_Full' was not found in the DataFrame.")
        return None, None

    # Evaluation levels mapping (Title, DataFrame Column Base Name)
    eval_modes = [
        ("Overall", "R2"),
        ("Average Per Gene", "average_R2_per_gene"),
        ("Average Per Sample", "average_R2_per_sample"),
    ]

    # Set up Seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(eval_modes),
        figsize=(5 * len(eval_modes), 5),  # 15x5 figure
        sharey=True,  # Share Y-axis so dataset labels only appear on the far left
    )

    for col_idx, (mode_label, base_col) in enumerate(eval_modes):
        ax = axes[col_idx]

        # Construct the specific Full and Main columns
        full_col = f"{base_col}_Full"
        main_col = f"{base_col}_MainOnly"

        # Check if columns exist; skip subplot if missing
        if main_col not in comparison_df.columns or full_col not in comparison_df.columns:
            print(f"Warning: Missing columns for {base_col}. Skipping.")
            continue

        # Prepare data in long-format for Seaborn grouped barplot
        plot_data = []
        for _, row in comparison_df.iterrows():
            plot_data.append(
                {
                    "Dataset": row["Dataset"],
                    "Model": "Restricted",
                    "Score": row[main_col],
                }
            )
            plot_data.append(
                {
                    "Dataset": row["Dataset"],
                    "Model": "Full",
                    "Score": row[full_col],
                }
            )

        df_metric = pd.DataFrame(plot_data)

        # Draw horizontal grouped bar plot (y="Dataset", x="Score")
        sns.barplot(
            data=df_metric,
            y="Dataset",
            x="Score",
            hue="Model",
            palette=["#C77FB5", "#2B5C8F"],
            ax=ax,
            edgecolor="black",
            linewidth=0.8,
        )

        # Subplot Title & Vertical reference line at 0
        ax.set_title(mode_label, fontsize=13)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

        # X-axis label for Score ($R^2$)
        ax.set_xlabel(r"$R^2$", fontsize=14, fontweight="bold")

        # Y-axis handling (Dataset labels)
        if col_idx == 0:
            ax.set_ylabel("Dataset", fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", labelsize=9)
        else:
            ax.set_ylabel("")

        # --- LEGEND HANDLING ---
        # Place legend on the last (rightmost) subplot
        if col_idx == len(eval_modes) - 1:
            ax.legend(title="Model Type", frameon=True, loc="best", fontsize=11)
        else:
            # Remove Seaborn's auto-generated legend from other subplots
            if ax.get_legend() is not None:
                ax.get_legend().remove()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved R2 metrics comparison plot to: {save_path}")

    return fig, axes

def generate_metrics_comparison(
    dataset_configs: list[dict],
    base_results_dir: str = "results",
    main_only_subfolder: str = "main_only",
    suffix: str = "",
) -> pd.DataFrame:
    """Reads test_metrics.csv files from full model runs and main-effects-only runs,

    then computes absolute and percentage differences.

    Parameters
    ----------
    dataset_configs : list of dict
        The configuration list defining the datasets and their relative result paths.
    base_results_dir : str
        Directory where results are stored.
    main_only_subfolder : str
        Suffix or subfolder name used to distinguish main-effects-only runs if saved separately.

    Returns
    -------
    pd.DataFrame
        A summary table comparing performance metrics across all datasets.
    """
    records = []

    for config in dataset_configs:
        results_dir = config["results_dir"]

        for treatment in config["names"]:
            # Path to Full Model Metrics
            full_path = os.path.join(results_dir, treatment, "test_metrics.csv")

            # Path to Main-Effects-Only Metrics
            # Adjust path depending on how you store main-effects runs (e.g., results/human_main_only/...)
            main_only_dir = results_dir + f"_{main_only_subfolder}"
            main_path = os.path.join(main_only_dir, treatment, "test_metrics.csv")

            if not os.path.exists(full_path):
                print(
                    f"Skipping {treatment}: Full model metrics not found at {full_path}"
                )
                continue
            if not os.path.exists(main_path):
                print(
                    f"Skipping {treatment}: Main-effects metrics not found at {main_path}"
                )
                continue

            full_metrics = pd.read_csv(full_path).iloc[0].to_dict()
            main_metrics = pd.read_csv(main_path).iloc[0].to_dict()

            # Build comparison row
            row = {
                "Group": os.path.basename(results_dir),
                "Dataset": treatment,
            }

            # Calculate differences for key metrics (e.g., Pearson, R2, RMSE)
            for metric in full_metrics:
                val_full = full_metrics[metric]
                val_main = main_metrics[metric]

                row[f"{metric}_Full"] = val_full
                row[f"{metric}_MainOnly"] = val_main

                # Absolute Delta: Full - Main
                diff = val_full - val_main
                row[f"{metric}_Diff"] = diff

                # Relative Percentage Gain (%)
                if val_main != 0:
                    pct_change = (diff / abs(val_main)) * 100
                    row[f"{metric}_Gain_%"] = round(pct_change, 2)

            records.append(row)

    comparison_df = pd.DataFrame(records)

    if not comparison_df.empty:
        summary_path = os.path.join(base_results_dir, f"metrics_comparison_summary{suffix}.csv")
        os.makedirs(base_results_dir, exist_ok=True)
        comparison_df.to_csv(summary_path, index=False)
        print(f"\nComparison completed successfully! Saved summary to: {summary_path}")

    return comparison_df

def print_metric_highlights(comparison_df: pd.DataFrame, metric_name: str = "Pearson"):
    """Prints a clear summary highlighting performance differences for a target metric."""
    if comparison_df.empty or f"{metric_name}_Gain_%" not in comparison_df.columns:
        return

    cols = [
        "Dataset",
        f"{metric_name}_MainOnly",
        f"{metric_name}_Full",
        f"{metric_name}_Diff",
        f"{metric_name}_Gain_%",
    ]
    summary = comparison_df[cols].copy()
    print(f"\n--- {metric_name} Performance Comparison ---")
    print(summary.to_string(index=False))
    # save to latex
    latex_path = os.path.join("results", f"{metric_name}_comparison_summary.tex")
    summary.to_latex(latex_path, index=False)


if __name__ == "__main__":
    from bridge.training.train_bsr import DATASET_CONFIGS
    treatments_pretty_names = {
        "PTI_TPM": "PTI (TPM)",
        "all_hormone_TPM": "Hormone (TPM)",
        "leaf_CPTT": "Leaf (CPTT)",
        "PTI_log2FC": r"PTI ($\log_2$FC)",
        "all_hormone_log2FC": r"Hormone ($\log_2$FC)",
        "PB_snRNA_hormone_log2FC": r"SN Hormone ($\log_2$FC)",
        "PB_snRNA_hormone_CPTT": "SN Hormone (CPTT)",
        "GTX": "GTEx (TPM)",
        "TabulaSapiens": "Tabula Sapiens (CPTT)",
        "PMBC_CPTT": "PBMC (CPTT)",
        "TCGA": "TCGA (TPM)",
    }

    df_comp = generate_metrics_comparison(DATASET_CONFIGS)
    df_comp["Dataset"] = df_comp["Dataset"].map(treatments_pretty_names).fillna(df_comp["Dataset"])
    # keep the original order of datasets as in treatments_pretty_names
    df_comp["Dataset"] = pd.Categorical(df_comp["Dataset"], categories=treatments_pretty_names.values(), ordered=True)
    df_comp = df_comp.sort_values("Dataset")
    # dataset TabulaSapiens has to be TabulaSapiens_CPTT
    df_comp["Dataset"] = df_comp["Dataset"].replace({"TabulaSapiens": "TabulaSapiens_CPTT"})
    df_comp["Dataset"] = df_comp["Dataset"].replace({"GTX": "GTX_TPM"})
    df_comp["Dataset"] = df_comp["Dataset"].replace({"TCGA": "TCGA_TPM"})

    #print_metric_highlights(df_comp, metric_name="Pearson")
    print_metric_highlights(df_comp, metric_name="R2")

    plot_r2_metrics_comparison_1x3(
        df_comp,
        save_path="results/R_2_comparison.png",
    )
