import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_train_metrics_1x3(
    dataset_configs: list[dict],
    save_path: str = "results/R2_train_performance.png",
):
    """Generates a 1x3 horizontal bar plot of Training R^2 metrics

    (Overall, Average Per Gene, Average Per Sample) for the full models.
    """
    records = []

    # 1. Gather training metrics from the full model directories
    for config in dataset_configs:
        results_dir = config["results_dir"]
        for treatment in config["names"]:
            train_path = os.path.join(results_dir, treatment, "train_metrics.csv")
            if not os.path.exists(train_path):
                print(f"Skipping {treatment}: train_metrics.csv not found.")
                continue

            metrics = pd.read_csv(train_path).iloc[0].to_dict()
            row = {"Dataset": treatment}
            for k, v in metrics.items():
                row[k] = v
            records.append(row)

    df_train = pd.DataFrame(records)
    if df_train.empty:
        print("No training metrics found.")
        return None, None

    # Apply your pretty names formatting
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

    df_train["Dataset"] = (
        df_train["Dataset"].map(treatments_pretty_names).fillna(df_train["Dataset"])
    )
    df_train["Dataset"] = pd.Categorical(
        df_train["Dataset"], categories=treatments_pretty_names.values(), ordered=True
    )
    df_train = df_train.sort_values("Dataset")
    df_train["Dataset"] = df_train["Dataset"].replace(
        {"TabulaSapiens": "TabulaSapiens_CPTT", "GTX": "GTX_TPM", "TCGA": "TCGA_TPM"}
    )

    # 2. Set up 1x3 plot aesthetics matching your test plot
    eval_modes = [
        ("Overall (Train)", "R2"),
        ("Average Per Gene (Train)", "average_R2_per_gene"),
        ("Average Per Sample (Train)", "average_R2_per_sample"),
    ]

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(nrows=1, ncols=len(eval_modes), figsize=(15, 5), sharey=True)

    for col_idx, (mode_label, col_name) in enumerate(eval_modes):
        ax = axes[col_idx]

        if col_name not in df_train.columns:
            continue

        sns.barplot(
            data=df_train,
            y="Dataset",
            x=col_name,
            color="#2B5C8F",  # Full model color
            ax=ax,
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_title(mode_label, fontsize=13)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel(r"$R^2$", fontsize=14, fontweight="bold")

        if col_idx == 0:
            ax.set_ylabel("Dataset", fontsize=12, fontweight="bold")
            ax.tick_params(axis="y", labelsize=9)
        else:
            ax.set_ylabel("")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training performance plot to: {save_path}")

    return fig, axes


if __name__ == "__main__":
    from bridge.training.train_bsr import DATASET_CONFIGS
    plot_train_metrics_1x3(DATASET_CONFIGS)