import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_mappings():
    """Loads and cleans the ID-to-symbol mappings for both organisms."""
    # Arabidopsis mapping
    ath_mapping = pd.read_csv("data/processed/regulation/Ath/Ath_id_mapping.csv")
    gene_to_symbol_ath = {
        str(k).lower(): min(str(v).split(","), key=len)
        for k, v in zip(ath_mapping["Gene_id"], ath_mapping["symbol"])
        if pd.notna(v) and str(v) not in ["nan", "None"]
    }

    # Human mapping
    human_mapping = pd.read_csv("data/processed/regulation/human/Human_id_mapping.csv")
    gene_to_symbol_human = dict(zip(human_mapping["Gene_id"], human_mapping["symbol"]))

    return gene_to_symbol_ath, gene_to_symbol_human


def extract_distributions(experiments):
    """Reads Theta matrices and extracts the active counts per TF and per k-mer."""
    kmers_per_tf = {}
    tfs_per_kmer = {}

    for base_dir, treatments in experiments:
        for treatment in treatments:
            treatment_dir = f"results/{base_dir}/{treatment}"

            # Handle the casing inconsistency (All_theta vs All_Theta)
            theta_path = f"{treatment_dir}/Theta.csv"
            if not os.path.exists(theta_path):
                print(f"Warning: {theta_path} not found. Skipping {treatment}.")
                continue

            theta = pd.read_csv(theta_path, index_col=0)

            kmers_per_tf[treatment] = (theta != 0).sum(axis=0)  # Columns (TFs)
            tfs_per_kmer[treatment] = (theta != 0).sum(axis=1)  # Rows (k-mers)

    return kmers_per_tf, tfs_per_kmer


def plot_combined_histograms(
    distributions, title_template, xlabel, output_filename, color_palette, mappings=None
):
    """
    Generates a multi-panel figure for the given distributions.
    If mappings are provided, it attempts to resolve the index names (for TFs).
    """
    treatments = list(distributions.keys())
    num_treatments = len(treatments)
    cols = 3
    rows = (num_treatments + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()

    # Unpack styling configurations
    bar_color, edge_color, arrow_color, hline_color, box_color = color_palette

    for idx, treatment in enumerate(treatments):
        dist = distributions[treatment]
        ax = axes[idx]

        sns.histplot(
            dist, bins=50, kde=False, color=bar_color, edgecolor=edge_color, ax=ax
        )

        # Annotate top 3
        top_3 = dist.nlargest(3).sort_values()

        xlims = ax.get_xlim()
        x_range = xlims[1] - xlims[0]

        # Staggered text coordinates to prevent overlap
        text_x_positions = [
            top_3.iloc[0] - (x_range * 0.12),
            top_3.iloc[1],
            top_3.iloc[2] + (x_range * 0.12),
        ]
        text_y_positions = [35, 45, 35]

        for rank, (name, x_val) in enumerate(top_3.items()):
            y_val = (dist == x_val).sum()
            y_target = min(y_val, 42)

            # Resolve TF names if mapping dictionaries are provided
            display_name = str(name)
            if mappings:
                ath_map, human_map = mappings
                if display_name.lower().startswith("at"):
                    display_name = ath_map.get(display_name.lower(), display_name)
                else:
                    display_name = human_map.get(display_name, display_name)

            ax.annotate(
                f"{display_name}\n({int(x_val)})",
                xy=(x_val, y_target),
                xytext=(text_x_positions[rank], text_y_positions[rank]),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.2),
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor=box_color,
                    alpha=0.85,
                    edgecolor="gray",
                    lw=0.5,
                ),
                fontsize=8,
                ha="center",
                va="center",
            )

        # Plot formatting
        ax.set_title(title_template.format(treatment), fontsize=12)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.set_ylim(0, 50)
        ax.axhline(y=1, color=hline_color, linestyle="--", linewidth=2)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_yticklabels(["0", "10", "20", "30", "40", "50+"])
        ax.grid(axis="y", alpha=0.75)

    # Hide unused subplots
    for idx in range(num_treatments, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"Saved: {output_filename}")


if __name__ == "__main__":
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
    experiments = [
        ("ath", ["PTI_TPM", "all_hormone_TPM", "leaf_CPTT", "PTI_log2FC", "all_hormone_log2FC", "PB_snRNA_hormone_log2FC", "PB_snRNA_hormone_CPTT"]),
         ("human", ["GTX", "TabulaSapiens", "PMBC_CPTT", "TCGA"])
    ]

    print("Loading mappings and extracting data...")
    gene_to_symbol_ath, gene_to_symbol_human = load_mappings()
    kmers_per_tf, tfs_per_kmer = extract_distributions(experiments)
    # re-name the keys in the distributions to their pretty names
    kmers_per_tf = {treatments_pretty_names.get(k, k): v for k, v in kmers_per_tf.items()}
    tfs_per_kmer = {treatments_pretty_names.get(k, k): v for k, v in tfs_per_kmer.items()}
    print("Generating 'Active k-mers per TF' plot...")
    plot_combined_histograms(
        distributions=kmers_per_tf,
        title_template="Number of Active kmers per TF for {}",
        xlabel="Number of Active kmers",
        output_filename="results/active_kmers_per_TF_distribution_combined.png",
        color_palette=("skyblue", "black", "red", "red", "wheat"),
        mappings=(
            gene_to_symbol_ath,
            gene_to_symbol_human,
        ),  # Apply name mappings for TFs
    )

    print("Generating 'Active TFs per k-mer' plot...")
    plot_combined_histograms(
        distributions=tfs_per_kmer,
        title_template="Number of Active TFs per kmer for {}",
        xlabel="Number of Active TFs",
        output_filename="results/active_TFs_per_kmer_distribution_combined.png",
        color_palette=("lightgreen", "black", "blue", "blue", "lightblue"),
        mappings=None,  # k-mers don't need gene symbol resolution
    )

    print("Done!")
