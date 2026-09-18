import os
import pandas as pd


def format_to_3_decimals(val):
    """Safely format numeric strings/floats to 3 decimal places."""
    if (
        pd.isna(val)
        or val is None
        or str(val).strip().upper() in ["N/A", "NAN", "NONE"]
    ):
        return "N/A"
    try:
        return f"{float(val):.3f}"
    except (ValueError, TypeError):
        return str(val)


def load_hyperparameters(file_path):
    """Load and parse the best hyperparameters text file into a dictionary."""
    if not os.path.exists(file_path):
        return {}

    df = pd.read_csv(file_path, sep=":", header=None, names=["Hyperparameter", "Value"])

    # Clean keys and values
    keys = df["Hyperparameter"].astype(str).str.strip().str.rstrip(":")
    values = df["Value"].astype(str).str.strip()
    return dict(zip(keys, values))


def extract_dataset_metrics(base_dir, treatment):
    """Extract matrix dimensions, sparsity, and active components from results."""
    treatment_dir = f"results/{base_dir}/{treatment}"

    # Handle the file casing inconsistency from the original code
    theta_path = f"{treatment_dir}/All_theta.csv"
    if not os.path.exists(theta_path):
        theta_path = f"{treatment_dir}/All_Theta.csv"

    theta = pd.read_csv(theta_path, index_col=0)
    D = pd.read_csv(f"{treatment_dir}/D.csv", index_col=0)
    A = pd.read_csv(f"{treatment_dir}/A.csv", index_col=0)

    sparsity = (theta == 0).mean().mean()
    sparsity_str = f"{sparsity * 100:.2f}\%" if sparsity <= 1.0 else f"{sparsity:.2f}"
    
    return {
        "Species": "Ath" if base_dir == "Ath" else "Human",
        "Samples": int(round(D.shape[1])),
        "Genes": int(round(A.shape[0])),
        "TFs": theta.shape[1],
        "Sparsity": sparsity_str,
        "Active TFs": int((theta.values[1:, 1:] != 0).any(axis=0).sum()),
        "Active Motifs": int((theta.values[1:, 1:]!= 0).any(axis=1).sum()),
        "Active intercepts TFs": int((theta.values[0, 1:] != 0).sum()),
        "Active intercepts Motifs": int((theta.values[1:, 0] != 0).sum()),

        
    }


def generate_summary_table():
    # Define the datasets and their corresponding base directories
    experiments = [
        ("ath", ["PTI_TPM", "all_hormone_TPM", "leaf_CPTT", "PTI_log2FC", "all_hormone_log2FC", "PB_snRNA_hormone_log2FC", \
"PB_snRNA_hormone_CPTT"]),
        ("human", ["GTX", "TabulaSapiens", "PMBC_CPTT", "TCGA"])
    ]

    table_rows = []

    for base_dir, treatments in experiments:
        for treatment in treatments:
            print(f"Processing {treatment}...")

            # Extract matrices characteristics
            metrics = extract_dataset_metrics(base_dir, treatment)

            # Extract hyperparameters
            hyp_dict = load_hyperparameters(
                f"results/{base_dir}/{treatment}/best_hyperparameters.txt"
            )
            alpha_val = hyp_dict.get("alpha", hyp_dict.get("Alpha", "N/A"))

            # Build the row dictionary
            row = {
                "Dataset": treatment.replace("_", r"\_"),
                "Species": metrics["Species"],
                "Samples": metrics["Samples"],
                "Genes": metrics["Genes"],
                "TFs": metrics["TFs"],
                "Sparsity": metrics["Sparsity"],
                "Active TFs": metrics["Active TFs"],
                "Active 6-mers": metrics["Active Motifs"],
                "Active intercepts TFs": metrics["Active intercepts TFs"],
                "Active intercepts 6-mers": metrics["Active intercepts Motifs"],
                r"$\alpha$": format_to_3_decimals(alpha_val),
                r"$\lambda$": format_to_3_decimals(hyp_dict.get("lambd")),
            }
            table_rows.append(row)

    # Convert to DataFrame
    df_summary = pd.DataFrame(table_rows)

    # Generate the polished LaTeX table code
    latex_code = df_summary.to_latex(
        index=False,
        column_format="lcccccccccc",  # 11 columns total
        caption="Summary of dataset characteristics, network structural dimensions, sparsity levels, and optimized regularized hyperparameters.",
        label="tab:dataset_characteristics",
        position="th",
        bold_rows=False,
        escape=False,  # CRITICAL: Keeps math syntax ($\alpha$) intact
    )

    # Save and output
    with open("results/summary_table.tex", "w") as f:
        f.write(latex_code)

    print("\n--- Generated Clean LaTeX Table Code ---")
    print(latex_code)


if __name__ == "__main__":
    generate_summary_table()
