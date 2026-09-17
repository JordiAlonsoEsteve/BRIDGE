import os
import shutil

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def parse_motif_df_from_meme(meme_file, motif_id):
    """Parses letter probability matrix for a specific motif ID from a MEME file."""
    with open(meme_file, "r") as f:
        lines = f.readlines()

    matrix_data = []
    found_motif = False
    in_matrix = False

    for line in lines:
        if line.startswith(f"MOTIF {motif_id}"):
            found_motif = True
            continue

        if found_motif and line.strip().startswith("letter-probability matrix"):
            in_matrix = True
            continue

        if in_matrix:
            line = line.strip()
            if not line or line.startswith(("MOTIF", "URL")):
                if len(matrix_data) > 0:
                    break
                continue

            try:
                row = [float(x) for x in line.split()]
                if len(row) > 0:
                    matrix_data.append(row)
            except ValueError:
                continue

    if not matrix_data:
        print(f"Warning: Could not parse matrix for {motif_id} in {meme_file}")
        return None

    cols = ["A", "C", "G", "T"] if len(matrix_data[0]) == 4 else None
    return pd.DataFrame(matrix_data, columns=cols)

def reverse_complement_pwm(df):
    """Reverse complements a PWM DataFrame with columns A, C, G, T."""
    rc = df.iloc[::-1].reset_index(drop=True)
    rc = rc[["T", "G", "C", "A"]]
    rc.columns = ["A", "C", "G", "T"]
    return rc

def counts_to_info(df):
    """Standardizes PWM probabilities to Information Content (bits)."""
    prob_df = df.div(df.sum(axis=1), axis=0).fillna(0.25)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -(prob_df * np.log2(prob_df)).sum(axis=1)
    info_content = 2 - entropy
    return prob_df.multiply(info_content, axis=0)

def export_single_motif_pair(t_df, t_name, q_df, q_name, output_filename, pvalue, dataset):
    """Exports a single pair of motifs (Target vs Query) as a vector PDF."""
    # 1. Bump height slightly to 1.6 to fit both top title and bottom labels comfortably
    _fig, axes = plt.subplots(1, 2, figsize=(6, 1.6), sharey=True)

    # Target Motif (Experimental / Database)
    logo_t = logomaker.Logo(
        counts_to_info(t_df), ax=axes[0], color_scheme="classic"
    )
    logo_t.style_spines(visible=False)
    axes[0].spines["bottom"].set_visible(True)
    axes[0].spines["left"].set_visible(True)
    axes[0].set_ylim([0, 2.0])
    axes[0].set_yticks([0, 1, 2])

    # Query Motif (Inferred)
    logo_q = logomaker.Logo(
        counts_to_info(q_df), ax=axes[1], color_scheme="classic"
    )
    logo_q.style_spines(visible=False)
    axes[1].spines["bottom"].set_visible(True)
    axes[1].spines["left"].set_visible(False)
    axes[1].tick_params(left=False)

    # Set x-ticks based on each motif's actual length
    axes[0].set_xticks(range(len(t_df)))
    axes[1].set_xticks(range(len(q_df)))
    for ax in axes:
        ax.set_xticklabels([])

    # 2. Add motif names as x-axis labels below each plot
    axes[0].set_xlabel(t_name, fontsize=12, labelpad=6)
    axes[1].set_xlabel("BRIDGE-" +  q_name, fontsize=12, labelpad=6)

    # 3. Adjust top/bottom margins so neither the title nor the bottom labels are clipped
    plt.subplots_adjust(wspace=0.1, top=0.72, bottom=0.28)

    # Explicitly place suptitle above the top edge
    plt.suptitle(f"{treatments_pretty_names[dataset]} | adj. p-value: {pvalue:.2e}", fontsize=12, y=0.85)

    plt.savefig(
        output_filename, format="pdf", bbox_inches="tight", transparent=True, dpi=300
    )
    plt.close()

def get_top_n_matches(
    tomtom_df: pd.DataFrame, n: int = 5, match_mode: str = "all"
) -> pd.DataFrame:
    """Filters by mode (exact matches on mapped Gene IDs), calculates Bonferroni-

    corrected p-values based on subsection size, and returns top N matches.
    """
    valid_modes = ["all", "exact", "family"]
    if match_mode not in valid_modes:
        raise ValueError(
            f"Invalid match_mode '{match_mode}'. Choose from {valid_modes}."
        )

    df = tomtom_df.copy()

    # 1. Filter subsection
    if match_mode == "exact":
        # Compare mapped Gene/Locus IDs if available, fall back to raw IDs
        q_col = "Query_Gene_ID" if "Query_Gene_ID" in df.columns else "Query_ID"
        t_col = (
            "Target_Gene_ID" if "Target_Gene_ID" in df.columns else "Target_ID"
        )
        df = df[df[q_col] == df[t_col]]

    elif match_mode == "family":
        df = df[
            (df["Query_Family"] == df["Target_Family"])
            & (df["Query_Family"].notna())
            & (df["Target_Family"].notna())
        ]

    # 2. Calculate Bonferroni correction based on exact subsection size
    subsection_size = len(df)
    if subsection_size > 0 and "p-value" in df.columns:
        df["Subset_Size"] = subsection_size
        df["p-value_BF"] = (df["p-value"] * subsection_size).clip(upper=1.0)
    else:
        df["Subset_Size"] = 0
        df["p-value_BF"] = np.nan

    # 3. Sort by unadjusted p-value and return top N
    df_sorted = df.sort_values(by="p-value", ascending=True)
    return df_sorted.head(n)

def get_most_important_matches(
    tomtom_df: pd.DataFrame, 
    importances: pd.DataFrame, 
    n: int = 5,
    gene_mapping: dict | None = None
) -> pd.DataFrame:
    """
    Returns the most important motifs/TFs (max importance score per TF) and 
    indicates whether an exact match exists in the target JASPAR database.
    """
    df_imp = importances.copy()

    # 1. Parse Gene/TF ID from Parameter (filtering out intercept and non-motif parameters)
    if "Parameter" in df_imp.columns:
        df_imp = df_imp[df_imp["Parameter"].str.contains("-", na=False)].copy()
        df_imp["Gene"] = df_imp["Parameter"].apply(lambda x: str(x).split("-")[0])
    elif "Gene" not in df_imp.columns:
        df_imp["Gene"] = df_imp.index

    # 2. Calculate maximum importance score per TF and rank TFs
    tf_importance = (
        df_imp.groupby("Gene", as_index=False)["Importance"]
        .max()
        .sort_values(by="Importance", ascending=False)
    )
    if gene_mapping:
        tf_importance["Gene"] = tf_importance["Gene"].map(gene_mapping).fillna(tf_importance["Gene"])
    
    tomtom_df["Query_Gene_ID"] = tomtom_df["Query_Gene_ID"].str.upper()  # Ensure consistent casing for merging
    tomtom_df["Target_Gene_ID"] = tomtom_df["Target_Gene_ID"].str.upper()  # Ensure consistent casing for merging

    df_genes = set(tf_importance["Gene"].unique())
    tomtom_genes = set(tomtom_df["Query_Gene_ID"].dropna().unique())
    # we only care about the intersection, the rest might be just removed due deficient motifs (lack of information content)
    my_set = df_genes.intersection(tomtom_genes)
    # 3. Identify exact matches present in TOMTOM results (Query == Target)
    q_col = "Query_Gene_ID" if "Query_Gene_ID" in tomtom_df.columns else "Query_ID"
    t_col = "Target_Gene_ID" if "Target_Gene_ID" in tomtom_df.columns else "Target_ID"
    #t_name_col = "Target_Name" if "Target_Name" in tomtom_df.columns else t_col

    # Set of TFs that have an exact database match in JASPAR
    exact_matches = tomtom_df[tomtom_df[q_col] == tomtom_df[t_col]]
    exact_db_genes = set(exact_matches[q_col].dropna().unique())

    # Get lowest p-value TOMTOM match for each query motif (if available)
    sort_cols = [col for col in ["p-value", "q-value"] if col in tomtom_df.columns]
    if sort_cols:
        best_tomtom = tomtom_df.sort_values(by=sort_cols[0], ascending=True).groupby(q_col, as_index=False).first()
    else:
        best_tomtom = tomtom_df.groupby(q_col, as_index=False).first()
    
    # 4. Merge importance ranking with TOMTOM matching info
    #cols_to_keep = [q_col, t_name_col] + sort_cols
    merged = pd.merge(
        tf_importance,
        best_tomtom,
        left_on="Gene",
        right_on="Query_Gene_ID",
        how="left"
    )
    # remove elements not in my_set, i.e. those that are not in the intersection of the two sets
    merged = merged[merged["Gene"].isin(my_set)].copy()
    # 5. Flag database presence and novelty
    merged["in_jaspar_db"] = merged["Gene"].isin(exact_db_genes)
    merged["is_novel_candidate"] = ~merged["in_jaspar_db"]
    # rename q-value to p-value_BF
    merged.rename(columns={"q-value": "p-value_BF"}, inplace=True)
    # Clean up column layout
    #output_cols = ["Gene", "Importance", "in_jaspar_db", "is_novel_candidate", t_name_col] + sort_cols
    #existing_output_cols = [c for c in output_cols if c in merged.columns]
    # remove the ones that are in JASPAR
    merged = merged[~merged["in_jaspar_db"]].copy()
    return merged.sort_values(by="Importance", ascending=False).head(n)

if __name__ == "__main__":
    # Define dataset structure
    SPECIES_CONFIG = {
        "Ath": {
            "treatments": [
                "PTI_TPM",
                "all_hormone_TPM",
                "leaf_CPTT",
                "PTI_log2FC",
                "all_hormone_log2FC",
                "PB_snRNA_hormone_log2FC",
                "PB_snRNA_hormone_CPTT",
            ],
            "target_meme": "data/processed/regulation/Ath/JASPAR2026_CORE_plants_non-redundant_pfms_meme_uniprot.txt",
        },
        "human": {
            "treatments": ["TabulaSapiens", "GTX", "PMBC_CPTT", "TCGA"],
            "target_meme": "data/processed/regulation/human/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme_uniprot.txt",
        },
    }

    # Match modes to generate per treatment (and how many matches to grab)
    MATCH_CONFIG = {
        "all": 10,
        "exact": 10,
        "family": 10,
        "important": 10,  # Included feature importance mode
    }

    # Outer Loop: Species
    for species, cfg in SPECIES_CONFIG.items():
        target_meme_path = cfg["target_meme"]

        # Inner Loop: Treatments
        for treatment in cfg["treatments"]:
            base_dir = f"results/{species}/{treatment}"
            tomtom_path = f"{base_dir}/tomtom_df.csv"
            query_meme_path = f"{base_dir}/inferred_motifs.meme"
            importance_path = f"{base_dir}/parameter_importance.csv"

            if not os.path.exists(tomtom_path) or not os.path.exists(query_meme_path):
                print(f"Skipping [{species}/{treatment}]: File(s) missing.")
                continue

            print(f"\nProcessing [{species} / {treatment}]...")

            tomtom_df = pd.read_csv(tomtom_path)

            output_folder = f"{base_dir}/logo_plots"
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder, exist_ok=True)
            
            metadata_records = []

            # Category Loop: overall, exact, family, important
            for cat_name, n_matches in MATCH_CONFIG.items():
                cat_dir = os.path.join(output_folder, cat_name)
                os.makedirs(cat_dir, exist_ok=True)

                # Route query matching strategy based on category
                if cat_name == "important":
                    if not os.path.exists(importance_path):
                        print(f"  Skipping category [{cat_name}]: {importance_path} not found.")
                        continue
                    imp_df = pd.read_csv(importance_path)
                    if species == "human":
                        mapping = pd.read_csv("data/processed/regulation/human/Human_id_mapping.csv")
                        # generate a dict of Gene_id:symbol
                        gene_mapping = dict(zip(mapping["Gene_id"], mapping["symbol"]))
                    top_matches = get_most_important_matches(
                        tomtom_df, imp_df, n=n_matches, gene_mapping=gene_mapping if species == "human" else None
                    )
                else:
                    top_matches = get_top_n_matches(
                        tomtom_df, n=n_matches, match_mode=cat_name
                    )
                   

                # Use tqdm to track logo rendering and export per category
                pbar = tqdm(
                    enumerate(top_matches.iterrows(), start=1),
                    total=len(top_matches),
                    desc=f"  Generating PDF logos [{cat_name}]",
                    leave=False,
                )

                for rank, (_, row) in pbar:
                    q_id = row.get("Query_ID")
                    t_id = row.get("Target_ID")

                    q_gene = row.get("Query_Gene_ID", row.get("Gene", q_id))
                    t_gene = row.get("Target_Gene_ID", t_id)

                    q_name = (
                        str(row["Query_Name"])
                        if "Query_Name" in row and pd.notna(row["Query_Name"])
                        else str(q_gene)
                    )
                    t_name = (
                        str(row["Target_Name"])
                        if "Target_Name" in row and pd.notna(row["Target_Name"])
                        else str(t_id) if pd.notna(t_id) else "Unmatched"
                    )

                    orientation = row.get("Orientation", "+")
                    p_val = row.get("p-value", np.nan)
                    p_val_bf = row.get("p-value_BF", np.nan)
                    subset_size = row.get("Subset_Size", 0)
                    q_val = row.get("q-value", np.nan)
                    importance_score = row.get("Importance", np.nan)
                    in_db = row.get("in_jaspar_db", np.nan)
                    is_novel = row.get("is_novel_candidate", np.nan)

                    # Parse matrices
                    t_df = parse_motif_df_from_meme(target_meme_path, t_id) if pd.notna(t_id) else None
                    q_df = parse_motif_df_from_meme(query_meme_path, q_id) if pd.notna(q_id) else None

                    pdf_filename = None
                    if t_df is not None and q_df is not None:
                        # Reverse complement if aligned on negative strand
                        if orientation == "-":
                            q_df = reverse_complement_pwm(q_df)

                        # Sanitize filenames
                        safe_q = str(q_name).replace("/", "_").replace(" ", "_")
                        safe_t = str(t_name).replace("/", "_").replace(" ", "_")
                        pdf_filename = f"rank{rank:02d}_{safe_t}_vs_{safe_q}.pdf"
                        output_filepath = os.path.join(cat_dir, pdf_filename)

                        # Export individual PDF logo pair
                        export_single_motif_pair(
                            t_df, t_name, q_df, q_name, output_filepath, p_val_bf, dataset=treatment
                        )

                    # Log metadata regardless of alignment status
                    metadata_records.append({
                        "Category": cat_name,
                        "Rank": rank,
                        "Target_Name": t_name,
                        "Target_ID": t_id,
                        "Target_Gene_ID": t_gene,
                        "Query_Name": q_name,
                        "Query_ID": q_id,
                        "Query_Gene_ID": q_gene,
                        "Importance": importance_score,
                        "In_JASPAR_DB": in_db,
                        "Is_Novel_Candidate": is_novel,
                        "Orientation": orientation,
                        "p-value": p_val,
                        "p-value_BF": p_val_bf,
                        "Subset_Size": subset_size,
                        "q-value": q_val,
                        "PDF_Filename": pdf_filename,
                    })

            # Save metadata summary CSV per treatment
            if metadata_records:
                summary_df = pd.DataFrame(metadata_records)
                summary_csv_path = os.path.join(output_folder, "summary_metadata.csv")
                summary_df.to_csv(summary_csv_path, index=False)

            print(f"  Finished [{species} / {treatment}] -> {output_folder}")