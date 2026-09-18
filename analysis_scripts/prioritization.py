import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


def get_family_correlation(
    importances: pd.DataFrame,
    tomtom_results: pd.DataFrame,
    gene_to_uniprot_mapping: dict,
):
    """
    Computes correlation between importance and Tomtom p-values, but instead of 
    matching exact Query_ID to Target_ID, it finds the lowest p-value among all 
    targets in the SAME TF family as the query.
    """
    # 1. Process the importances
    importances["Beta"] = importances["Beta"].abs()
    importances = importances[importances["Parameter"].str.contains("-")].copy()
    importances["Gene"] = importances["Parameter"].apply(lambda x: x.split("-")[0])
    
    # Take average over each gene
    importances = (
        importances.groupby("Gene")[["Importance", "Beta"]].mean().reset_index()
    )
    importances["UniProt_AC"] = importances["Gene"].map(gene_to_uniprot_mapping)
    
    # 2. Process Tomtom results for Family-level p-values
    # Drop rows where family is NaN to prevent false matching or errors
    valid_tomtom = tomtom_results.dropna(subset=["Query_Family", "Target_Family"]).copy()
    
    # Filter to only keep rows where the target is in the same family as the query
    tomtom_family_matches = valid_tomtom[
        valid_tomtom["Query_Family"] == valid_tomtom["Target_Family"]
    ]
    
    # Find the lowest p-value for each Query_ID within its family
    tomtom_pvalues = (
        tomtom_family_matches
        .groupby(["Query_ID", "Query_Family"])["p-value"]
        .min()
        .reset_index()
    )
    
    # Handle exact 0 p-values to avoid log(0) warnings
    tomtom_pvalues["p-value"] = tomtom_pvalues["p-value"].replace(0, 1e-300)
    tomtom_pvalues["-log10(p-value)"] = -np.log10(tomtom_pvalues["p-value"])
    
    # 3. Merge the importances with the tomtom family p-values
    merged_df = pd.merge(
        importances,
        tomtom_pvalues,
        left_on="UniProt_AC",
        right_on="Query_ID",
        how="inner",
    )
    
    n_instances = len(merged_df)
    
    # Ensure there's enough data to calculate correlations
    if n_instances < 2:
        return {
            "Pearson": np.nan,
            "Spearman": np.nan,
            "pval_Pearson": np.nan,
            "pval_Spearman": np.nan,
            "n": n_instances
        }
    
    # 4. Calculate Correlations
    pearson_corr, pval_pearson = pearsonr(
        merged_df["Importance"], merged_df["-log10(p-value)"]
    )
    spearman_corr, pval_spearman = spearmanr(
        merged_df["Importance"], merged_df["-log10(p-value)"]
    )
    
    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "pval_Pearson": pval_pearson,
        "pval_Spearman": pval_spearman,
        "n": n_instances
    }

def get_correlation(
    importances: pd.DataFrame,
    tomtom_results: pd.DataFrame,
    gene_to_uniprot_mapping: dict,
):
    #import pdb; pdb.set_trace()
    # Process the importances
    importances["Beta"] = importances["Beta"].abs()
    importances = importances[importances["Parameter"].str.contains("-")]
    importances["Gene"] = importances["Parameter"].apply(lambda x: x.split("-")[0])
    
    # Take average over each gene
    importances = ( # Max improves over mean, which is not too surprising given the anchor strategy.
        importances.groupby("Gene")[["Importance", "Beta"]].max().reset_index() # BETA is not used really!
    )
    importances["UniProt_AC"] = importances["Gene"].map(gene_to_uniprot_mapping)
    
    # Filter the tomtom results to only include the elements where query and target are the same
    tomtom_filtered = tomtom_results[
        tomtom_results["Query_ID"] == tomtom_results["Target_ID"]
    ]
    tomtom_pvalues = tomtom_filtered[["Query_ID", "p-value", "Query_Family"]].copy()
    
    # Handle exact 0 p-values to avoid log(0) warnings
    tomtom_pvalues["p-value"] = tomtom_pvalues["p-value"].replace(0, 1e-300)
    tomtom_pvalues["-log10(p-value)"] = -np.log10(tomtom_pvalues["p-value"])
    
    # Merge the importances with the tomtom p-values on the Query_ID and Feature columns
    merged_df = pd.merge(
        importances,
        tomtom_pvalues,
        left_on="UniProt_AC",
        right_on="Query_ID",
        how="inner",
    )
    
    n_instances = len(merged_df)
    
    pearson_corr, pval_pearson = pearsonr(
        merged_df["Importance"], merged_df["-log10(p-value)"]
    )
    spearman_corr, pval_spearman = spearmanr(
        merged_df["Importance"], merged_df["-log10(p-value)"]
    )
    
    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "pval_Pearson": pval_pearson,
        "pval_Spearman": pval_spearman,
        "n": n_instances
    }

def get_correlation_IC( IC_contents: pd.DataFrame,
                        tomtom_results: pd.DataFrame,
                        gene_to_uniprot_mapping: dict):
    # Process the IC contents (a DataFrame with columns: TF, IC)
    IC_contents["UniProt_AC"] = IC_contents["TF"].map(gene_to_uniprot_mapping)
    IC_contents = IC_contents.dropna(subset=["UniProt_AC"])

    # Filter the tomtom results to only include the elements where query and target are the same
    tomtom_filtered = tomtom_results[
        tomtom_results["Query_ID"] == tomtom_results["Target_ID"]
    ]
    tomtom_pvalues = tomtom_filtered[["Query_ID", "p-value", "Query_Family"]].copy()

    # Handle exact 0 p-values to avoid log(0) warnings
    tomtom_pvalues["p-value"] = tomtom_pvalues["p-value"].replace(0, 1e-300)
    tomtom_pvalues["-log10(p-value)"] = -np.log10(tomtom_pvalues["p-value"])

    # Merge the IC contents with the tomtom p-values on the Query_ID and Feature columns
    merged_df = pd.merge(
        IC_contents,
        tomtom_pvalues,
        left_on="UniProt_AC",
        right_on="Query_ID",
        how="inner",
    )   

    n_instances = len(merged_df)

    pearson_corr, pval_pearson = pearsonr(
        merged_df["IC"], merged_df["-log10(p-value)"]
    )
    spearman_corr, pval_spearman = spearmanr(
        merged_df["IC"], merged_df["-log10(p-value)"]
    )

    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "pval_Pearson": pval_pearson,
        "pval_Spearman": pval_spearman,
        "n": n_instances
    }

def get_correlation_IC_family( IC_contents: pd.DataFrame,
                               tomtom_results: pd.DataFrame,
                               gene_to_uniprot_mapping: dict):
    # Process the IC contents (a DataFrame with columns: TF, IC)
    IC_contents["UniProt_AC"] = IC_contents["TF"].map(gene_to_uniprot_mapping)
    IC_contents = IC_contents.dropna(subset=["UniProt_AC"]) 

    # Filter the tomtom results to only include the elements where query and target are in the same family
    valid_tomtom = tomtom_results.dropna(subset=["Query_Family", "Target_Family"]).copy()
    tomtom_family_matches = valid_tomtom[
        valid_tomtom["Query_Family"] == valid_tomtom["Target_Family"]
    ]   

    # Find the lowest p-value for each Query_ID within its family
    tomtom_pvalues = (
        tomtom_family_matches
        .groupby(["Query_ID", "Query_Family"])["p-value"]
        .min()
        .reset_index()
    )

    # Handle exact 0 p-values to avoid log(0) warnings
    tomtom_pvalues["p-value"] = tomtom_pvalues["p-value"].replace(0, 1e-300)
    tomtom_pvalues["-log10(p-value)"] = -np.log10(tomtom_pvalues["p-value"])

    # Merge the IC contents with the tomtom family p-values
    merged_df = pd.merge(
        IC_contents,
        tomtom_pvalues,
        left_on="UniProt_AC",
        right_on="Query_ID",
        how="inner",
    )

    n_instances = len(merged_df)

    pearson_corr, pval_pearson = pearsonr(
        merged_df["IC"], merged_df["-log10(p-value)"]
    )
    spearman_corr, pval_spearman = spearmanr(
        merged_df["IC"], merged_df["-log10(p-value)"]
    )

    return {
        "Pearson": pearson_corr,
        "Spearman": spearman_corr,
        "pval_Pearson": pval_pearson,
        "pval_Spearman": pval_spearman,
        "n": n_instances
    }

def get_prioritization_correlation_Ath(treatment):
    importances = pd.read_csv(f"results/Ath/{treatment}/parameter_importance.csv")
    tomtom_results = pd.read_csv(f"results/Ath/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/Ath/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_correlation(importances, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_Ath_family(treatment):
    importances = pd.read_csv(f"results/Ath/{treatment}/parameter_importance.csv")
    tomtom_results = pd.read_csv(f"results/Ath/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/Ath/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_family_correlation(importances, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_human(treatment):
    importances = pd.read_csv(f"results/human/{treatment}/parameter_importance.csv")
    tomtom_results = pd.read_csv(f"results/human/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/human/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_correlation(importances, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_human_family(treatment):
    importances = pd.read_csv(f"results/human/{treatment}/parameter_importance.csv")
    tomtom_results = pd.read_csv(f"results/human/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/human/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_family_correlation(importances, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_IC_Ath(treatment):
    IC_contents = pd.read_csv(f"results/Ath/{treatment}/ic_distribution.csv")
    tomtom_results = pd.read_csv(f"results/Ath/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/Ath/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_correlation_IC(IC_contents, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_IC_human(treatment):
    IC_contents = pd.read_csv(f"results/human/{treatment}/ic_distribution.csv")
    tomtom_results = pd.read_csv(f"results/human/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/human/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_correlation_IC(IC_contents, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_IC_Ath_family(treatment):
    IC_contents = pd.read_csv(f"results/Ath/{treatment}/ic_distribution.csv")
    tomtom_results = pd.read_csv(f"results/Ath/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/Ath/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_correlation_IC_family(IC_contents, tomtom_results, gene_to_uniprot_mapping)

def get_prioritization_correlation_IC_human_family(treatment):
    IC_contents = pd.read_csv(f"results/human/{treatment}/ic_distribution.csv")
    tomtom_results = pd.read_csv(f"results/human/{treatment}/tomtom_df.csv")
    mapping = pd.read_csv("data/processed/regulation/human/JASPAR_UNIPROT_mapping.csv")
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    return get_correlation_IC_family(IC_contents, tomtom_results, gene_to_uniprot_mapping)

def get_combined_correlation_data():
    """Fetches correlations for both organisms and returns a unified DataFrame."""
    records = []
    
    ath_treatments = ["PTI_TPM", "leaf_CPTT", "all_hormone_TPM", "PTI_log2FC", "all_hormone_log2FC", "PB_snRNA_hormone_log2FC", "PB_snRNA_hormone_CPTT"]
    for treatment in ath_treatments:
        corr = get_prioritization_correlation_Ath(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Arabidopsis",
            **corr
        })
        
    human_treatments = ["PMBC_CPTT", "TabulaSapiens", "GTX", "TCGA"]
    for treatment in human_treatments:
        corr = get_prioritization_correlation_human(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Human",
            **corr
        })
        
    return pd.DataFrame(records)

def get_combined_correlation_data_IC():
    """Fetches IC correlations for both organisms and returns a unified DataFrame."""
    records = []
    
    ath_treatments = ["PTI_TPM", "leaf_CPTT", "all_hormone_TPM", "PTI_log2FC", "all_hormone_log2FC"]
    for treatment in ath_treatments:
        corr = get_prioritization_correlation_IC_Ath(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Arabidopsis",
            **corr
        })
        
    human_treatments = ["PMBC_CPTT", "TabulaSapiens", "GTX", "TCGA"]
    for treatment in human_treatments:
        corr = get_prioritization_correlation_IC_human(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Human",
            **corr
        })
        
    return pd.DataFrame(records)

def get_combined_correlation_data_IC_family():
    """Fetches IC correlations for both organisms at the family level and returns a unified DataFrame."""
    records = []
    
    ath_treatments = ["PTI_TPM", "leaf_CPTT", "all_hormone_TPM", "PTI_log2FC", "all_hormone_log2FC"]
    for treatment in ath_treatments:
        corr = get_prioritization_correlation_IC_Ath_family(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Arabidopsis",
            **corr
        })
        
    human_treatments = ["PMBC_CPTT", "TabulaSapiens", "GTX", "TCGA"]
    for treatment in human_treatments:
        corr = get_prioritization_correlation_IC_human_family(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Human",
            **corr
        })
        
    return pd.DataFrame(records)

def get_combined_correlation_data_family():
    """Fetches family-level correlations for both organisms and returns a unified DataFrame."""
    records = []
    
    ath_treatments = ["PTI_TPM", "leaf_CPTT", "all_hormone_TPM", "PTI_log2FC", "all_hormone_log2FC"]
    for treatment in ath_treatments:
        corr = get_prioritization_correlation_Ath_family(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Arabidopsis",
            **corr
        })
        
    human_treatments = ["PMBC_CPTT", "TabulaSapiens", "GTX", "TCGA"]
    for treatment in human_treatments:
        corr = get_prioritization_correlation_human_family(treatment)
        records.append({
            "Dataset": treatment,
            "Organism": "Human",
            **corr
        })
        
    return pd.DataFrame(records)

def plot_combined_correlations(
    df: pd.DataFrame, 
    save_path: str = "results/combined_correlations.png"
):
    """Generates a combined bar plot mimicking the requested Seaborn aesthetic."""
    if df.empty:
        print("Dataframe is empty. Skipping plot generation.")
        return

    # Add n to dataset label for y-axis
    df["Dataset_Label"] = df.apply(lambda row: f"{row['Dataset']} (n={row['n']})", axis=1)
    
    # Set up Seaborn style
    sns.set_theme(style="whitegrid", font_scale=1.0)
    
    metric = "Spearman"
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(8, 8),
        sharey=False,
    )

    def get_stars(p):
        if p < 1e-3: return "***"
        elif p < 1e-2: return "**"
        elif p < 5e-2: return "*"
        return "ns"

    # Draw grouped bar plot, coloring by Organism
    sns.barplot(
        data=df,
        y="Dataset_Label",
        x=metric,
        hue="Organism",
        dodge=False, # Keeps bars sequential since each dataset is unique
        palette=["#E8A87C", "#5B9FB5"],
        ax=ax,
        edgecolor="black",
        linewidth=0.8,
        order=df["Dataset_Label"] # Strictly follow order of DataFrame
    )

    ax.set_xlabel(fr"${metric}$", fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_ylabel("")

    # Add significance stars
    pval_col = f"pval_{metric}"
    for i, row in df.iterrows():
        p_val = row[pval_col]
        corr_val = row[metric]
        stars = get_stars(p_val)

        offset = 0.01 if corr_val >= 0 else -0.01
        ha = "left" if corr_val >= 0 else "right"

        ax.text(
            corr_val + offset, 
            i, 
            stars, 
            ha=ha, 
            va="center", 
            fontsize=11, 
            fontweight="bold", 
            color="black"
        )

    # Handle legend nicely
    ax.legend(title="Organism", frameon=True, loc="best", fontsize=11)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved combined correlation plot to: {save_path}")

def print_overall_pooled_correlations():
    """
    Gathers raw data points across all datasets, pools them, 
    and computes the overall Pearson and Spearman correlations.
    """
    print("\n" + "="*60)
    print("OVERALL POOLED CORRELATIONS (ALL DATASETS COMBINED)")
    print("="*60)
    
    ath_treatments = ["PTI_TPM", "leaf_CPTT", "all_hormone_TPM", "PTI_log2FC", "all_hormone_log2FC"]
    human_treatments = ["PMBC_CPTT", "TabulaSapiens", "GTX", "TCGA"]
    datasets = [("Ath", t) for t in ath_treatments] + [("human", t) for t in human_treatments]
    
    pooled_data = {
        "Importance_Exact": [],
        "Importance_Family": [],
        "IC_Exact": [],
        "IC_Family": []
    }
    
    for org, treatment in datasets:
        importances_path = f"results/{org}/{treatment}/parameter_importance.csv"
        ic_path = f"results/{org}/{treatment}/ic_distribution.csv"
        tomtom_path = f"results/{org}/{treatment}/tomtom_df.csv"
        map_path = f"data/processed/regulation/{org}/JASPAR_UNIPROT_mapping.csv"
        
        if not (os.path.exists(importances_path) and os.path.exists(tomtom_path) and os.path.exists(map_path)):
            continue
            
        importances = pd.read_csv(importances_path)
        tomtom_results = pd.read_csv(tomtom_path)
        mapping = pd.read_csv(map_path)
        gene_to_uniprot = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
        
        # --- TOMTOM PROCESSING ---
        # Exact Match
        tt_exact = tomtom_results[tomtom_results["Query_ID"] == tomtom_results["Target_ID"]].copy()
        tt_exact["p-value"] = tt_exact["p-value"].replace(0, 1e-300)
        tt_exact["-log10(p-value)"] = -np.log10(tt_exact["p-value"])
        
        # Family Match
        valid_tt_fam = tomtom_results.dropna(subset=["Query_Family", "Target_Family"]).copy()
        tt_fam_matches = valid_tt_fam[valid_tt_fam["Query_Family"] == valid_tt_fam["Target_Family"]]
        tt_fam = tt_fam_matches.groupby(["Query_ID", "Query_Family"])["p-value"].min().reset_index()
        tt_fam["p-value"] = tt_fam["p-value"].replace(0, 1e-300)
        tt_fam["-log10(p-value)"] = -np.log10(tt_fam["p-value"])
        
        # --- IMPORTANCE PROCESSING ---
        imp_base = importances.copy()
        imp_base["Beta"] = imp_base["Beta"].abs()
        imp_base = imp_base[imp_base["Parameter"].str.contains("-")].copy()
        imp_base["Gene"] = imp_base["Parameter"].apply(lambda x: x.split("-")[0])
        
        # Exact (using max)
        imp_exact = imp_base.groupby("Gene")[["Importance", "Beta"]].max().reset_index()
        imp_exact["UniProt_AC"] = imp_exact["Gene"].map(gene_to_uniprot)
        pooled_data["Importance_Exact"].append(
            pd.merge(imp_exact, tt_exact[["Query_ID", "-log10(p-value)"]], 
                     left_on="UniProt_AC", right_on="Query_ID", how="inner")
        )
        
        # Family (using mean)
        imp_fam = imp_base.groupby("Gene")[["Importance", "Beta"]].mean().reset_index()
        imp_fam["UniProt_AC"] = imp_fam["Gene"].map(gene_to_uniprot)
        pooled_data["Importance_Family"].append(
            pd.merge(imp_fam, tt_fam[["Query_ID", "-log10(p-value)"]], 
                     left_on="UniProt_AC", right_on="Query_ID", how="inner")
        )
        
        # --- IC PROCESSING ---
        if os.path.exists(ic_path):
            ic_df = pd.read_csv(ic_path)
            ic_df["UniProt_AC"] = ic_df["TF"].map(gene_to_uniprot)
            ic_df = ic_df.dropna(subset=["UniProt_AC"])
            
            pooled_data["IC_Exact"].append(
                pd.merge(ic_df, tt_exact[["Query_ID", "-log10(p-value)"]], 
                         left_on="UniProt_AC", right_on="Query_ID", how="inner")
            )
            pooled_data["IC_Family"].append(
                pd.merge(ic_df, tt_fam[["Query_ID", "-log10(p-value)"]], 
                         left_on="UniProt_AC", right_on="Query_ID", how="inner")
            )
            
    # --- CALCULATE & PRINT RESULTS ---
    configs = [
        ("Importance vs Exact Tomtom", "Importance_Exact", "Importance"),
        ("Importance vs Family Tomtom", "Importance_Family", "Importance"),
        ("IC vs Exact Tomtom", "IC_Exact", "IC"),
        ("IC vs Family Tomtom", "IC_Family", "IC")
    ]
    
    for name, key, col in configs:
        if not pooled_data[key]:
            continue
        
        combined = pd.concat(pooled_data[key], ignore_index=True)
        combined = combined.dropna(subset=[col, "-log10(p-value)"])
        
        if len(combined) < 2:
            continue
            
        pearson_corr, pval_pearson = pearsonr(combined[col], combined["-log10(p-value)"])
        spearman_corr, pval_spearman = spearmanr(combined[col], combined["-log10(p-value)"])
        
        print(f"{name} (n = {len(combined)}):")
        print(f"  Pearson:  r = {pearson_corr:7.4f}, p = {pval_pearson:.4e}")
        print(f"  Spearman: r = {spearman_corr:7.4f}, p = {pval_spearman:.4e}\n")
    print("="*60 + "\n")


if __name__ == "__main__":

    # IC
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
    combined_IC_df = get_combined_correlation_data_IC()
    combined_IC_df["Dataset"] = combined_IC_df["Dataset"].map(treatments_pretty_names)
    combined_IC_df["Dataset"] = pd.Categorical(combined_IC_df["Dataset"], categories=treatments_pretty_names.values(), ordered=True)

    plot_combined_correlations(
        df=combined_IC_df, 
        save_path="results/combined_prioritization_correlations_IC.png"
    )
    # IC family
    combined_IC_family_df = get_combined_correlation_data_IC_family()
    plot_combined_correlations(
        df=combined_IC_family_df, 
        save_path="results/combined_prioritization_correlations_IC_family.png"
    )
    # Gather combined data containing n
    combined_df = get_combined_correlation_data()
    combined_df["Dataset"] = combined_df["Dataset"].map(treatments_pretty_names)
    combined_df["Dataset"] = pd.Categorical(combined_df["Dataset"], categories=treatments_pretty_names.values(), ordered=True)
    # Plot combined data using the updated aesthetic
    plot_combined_correlations(
        df=combined_df, 
        save_path="results/combined_prioritization_correlations.png"
    )

    combined_family_df = get_combined_correlation_data_family()
    plot_combined_correlations(
        df=combined_family_df, 
        save_path="results/combined_prioritization_correlations_family.png"
    )
    
    # Print overall combined dataset statistics
    print_overall_pooled_correlations()