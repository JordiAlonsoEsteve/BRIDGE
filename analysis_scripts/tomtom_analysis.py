import os
import subprocess
from multiprocessing import Pool, cpu_count

import pandas as pd
import requests
from diskcache import Cache
from tqdm import tqdm

from bridge.motif.motif_construction import (
    generate_meme_file,
    generate_PWM_from_theta,
)


cache = Cache(".uniprot_cache")


@cache.memoize()
def uniprot_to_gene_id(uniprot_id: str) -> str:
    """Queries UniProt REST API to convert any accession (Swiss-Prot or TrEMBL)

    into a canonical Gene/Locus ID (e.g., TAIR 'AT1G01060' or HGNC Symbol).
    """
    if pd.isna(uniprot_id) or not isinstance(uniprot_id, str):
        return str(uniprot_id)

    clean_id = uniprot_id.split("_")[0].strip()

    url = f"https://rest.uniprot.org/uniprotkb/{clean_id}.json"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return clean_id

        data = response.json()

        # 1. Try to extract ordered locus name (e.g., AT1G01060 for plants)
        if "genes" in data:
            for gene in data["genes"]:
                if gene.get("orderedLocusNames"):
                    return gene["orderedLocusNames"][0]["value"]
                if "geneName" in gene:
                    return gene["geneName"]["value"]

        # 2. Fall back to primary canonical Swiss-Prot accession if locus unavailable
        return data.get("primaryAccession", clean_id)

    except (requests.RequestException, ValueError):
        return clean_id


# ==========================================================
# TOMTOM SUBPROCESS EXECUTION
# ==========================================================


def run_tomtom(query_meme, target_meme, output_dir) -> pd.DataFrame:
    """Executes a TOMTOM run via subprocess and returns the path to the results.

    Parameters:
    - query_meme: Path to your generated inferred motifs.
    - target_meme: Path to the JASPAR/database MEME file.
    - output_dir: Directory to store results.
    """
    tsv_path = os.path.join(output_dir, "tomtom.tsv")
    if os.path.exists(tsv_path):
        print(
            f"Results found in {output_dir}. Removing old results and re-running TOMTOM..."
        )
        os.remove(tsv_path)

    cmd = [
        "tomtom",
        "-oc",
        output_dir,
        "-thresh",
        "1",
        query_meme,
        target_meme,
        "-verbosity",
        "2",
    ]

    print(f"Executing: {' '.join(cmd)}")

    try:
        _ = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("TOMTOM run completed successfully.")

        if os.path.exists(tsv_path):
            df = pd.read_csv(tsv_path, sep="\t", comment="#")

            # Cleanup extra output files
            for ext in ["tomtom.html", "tomtom.xml"]:
                file_to_remove = os.path.join(output_dir, ext)
                if os.path.exists(file_to_remove):
                    os.remove(file_to_remove)

            return df
        else:
            print("Warning: TOMTOM finished but tomtom.tsv was not found.")
            return None

    except subprocess.CalledProcessError as e:
        print(f"Error during TOMTOM execution:\n{e.stderr}")
        return None
    except FileNotFoundError:
        print(
            "Error: 'tomtom' command not found. Is MEME Suite installed and in your PATH?"
        )
        return None


# ==========================================================
# WORKER PROCESS FOR INDIVIDUAL TREATMENTS
# ==========================================================


def _process_treatment(
    treatment,
    base_result_dir,
    gene_to_uniprot,
    gene_to_family_mapping,
    gene_to_name_mapping,
    uniprot_to_family_mapping,
    uniprot_to_name_mapping,
    meme_JASPAR,
    strict_mapping,
):
    """Worker function to process a single treatment for a given organism."""
    input_dir = f"{base_result_dir}/{treatment}"

    # Load importance matrix and generate PWM
    theta = pd.read_csv(f"{input_dir}/importance_matrix.csv", index_col=0)
    PWM_dict, ic_distribution = generate_PWM_from_theta(theta)

    # Transform ic distribution to a DataFrame and save
    ic_df = pd.DataFrame(list(ic_distribution.items()), columns=["TF", "IC"])
    ic_df.to_csv(f"{input_dir}/ic_distribution.csv", index=False)

    print(f"[{base_result_dir}] Generated PWM for {treatment}, TFs: {len(PWM_dict)}")

    # Map PWMs to UniProt
    PWM_dict_mapped = {}
    for tf_name, pwm in PWM_dict.items():
        if tf_name in gene_to_uniprot:
            uniprot_ac = gene_to_uniprot[tf_name]
            PWM_dict_mapped[uniprot_ac] = pwm
        else:
            msg = (
                f"TF name {tf_name} not found in mapping. Cannot convert to UniProt AC."
            )
            if strict_mapping:
                raise ValueError(msg)
            else:
                print(msg)

    # Generate MEME file
    meme_path = f"{input_dir}/inferred_motifs.meme"
    generate_meme_file(PWM_dict_mapped, meme_path)

    # Run TOMTOM
    output_dir = f"{input_dir}/tomtom_results"
    tomtom_df = run_tomtom(meme_path, meme_JASPAR, output_dir)

    if tomtom_df is not None:
        # 1. Map Query_ID and Target_ID to canonical Gene IDs (resolving Swiss-Prot / TrEMBL IDs)
        unique_ids = set(tomtom_df["Query_ID"].dropna()).union(
            set(tomtom_df["Target_ID"].dropna())
        )
        print(
            f"[{treatment}] Resolving {len(unique_ids)} unique UniProt/Gene IDs via API..."
        )
        id_to_gene = {}
        for uid in tqdm(unique_ids, desc=f"[{treatment}] Resolving IDs", unit="ID"):
            id_to_gene[uid] = uniprot_to_gene_id(uid)

        tomtom_df["Query_Gene_ID"] = (
            tomtom_df["Query_ID"].map(id_to_gene).fillna(tomtom_df["Query_ID"])
        )
        tomtom_df["Target_Gene_ID"] = (
            tomtom_df["Target_ID"].map(id_to_gene).fillna(tomtom_df["Target_ID"])
        )

        # 2. Map Names and Families (primary lookup via Gene_ID, fallback to UniProt_AC)
        tomtom_df["Query_Name"] = (
            tomtom_df["Query_Gene_ID"]
            .map(gene_to_name_mapping)
            .fillna(tomtom_df["Query_ID"].map(uniprot_to_name_mapping))
        )
        tomtom_df["Target_Name"] = (
            tomtom_df["Target_Gene_ID"]
            .map(gene_to_name_mapping)
            .fillna(tomtom_df["Target_ID"].map(uniprot_to_name_mapping))
        )

        tomtom_df["Query_Family"] = (
            tomtom_df["Query_Gene_ID"]
            .map(gene_to_family_mapping)
            .fillna(tomtom_df["Query_ID"].map(uniprot_to_family_mapping))
        )
        tomtom_df["Target_Family"] = (
            tomtom_df["Target_Gene_ID"]
            .map(gene_to_family_mapping)
            .fillna(tomtom_df["Target_ID"].map(uniprot_to_family_mapping))
        )

        tomtom_df.to_csv(f"{input_dir}/tomtom_df.csv", index=False)

    return f"{base_result_dir}/{treatment} successful"


# ==========================================================
# ORGANISM PIPELINE RUNNER
# ==========================================================


def run_organism_pipeline(
    organism_name,
    base_result_dir,
    mapping_path,
    jaspar_path,
    tf_list_path,
    treatments,
    strict_mapping,
):
    """Sets up data mappings and multiprocesses treatments for a single organism."""
    print(f"\n--- Starting processing for {organism_name} ---")

    # Load mappings
    mapping = pd.read_csv(mapping_path)
    family_mapping = pd.read_table(tf_list_path, sep="\t")

    # Create dictionary mappings (Gene_ID based and UniProt_AC based)
    gene_to_uniprot_mapping = dict(zip(mapping["Gene_id"], mapping["UniProt_AC"]))
    gene_to_name_mapping = dict(zip(mapping["Gene_id"], mapping["Motif_Name"]))
    uniprot_to_name_mapping = dict(zip(mapping["UniProt_AC"], mapping["Motif_Name"]))

    gene_to_family_mapping = dict(
        zip(family_mapping["Gene_ID"], family_mapping["Family"])
    )

    # Add UniProt_AC to the family mapping for fallbacks
    family_mapping["UniProt_AC"] = family_mapping["Gene_ID"].map(
        gene_to_uniprot_mapping
    )
    family_mapping_valid = family_mapping.dropna(subset=["UniProt_AC"])
    uniprot_to_family_mapping = dict(
        zip(family_mapping_valid["UniProt_AC"], family_mapping_valid["Family"])
    )

    # Bundle arguments for starmap
    task_args = [
        (
            treatment,
            base_result_dir,
            gene_to_uniprot_mapping,
            gene_to_family_mapping,
            gene_to_name_mapping,
            uniprot_to_family_mapping,
            uniprot_to_name_mapping,
            jaspar_path,
            strict_mapping,
        )
        for treatment in treatments
    ]

    num_cores = min(len(treatments), cpu_count())
    print(f"Spawning {num_cores} parallel worker processes for {organism_name}...")

    with Pool(processes=num_cores) as pool:
        results = pool.starmap(_process_treatment, task_args)

    print(f"All treatments processed for {organism_name}:")
    for res in results:
        print("  -", res)


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    workflows = [
        {
            "organism_name": "Arabidopsis",
            "base_result_dir": "results/ath",
            "mapping_path": "data/processed/regulation/ath/JASPAR_UNIPROT_mapping.csv",
            "jaspar_path": "data/processed/regulation/ath/JASPAR2026_CORE_plants_non-redundant_pfms_meme_uniprot.txt",
            "tf_list_path": "data/raw/dna_tair10/TF_list.txt",
            "treatments": [
                "PTI_TPM",
                "all_hormone_TPM",
                "leaf_CPTT",
                "PTI_log2FC",
                "all_hormone_log2FC",
                "PB_snRNA_hormone_log2FC",
                "PB_snRNA_hormone_CPTT",
            ],
            "strict_mapping": False,
        },
        {
            "organism_name": "Human",
            "base_result_dir": "results/human",
            "mapping_path": "data/processed/regulation/human/JASPAR_UNIPROT_mapping.csv",
            "jaspar_path": "data/processed/regulation/human/JASPAR2026_CORE_vertebrates_non-redundant_pfms_meme_uniprot.txt",
            "tf_list_path": "data/raw/dna_human38/TF_list.txt",
            "treatments": ["TabulaSapiens", "GTX", "PMBC_CPTT", "TCGA"],
            "strict_mapping": False,
        },
    ]

    for config in workflows:
        run_organism_pipeline(**config)
