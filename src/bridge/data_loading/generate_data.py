from pathlib import Path

import numpy as np
import pandas as pd

from bridge.kmer_processing.get_formated_dna import get_dna_kmer_representation
from BRIDGE.src.bridge.data_loading.load_mrna import load_processed_dataset


def generate_data(
    dataset_name: str,
    DNA_folder: Path,
    DNA_params: dict,
):
    """
    Takes a dataset name, a folder with the genome and annotation, the charactertsitics for
    the construction of the genomic features, a list of TFs and outputs
    X, Y, C.
    """
    TF_list = sorted(set(pd.read_table(f"{DNA_folder}/TF_list.txt")["Gene_ID"]))
    # Get counts
    mrna_df = load_processed_dataset(dataset_name)
    # assert no Nans
    if mrna_df.isna().sum().sum() > 0:
        raise ValueError(
            "Input data contains missing values. Please impute or remove them before running inference."
        )
    # remove genes with that are constant across samples (variance 0)
    mrna_df = mrna_df.loc[mrna_df.var(axis=1) > 0, :]
    # DNA stuff
    if DNA_params["region"] == "TSS&TTS":

        DNA_params["region"] = "TSS"
        count_vectors_tss, kmer_index_tss = get_dna_kmer_representation(
            DNA_folder,
            output_folder="data/processed/dna_tmp",
            region=DNA_params["region"],
            k=DNA_params["k"],
            bp=DNA_params["bp"],
        )
        DNA_params["region"] = "TTS"
        count_vectors_tts, kmer_index_tts = get_dna_kmer_representation(
            DNA_folder,
            output_folder="data/processed/dna_tmp",
            region=DNA_params["region"],
            k=DNA_params["k"],
            bp=DNA_params["bp"],
        )
        assert kmer_index_tss == kmer_index_tts

        # Generate C
        C_tss = pd.DataFrame(count_vectors_tss, index=kmer_index_tss.keys()).T.astype(
            "float64"
        )
        C_tts = pd.DataFrame(count_vectors_tts, index=kmer_index_tss.keys()).T.astype(
            "float64"
        )
        C = C_tss + C_tts

    else:
        count_vectors, kmer_index = get_dna_kmer_representation(
            DNA_folder,
            output_folder="data/processed/dna_tmp",
            region=DNA_params["region"],
            k=DNA_params["k"],
            bp=DNA_params["bp"],
        )

        C = pd.DataFrame(count_vectors, index=kmer_index.keys()).T.astype("float64")

    common_genes = sorted(set(C.index).intersection(mrna_df.index))
    C = C.loc[common_genes]
    mrna_df = mrna_df.loc[common_genes]

    # Split in Y and X
    TF_list = [x for x in TF_list if x in mrna_df.index.to_list()]
    X = mrna_df.loc[TF_list, :]

    # Remove TFs with 0 variance
    X = X.loc[X.var(axis=1) > 0, :]

    Y = mrna_df.loc[~mrna_df.index.isin(TF_list)]
    C = C.loc[~mrna_df.index.isin(TF_list)]

    assert len(set(Y.index) - set(C.index)) == 0
    assert len(set(X.index).intersection(C.index)) == 0
    assert Y.index.equals(C.index), "Y and C indices are not in the same order"

    # row groups (So we are not predicting genes in the same family)
    try:
        gene_families = pd.read_csv(f"{DNA_folder}/gene_families.csv").set_index(
            "gene_id"
        )

    except FileNotFoundError:
        print(
            f"gene_families.csv not found in {DNA_folder}. Generating a dummy file where each gene is in its own family."
        )
        # generate a dummy file where each gene is in its own family
        gene_families = pd.DataFrame(
            {"family_id": range(len(Y.index)), "gene_id": Y.index}
        ).set_index("gene_id")

    try:
        row_groups = gene_families.loc[Y.index]["family_id"].to_list()
    except KeyError:
        print(
            f"Some genes in {dataset_name} are not in the gene_families.csv file. Extending the file with unique family IDs for those genes."
        )
        # for the genes that are not in the gene families, just give them a unique family ID that
        # is not in the gene families file
        missing_genes = set(Y.index) - set(gene_families.index)
        missing_families = range(
            len(gene_families), len(gene_families) + len(missing_genes)
        )
        missing_gene_families = pd.DataFrame(
            {"family_id": missing_families, "gene_id": list(missing_genes)}
        ).set_index("gene_id")
        gene_families = pd.concat([gene_families, missing_gene_families])
        row_groups = gene_families.loc[Y.index]["family_id"].to_list()

    print(f"X shape: {X.shape}, Y shape: {Y.shape}, C shape: {C.shape}")

    X = X.sort_index(axis=0)
    Y = Y.sort_index(axis=0)
    C = C.sort_index(axis=0)
    assert Y.index.equals(C.index), "Y and C indices are not in the same order"
    assert Y.columns.equals(X.columns), "X and Y columns are not in the same order"

    return X, Y, C, np.array(row_groups)
