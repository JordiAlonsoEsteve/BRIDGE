import pandas as pd
import numpy as np
from scipy.sparse import issparse
import scanpy as sc
from scipy.sparse import csr_matrix
from tqdm import tqdm
import sys
from utils import median_of_ratios_normalization, process_counts_log2FC, to_CPTT

def handle_PB_snRNA_hormone(file_path):
    # Read the pseudobulk counts CSV (or use DataFrame directly)
    if isinstance(file_path, pd.DataFrame):
        X = file_path.copy()
    else:
        X = pd.read_csv(file_path, index_col=0)

    def convert_sample_name(col_name):
        # Expects original format: "Hormone_Time_Celltype_Replicate" 
        # e.g., "Auxin_0.5h_Epidermis_rep1" or "mock_3h_Mesophyll_1"
        parts = col_name.split("_")
        
        hormone = parts[0]
        time_raw = parts[1]
        celltype = parts[2]
        rep_raw = parts[3]

        # 1. Format Treatment (mock -> control)
        treatment = "control" if hormone.lower() in ["mock", "control"] else hormone

        # 2. Convert Time to minutes (0.5h -> 30, 3h -> 180)
        time_val = float(time_raw.lower().replace("h", "").replace("time", ""))
        time_min = int(time_val * 60)

        # 3. Clean Replicate number (rep1 -> 1)
        replicate = rep_raw.lower().replace("rep", "")

        return f"PBsnRNA_{treatment}_{time_min}_{celltype}_{replicate}"

    X.columns = [convert_sample_name(col) for col in X.columns]

    cell_types_not_in_mock = set(col.split("_")[3] for col in X.columns if "control" not in col)
    cell_types_in_mock = set(col.split("_")[3] for col in X.columns if "control" in col)
    # assert those two sets are equal, otherwise raise a warning
    assert cell_types_not_in_mock == cell_types_in_mock, (
        f"Cell types in mock and treatment do not match: "
        f"mock: {cell_types_in_mock}, treatment: {cell_types_not_in_mock}"
    )
    return X



if __name__ == "__main__":
    # Load the .h5ad file into an AnnData object
    adata = sc.read_h5ad("data/raw/rna_seq_ath/SN_hormone/hormone_ara.h5ad")

    # Inspect the loaded object
    print(adata)


    # 1. Create a unique ID for each pseudobulk sample
    adata.obs['pb_sample'] = (
        adata.obs['Hormone'].astype(str) + "_" +
        adata.obs['Time'].astype(str) + "_" +
        adata.obs['celltype'].astype(str) + "_rep" +
        adata.obs['Replicates'].astype(str)
    )

    # 2. Extract the raw counts matrix (ensure it's sparse for efficient math)
    raw_matrix = adata.layers['raw']
    if not issparse(raw_matrix):
        raw_matrix = csr_matrix(raw_matrix)

    # 3. Group cells by the unique sample ID
    groups = adata.obs.groupby('pb_sample')

    # 4. Sum counts for each group
    pb_counts = {}
    for sample_id, indices in tqdm(groups.indices.items()):
        # Sum the rows (cells) for this specific group
        # .A1 flattens the 1D matrix into a standard numpy array
        pb_counts[sample_id] = raw_matrix[indices, :].sum(axis=0).A1

    # 5. Convert the dictionary into a Pandas DataFrame (Genes x Samples)
    # We transpose (.T) so genes are rows and samples are columns
    df_counts = pd.DataFrame(pb_counts, index=adata.var_names)

    # 6. Create the Metadata DataFrame (Samples x Variables)
    # Drop duplicate cell-level metadata to get sample-level metadata
    df_metadata = adata.obs[['pb_sample', 'Hormone', 'Time', 'celltype', 'Replicates']].drop_duplicates()
    df_metadata.set_index('pb_sample', inplace=True)

    # save
    df_counts.to_csv("data/raw/rna_seq_ath/SN_hormone/pseudobulk_hormone_counts.csv")
    df_metadata.to_csv("data/raw/rna_seq_ath/SN_hormone/pseudobulk_hormone_metadata.csv")

    PB_counts = handle_PB_snRNA_hormone(
        "data/raw/rna_seq_ath/SN_hormone/pseudobulk_hormone_counts.csv"
    )
    PB_counts = median_of_ratios_normalization(PB_counts)
    PB_counts = process_counts_log2FC(PB_counts, average_treatments=False, clipping=4)
    PB_counts.to_csv("data/processed/mRNA/ath/PB_snRNA_hormone_log2FC.csv")

    # Also to CPTT
    PB_counts_CPTT = to_CPTT(PB_counts)
    PB_counts_CPTT.to_csv("data/processed/mRNA/ath/PB_snRNA_hormone_CPTT.csv")
