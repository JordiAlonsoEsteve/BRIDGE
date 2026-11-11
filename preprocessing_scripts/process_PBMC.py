import scanpy as sc
import numpy as np
import pandas as pd

adata = sc.read_10x_mtx(
    "data/raw/rna_seq_human/PMBC/filtered_feature_bc_matrix/", var_names="gene_symbols"
)
adata.var_names_make_unique()
# compute the number of genes per cell (computes ‘n_genes' column)
sc.pp.filter_cells(adata, min_genes=0)
# mito and genes/counts cuts
mito_genes = adata.var_names.str.startswith("MT-")
# for each cell compute fraction of counts in mito genes vs. all genes
adata.obs["percent_mito"] = np.ravel(np.sum(adata[:, mito_genes].X, axis=1)) / np.ravel(
    np.sum(adata.X, axis=1)
)
# add the total counts per cell as observations-annotation to adata
adata.obs["n_counts"] = np.ravel(adata.X.sum(axis=1))
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs["n_genes"] < 4000, :]
adata = adata[adata.obs["percent_mito"] < 0.15, :]

# compute CPTT
adata.raw = adata


# 2. Compute CPTT by setting target_sum to 10,000
sc.pp.normalize_total(adata, target_sum=1e4)

# 3. Log-transform the data (almost always done immediately after CPTT/CPM normalization)
sc.pp.log1p(adata)

df_cells_x_genes = pd.DataFrame(
    adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
    index=adata.obs_names,
    columns=adata.var["gene_ids"],  # This switches symbols to Ensembl IDs
)

# 3. Transpose to get Genes x Cells
df_genes_x_cells = df_cells_x_genes.T

# 4. Save to a CSV file
df_genes_x_cells.to_csv("data/processed/mRNA/human/PBMC_CPTT.csv")
print("Successfully saved genes x cells matrix to CSV.")
