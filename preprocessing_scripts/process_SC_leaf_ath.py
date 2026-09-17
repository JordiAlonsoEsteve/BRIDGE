import pandas as pd
import numpy as np

file_path = "data/raw/rna_seq_ath/leaf/leaf.expression-matrix-raw.txt"

adata = pd.read_csv(file_path, sep="\t", index_col=0)
# convert to CPTT

# 1. Compute the total counts per cell
total_counts = adata.sum(axis=0)
# 2. Normalize each cell's counts to a target sum (e.g., 10,000)
target_sum = 1e4
normalized_counts = adata.div(total_counts, axis=1) * target_sum
# 3. Log-transform the normalized counts
log_transformed_counts = np.log1p(normalized_counts)
# save
log_transformed_counts.to_csv("data/processed/mRNA/ath//leaf_CPTT.csv")
