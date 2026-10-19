import pandas as pd
from bridge.kmer_processing.get_formated_dna import get_dna_kmer_representation

TF_list = sorted(set(pd.read_table("data/raw/dna_tair10/TF_list.txt")["Gene_ID"]))

all_counts = pd.read_csv("data/processed/mRNA/ath/PTI_log2FC.csv", index_col=0)

cunt_vectors, kmer_index = get_dna_kmer_representation(
    "data/raw/dna_tair10",
    output_folder="data/processed/dna_tmp",
    region="TSS",
    k=6,
    bp=(2000, 500),
)
C = pd.DataFrame(cunt_vectors, index=kmer_index.keys()).T.astype("float64")

common_genes = sorted(set(C.index).intersection(all_counts.index))
C = C.loc[common_genes]
all_counts = all_counts.loc[common_genes]

# Split in Y and X
TF_list = [x for x in TF_list if x in all_counts.index.to_list()]
X = all_counts.loc[TF_list, :]

# Remove TFs with 0 variance
X = X.loc[X.var(axis=1) > 0, :]

Y = all_counts.loc[~all_counts.index.isin(TF_list)]
C = C.loc[~all_counts.index.isin(TF_list)]

# save X, Y and C in demo folder
Y.to_csv("demo/Y.csv")
X.to_csv("demo/X.csv")
C.to_csv("demo/C.csv")
