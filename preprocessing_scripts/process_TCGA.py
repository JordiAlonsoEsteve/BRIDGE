import pandas as pd

X = pd.read_table(
    "data/raw/rna_seq_human/TCGA/tcga_RSEM_gene_log2_tpm+0.001.csv", index_col=0
)
# remove the stuff after the dot (inlcuding the dot) in the gene names
X.index = X.index.str.split(".").str[0]
# make float32 to save space
X = X.astype("float32")
X.to_csv("data/processed/mRNA/human/TCGA_TPM.csv", float_format=3)
