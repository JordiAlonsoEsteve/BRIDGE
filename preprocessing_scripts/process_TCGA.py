import polars as pl

# 1. Fast, multi-threaded read (assuming tab-separated based on read_table)
df = pl.read_csv(
    "data/raw/rna_seq_human/TCGA/tcga_RSEM_gene_log2_tpm+0.001.csv",
    separator="\t",
)

# 2. Clean the gene names (assuming the first column is the gene index)
gene_col = df.columns[0]
df = df.with_columns(pl.col(gene_col).str.split(".").list.get(0).alias(gene_col))

# 3. Cast numeric columns to float32 to save memory
numeric_cols = [c for c in df.columns if c != gene_col]
df = df.with_columns([pl.col(c).cast(pl.Float32) for c in numeric_cols])

# 4. Write back out super fast
df.write_csv("data/processed/mRNA/human/TCGA_TPM.csv")
print("Successfully processed and saved TCGA matrix!")
