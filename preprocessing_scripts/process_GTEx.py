import polars as pl
import polars.selectors as cs

# 1. Use scan_csv instead of read_csv for lazy evaluation (prevents OOM)
x = (
    pl.scan_csv(
        "data/raw/rna_seq_human/GTEx/GTEx.gct",
        separator="\t",
        skip_rows=2,
    )
    # Create the Unique_name column using fast string expressions
    .with_columns(pl.col("Name").str.split(".").list.get(0).alias("Unique_name"))
    # Drop the unneeded columns
    .drop(["Name", "Description"])
    # Group by the new column and sum all numeric columns
    .group_by("Unique_name").sum()
    # Drop any rows containing missing/null values
    .drop_nulls()
    .with_columns(cs.numeric().log1p().cast(pl.Float32))
    # Sort by your gene index column
    .sort("Unique_name")
    # Rename to Gene_ID
    .rename({"Unique_name": "Gene_ID"})
)

# 2. Evaluate/collect the lazy DataFrame into memory
df_final = x.collect()

# 3. SAVE AS PARQUET (As mentioned in your comment)
df_final.write_parquet(
    "data/processed/mRNA/human/GTX_TPM.parquet",
    compression="zstd"  # Highly efficient compression for genomic data
)