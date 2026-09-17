import polars as pl
import polars.selectors as cs

# 1. Load and process the massive GTEx file
x = (
    pl.read_csv(
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
)
x = x.with_columns(cs.numeric().log1p().cast(pl.Float32))

# 3. Sort by your gene index column
x = x.sort("Unique_name")
# change that to Gene_ID
x = x.rename({"Unique_name": "Gene_ID"})

# 4. SAVE AS PARQUET (The final piece)
x.write_csv(
    "data/processed/mRNA/human/GTX_TPM.csv",
    float_precision=3,  # <-- Only writes 3 decimal places (e.g., 0.693) to the text file
)
