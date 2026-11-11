import polars as pl
import polars.selectors as cs

lf = (
    pl.scan_csv(
        "data/raw/rna_seq_human/GTEx/GTEx.gct",
        separator="\t",
        skip_rows=2,
        infer_schema_length=0,          # read everything as strings first...
    )
    .with_columns(
        pl.col("Name").str.split(".").list.get(0).alias("Gene_ID"),
        (cs.all() - cs.by_name("Name", "Description")).cast(pl.Float32),  # ...then cast to Float32 up front
    )
    .drop(["Name", "Description"])
    .group_by("Gene_ID")
    .agg(cs.numeric().sum().log1p())
    .sort("Gene_ID")
)

lf.sink_csv("data/processed/mRNA/human/GTX_TPM.csv", float_precision=3)