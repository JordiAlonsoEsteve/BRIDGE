import pandas as pd


def compute_gene_regions(df: pd.DataFrame, flank: int = 500) -> pd.DataFrame:
    """
    Compute gene-to-gene regions with overlaps and optional flanking.

    Args:
        df (pd.DataFrame): Input BED DataFrame with columns ["Chr", "start", "end", "gene_id", "strand"]
        flank (int): Upstream/downstream flanking (default 500)

    Returns:
        pd.DataFrame: Input DataFrame with changed "start" and "end" columns and added the "."
    """
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    df = df.sort_values(["Chr", "start"]).reset_index(drop=True)

    region_starts = []
    region_ends = []

    chrom_max = df.groupby("Chr")["end"].max().to_dict()

    for idx in range(len(df)):
        chrom = df.loc[idx, "Chr"]
        start_i = df.loc[idx, "start"]
        end_i = df.loc[idx, "end"]

        # Previous gene
        if idx == 0 or df.loc[idx - 1, "Chr"] != chrom:
            prev_end = 0
        else:
            prev_end = df.loc[idx - 1, "end"]

        # Next gene
        if idx == len(df) - 1 or df.loc[idx + 1, "Chr"] != chrom:
            next_start = chrom_max[chrom]
        else:
            next_start = df.loc[idx + 1, "start"]

        # Compute region with flanking
        region_start = max(
            start_i - flank if start_i < prev_end else prev_end, start_i - 3000
        )  # take from the previous but up to 3000 before/after maximum
        region_end = min(
            end_i + flank if end_i > next_start else next_start, end_i + 3000
        )

        region_starts.append(region_start)
        region_ends.append(region_end)

    df["start"] = region_starts
    df["end"] = region_ends
    # Add the score column "."
    df["score"] = "."

    # Reorder columns BED-style
    df = df[["Chr", "start", "end", "gene_id", "score", "strand"]]

    return df
