import pandas as pd
import numpy as np
import warnings


def median_of_ratios_normalization(counts: pd.DataFrame) -> pd.DataFrame:

    # Step 1: Compute geometric mean for each gene
    def geo_mean_nonzero(x):
        x = x[x > 0]
        if len(x) == 0:
            return np.nan
        return np.exp(np.mean(np.log(x)))

    geometric_means = counts.apply(geo_mean_nonzero, axis=1)

    if geometric_means.isna().any():
        n_drop = geometric_means.isna().sum()
        warnings.warn(f"Dropping {n_drop} genes with zero counts in all samples")
        keep = geometric_means.notna()
        counts = counts.loc[keep]
        geometric_means = geometric_means.loc[keep]

    # Step 2: Compute ratios of counts to geometric means
    # ratios = counts.div(geometric_means, axis=0)
    ratios = counts.div(geometric_means.replace(0, np.nan), axis=0)

    # Step 3: Compute median ratio per sample: size factors
    size_factors = ratios.median(axis=0, skipna=True)
    zero_sf = size_factors[size_factors == 0]
    if not zero_sf.empty:
        warnings.warn(
            f"Size factor is zero for samples: {list(zero_sf.index)}, dropping"
        )
        counts = counts.drop(columns=zero_sf.index)
        ratios = ratios.drop(columns=zero_sf.index)
        size_factors = size_factors.drop(index=zero_sf.index)

    # Step 4: Normalize counts
    normalized_counts = counts.div(size_factors, axis=1)

    return normalized_counts


def process_counts_log2FC(counts: pd.DataFrame, average_treatments=False, clipping=4):
    # Uses the control per sample to normalize the data
    counts = counts.dropna(axis=0, how="any")  # axis=0 means rows
    # Extract column groups by the first two parts separated by '_'
    new_cols = counts.columns.to_series().apply(lambda x: "_".join(x.split("_")[:3]))
    # new_cols = pd.Series([counts.columns[0]] + new_cols.tolist(), index=counts.columns)
    # Identify control and treatment groups
    control_cols = [col for col in counts.columns if "control" in col]
    treatment_cols = [col for col in counts.columns if "control" not in col]
    # Average controls
    control_means = counts[control_cols].T.groupby(new_cols[control_cols]).mean().T
    if average_treatments:
        treatments = counts[treatment_cols].T.groupby(new_cols[treatment_cols]).mean().T
    # Combine averaged controls + original treatments
    else:
        treatments = counts[treatment_cols]

    counts = pd.concat([control_means, treatments], axis=1)

    # Normalize by controls
    control_map = {
        t: t.split("_")[0] + "_control_" + t.split("_")[2] for t in treatments.columns
    }

    counts_norm_Y = counts.copy()
    for t_col in treatments.columns:
        c_col = control_map[t_col]
        if c_col in counts.columns:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")  # catch all warnings
                # Convert columns to numeric, coerce non-numeric to NaN
                t_numeric = pd.to_numeric(counts[t_col], errors="coerce")
                c_numeric = pd.to_numeric(counts[c_col], errors="coerce")

                counts_norm_Y[t_col] = np.clip(
                    np.log2((t_numeric + 1) / (c_numeric + 1)), -clipping, clipping
                )  # CLIPPING!

                # Check if any warning occurred
                for warn in w:
                    print(
                        f"Warning during log2 calculation for {t_col}: {warn.message}"
                    )

        else:
            print(f"Warning: control column {c_col} not found for {t_col}")

    counts_norm_Y = counts_norm_Y[treatments.columns]

    counts_norm_Y = counts_norm_Y.sort_index()

    counts_norm_Y = counts_norm_Y.loc[:, counts_norm_Y.std() != 0]

    counts = counts_norm_Y.dropna()

    print(f"Final shape, {counts.shape}")
    return counts


def merge_data(conts_list: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge a list of DataFrames on their index, reporting number of rows lost.

    Parameters:
        conts_list (list[pd.DataFrame]): List of DataFrames to merge.

    Returns:
        pd.DataFrame: Merged DataFrame with common indices.
    """
    if not conts_list:
        raise ValueError("The input list is empty")

    # Start with the first DataFrame
    merged = conts_list[0].copy()
    initial_rows = merged.shape[0]

    for i, df in enumerate(conts_list[1:], start=2):
        merged = merged.join(df, how="inner", rsuffix=f"_df{i}")
        rows_after = merged.shape[0]
        # Actually, compute lost rows per join
        lost_rows = initial_rows - rows_after
        print(f"After merging DataFrame {i}: {lost_rows} rows lost")
        initial_rows = rows_after
    merged = merged.sort_index()
    print(f"Final merged DataFrame shape: {merged.shape}")
    return merged


def to_TPM(counts: pd.DataFrame, gene_length: pd.DataFrame) -> pd.DataFrame:
    # Divide counts by gene length in kilobases
    gene_length_kb = gene_length / 1000
    rpk = counts.div(gene_length_kb["Union_Exon_Length_bp"], axis=0)
    # Sum of RPK values for each sample
    rpk_sum = rpk.sum(axis=0)
    # Calculate TPM
    tpm_counts = rpk.div(rpk_sum, axis=1) * 1e6
    # Drop missing values if any
    tpm_counts = tpm_counts.dropna()
    # Drop genes with zero TPM across all samples
    tpm_counts = tpm_counts.loc[~(tpm_counts == 0).all(axis=1)]
    # Drop NaN values if any
    tpm_counts = tpm_counts.dropna()
    # Apply log1p transformation
    tpm_counts = np.log1p(tpm_counts)
    return tpm_counts


def to_CPTT(counts: pd.DataFrame):
    # Step 1: Compute the total counts per cell
    total_counts = counts.sum(axis=0)
    # Step 2: Normalize each cell's counts to a target sum (e.g., 10,000)
    target_sum = 1e4
    normalized_counts = counts.div(total_counts, axis=1) * target_sum
    # Step 3: Log-transform the normalized counts
    log_transformed_counts = np.log1p(normalized_counts)
    return log_transformed_counts
