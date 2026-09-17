from pathlib import Path

import pandas as pd

# Base directory for CSV files
BASE_DIR = Path("data/processed/mRNA")

# Mapping from logical names to the CSV stem names or folder names
CSV_MAP = {
    "TabulaSapiens": "sc_human/TabulaSapiens_full_CPTT",
    "GTX": "bulk_human/GTX_TPM",
    "all_hormone_TPM": "bulk_ath/all_hormone_TPM",
    "TCGA": "bulk_human/TCGA_TPM",
    "PTI_TPM": "bulk_ath/PTI_TPM",
    "PMBC_CPTT": "sc_human/PBMC_CPTT",
    "leaf_CPTT": "sc_ath/leaf_CPTT",
    "PTI_log2FC": "bulk_ath/PTI_log2FC",
    "all_hormone_log2FC": "bulk_ath/all_hormone_log2FC",
    "PB_snRNA_hormone_log2FC": "bulk_ath/PB_snRNA_hormone_log2FC",
    "PB_snRNA_hormone_CPTT": "bulk_ath/PB_snRNA_hormone_CPTT",
}


def load_csv_by_name(base_dir: str, name: str) -> pd.DataFrame:
    """
    Search recursively under `base_dir` for a CSV file whose stem matches `name`
    and load it as a DataFrame.

    Parameters
    ----------
    base_dir : str or Path
        Root directory to search in (e.g., "processed/mRNA").
    name : str
        Name of the file to load (without '.csv'), e.g., "ABA"

    Returns
    -------
    pd.DataFrame
        Loaded CSV.

    Raises
    ------
    FileNotFoundError
        If no matching CSV file is found.
    """
    base_path = Path(base_dir)
    matches = list(base_path.rglob(f"{name}.csv"))

    if not matches:
        raise FileNotFoundError(
            f"No CSV file found for name '{name}' under '{base_dir}'"
        )

    # Take the first match if multiple
    file_path = matches[0]
    print(f"Loading CSV: {file_path}")

    return pd.read_csv(file_path, index_col=0)


def load_processed_dataset(name: str) -> pd.DataFrame:
    """
    Load a CSV by a logical name using the CSV_MAP.

    Parameters
    ----------
    name : str
        Logical name (key in CSV_MAP)

    Returns
    -------
    pd.DataFrame
        Loaded CSV
    """
    if name not in CSV_MAP:
        raise ValueError(
            f"Name '{name}' not in CSV_MAP. Available keys: {list(CSV_MAP.keys())}"
        )

    # Extract the path stem from the mapping
    path_stem = CSV_MAP[name]

    # Split into folder and file
    folder, file_stem = Path(path_stem).parent, Path(path_stem).name

    # Construct the full directory path
    dir_path = BASE_DIR / folder

    # Use the previously defined function
    return load_csv_by_name(dir_path, file_stem)
