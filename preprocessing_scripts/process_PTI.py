from utils import median_of_ratios_normalization, process_counts_log2FC, to_CPTT, to_TPM
import pandas as pd

def handle_PTI(file_path):
    X = pd.read_csv(file_path, index_col=0)
    datasets = []

    # Handle mutants: No Col_... so merge and add Col
    columns = X.columns
    columns = [
        (
            "Col_" + col.split("_")[0] + "-" + "_".join(col.split("_")[1:])
            if "Col" != col.split("_")[0]
            else col
        )
        for col in columns
    ]
    treatments_plus_mutants = list(set([col.split("_")[1] for col in columns]))
    treatments_plus_mutants.sort()
    X.columns = columns
    for i, s in enumerate(treatments_plus_mutants):
        index = [s == col.split("_")[1] for col in X.columns]
        tmpX = X.loc[:, index]
        # For some reason the number of the replicate is the last one
        # remove the col info
        tmpX.columns = tmpX.columns.str.extract(r"_(.*)")[0]
        # only if the last number after the last _ has 4 digits it is indicating the replicate, otherwise is the time
        time = tmpX.columns.str.extract(rf"{s}_(.*)")[0]
        # remove the last digit if the number has more than 3 digits
        replicate = time.str.extract(r"(\d{4})")[0]
        time = time.str.extract(r"(\d{3})")[0]
        # get the last digit in each number
        replicate = replicate.str[-1]
        replicate = replicate.fillna(
            "0"
        )  # this is because the first replicate does not have a number.
        # Now generate new colnames using time and replicate
        if s == "mock":
            tmpX.columns = "PTI_" + "control" + "_" + time + "_" + replicate
        else:
            tmpX.columns = "PTI_" + s + "_" + time + "_" + replicate
        # print(tmpX)
        datasets.append(tmpX)

    X = pd.concat(datasets, axis=1)
    X.loc[:, X.columns.str.contains("mock")]

    return X


if __name__ == "__main__":
    counts = handle_PTI(
        "data/raw/rna_seq_ath/PTI/PRJEB25079_UncorrectedCounts.csv"
    )
    tpm_counts = to_TPM(
        counts=counts.copy(),
        gene_length=pd.read_table(
            "data/raw/dna_tair10/gene_exon_lengths.txt", index_col=0
        ),
    )
    tpm_counts.to_csv("data/processed/mRNA/ath/PTI_TPM.csv")

    counts = median_of_ratios_normalization(counts)
    counts = process_counts_log2FC(counts, average_treatments=False, clipping=4)
    counts.to_csv("data/processed/mRNA/ath/PTI_log2FC.csv")