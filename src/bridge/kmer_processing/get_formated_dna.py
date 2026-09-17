import subprocess
from pathlib import Path
from typing import Literal

from bridge.dna_parsing.load_promoters import load_promoters
from bridge.kmer_processing.k_mer_utils import generate_kmer_count_vector


def get_dna_kmer_representation(
    dna_folder: str | Path,
    output_folder: str | Path,
    region: Literal["TSS", "TTS", "GtG"],
    k: int,
    bp: tuple[int, int],
    rc=True,
    bed_rc=False,
):
    """
    Generate k-mer representations for DNA sequences in a genome folder.

    This function wraps a shell script (`get_dna_segment.sh`) that extracts
    DNA segments based on genomic regions (e.g., TSS, TTS, gene-to-gene)
    and computes k-mer representations.

    Parameters
    ----------
    dna_folder : str or Path
        Path to the folder containing genome files (.fasta and .gff/.gff3).
    output_folder : str or Path
        Path to the folder where output files (k-mer representations and coordinates) will be saved.
    region : {'TSS', 'TTS', 'TSS_TTS', 'GtG'}
        Genomic region to extract:
        - 'TSS': transcription start site,
        - 'TTS': transcription termination site,
        - 'GtG': gene-to-gene regions.
    k : int
        Length of k-mers to compute from the DNA sequences.
    bp : tuple of int, optional
        Upstream and downstream base-pairs to include relative to TSS/TTS.
        Required if `region` is 'TSS' or 'TTS'.
        Format: `(upstream_bp, downstream_bp)`.
    rc : bool, default True
        If True, computes reverse complements of sequences for k-mer representation.
    bed_rc : bool, default False
        If True, uses strand information when generating BED files for coordinates.
    """

    # compute k-mer representation using k
    if region in ["TSS", "TTS"]:
        cmd = [
            "src/dna_parsing/get_dna_segment.sh",
            "-u",
            str(bp[0]),
            "-d",
            str(bp[1]),
            "-r",
            str(region),
            "-i",
            str(dna_folder),
            "-o",
            str(output_folder),
            "-s",
            "True" if bed_rc else "False",
        ]
    else:
        cmd = [
            "src/dna_parsing/get_dna_segment.sh",
            "-g",
            "True",
            "-i",
            str(dna_folder),
            "-o",
            str(output_folder),
            "-s",
            "True" if bed_rc else "False",
        ]

    try:
        # capture_output=True lets us see WHY it failed if it does
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Bash Script Error Log:\n{e.stderr}")
        raise

    return generate_kmer_count_vector(
        load_promoters(f"{output_folder}/gene_sequences.fa"), k, reverse_complement=rc
    )
