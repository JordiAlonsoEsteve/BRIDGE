from __future__ import annotations

from pathlib import Path


def load_promoters(
    promoter_file_path: str | Path,
    exclude_non_ACGT: bool = False,
    return_full_names: bool = False,
) -> dict[str, str] | tuple[dict[str, str], list[str]]:
    """
    Load promoter regions from a FASTA-like file.

    Each promoter entry begins with a header line starting with ``>``,
    followed by one or more lines of nucleotide sequence. The header may contain
    additional metadata separated with ``::``; the gene name is taken as the
    substring before the first ``::``.

    Args:
        promoter_file_path (str | Path):
            Path to the FASTA-like promoter file.
        exclude_non_ACGT (bool, optional):
            If True, skip sequences that contain characters outside ``A, C, G, T``.
            Default is False.
        return_full_names (bool, optional):
            If True, return a tuple ``(promoters, full_names)`` where `full_names`
            contains every header line (minus the ``>``). Default is False.

    Returns:
        dict[str, str]:
            A dictionary mapping gene names to promoter sequences.
        tuple[dict[str, str], list[str]]:
            Returned only if ``return_full_names=True``.
    """
    promoters: dict[str, str] = {}
    full_names: list[str] = []

    non_acgt_count = 0
    valid_bases = {"A", "C", "G", "T"}

    promoter_file_path = Path(promoter_file_path)

    current_gene: str | None = None
    current_seq: list[str] = []

    with promoter_file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line.startswith(">"):
                # Finalize previous sequence if present
                if current_gene is not None:
                    sequence = "".join(current_seq)
                    if not set(sequence).issubset(valid_bases):
                        non_acgt_count += 1
                        if exclude_non_ACGT:
                            # Skip this entry
                            current_seq = []
                            current_gene = line[1:].split("::")[0].strip()
                            full_names.append(line[1:])
                            continue

                    promoters[current_gene] = sequence

                # Start a new record
                current_gene = line[1:].split("::")[0].strip()
                full_names.append(line[1:])
                current_seq = []
            else:
                # Sequence line
                current_seq.append(line)

        # Add final entry after loop
        if current_gene is not None:
            sequence = "".join(current_seq)
            if not set(sequence).issubset(valid_bases):
                non_acgt_count += 1
                if not exclude_non_ACGT:
                    promoters[current_gene] = sequence
            else:
                promoters[current_gene] = sequence

    print("Number of promoters with some non-ACGT letters:", non_acgt_count)
    print("Number of promoters:", len(promoters))

    if return_full_names:
        return promoters, full_names

    return promoters
