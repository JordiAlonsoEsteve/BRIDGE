from itertools import combinations, product

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def make_reverse_complement_gapped(kmer: str) -> str:
    """
    Reverse complement for gapped k-mers, preserving gaps ("_").
    """
    complement = {"A": "T", "T": "A", "C": "G", "G": "C", "_": "_"}
    return "".join(complement[b] for b in reversed(kmer))


def canonical_kmer(kmer: str) -> str:
    """
    Returns the lexicographically smaller between a k-mer and its reverse complement.
    """
    rc = make_reverse_complement_gapped(kmer)
    return min(kmer, rc)


def generate_masks_with_consecutive_bases(k, span):
    """
    Generate masks for gapped k-mers where at least `min_consecutive` bases are consecutive.

    Parameters:
    k (int): Number of bases to keep
    span (int): Total window length (including gaps)

    Returns:
    list of tuples: Each tuple represents positions to keep
    """
    min_consecutive = 1
    all_masks = combinations(range(span), k)
    filtered_masks = []

    for mask in all_masks:
        if mask[0] != 0 or mask[-1] != span - 1:
            continue
        # Count max consecutive positions
        max_consec = 1
        current_consec = 1
        for i in range(1, len(mask)):
            if mask[i] - mask[i - 1] == 1:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 1
        if max_consec >= min_consecutive:
            filtered_masks.append(mask)

    return filtered_masks


def unique_kmers_gapped(k, span=None, use_rev_comp=True):
    """
    Generate all possible gapped k-mers over A,C,G,T with optional RC equivariance.
    """
    if span is None:
        span = k
    masks = generate_masks_with_consecutive_bases(k, span)
    kmers = set()

    for mask in masks:
        for letters in product("ACGT", repeat=k):
            kmer_chars = ["_" for _ in range(span)]
            for idx, pos in enumerate(mask):
                kmer_chars[pos] = letters[idx]
            kmer = "".join(kmer_chars)
            if use_rev_comp:
                kmer = canonical_kmer(kmer)
            kmers.add(kmer)

    return sorted(kmers)


def make_kmer(promoters: dict, k: int, rc_equivariant: bool = False, n_jobs=-1) -> dict:
    """
    Generate contiguous or gapped k-mers for a set of promoters (parallelized over genes).

    Parameters
    ----------
    promoters : dict
        gene -> promoter sequence
    k : int
        number of kept positions (k-mer length)
    rc_equivariant : bool
        whether to return canonical reverse-complement k-mers
    n_jobs : int
        number of CPU cores to use (-1 = all)

    Returns
    -------
    kmer_rep : dict
        gene -> list of k-mers
    """

    span = k  # Do not use gapped

    # Precompute mask indices
    masks = generate_masks_with_consecutive_bases(k, span)
    mask_indices = [list(mask) for mask in masks]

    def process_promoter(gene_name, promoter):
        """Compute k-mers for a single promoter."""
        kmers = []
        for i in range(len(promoter) - span + 1):
            window = promoter[i : i + span]
            if not set(window).issubset({"A", "C", "G", "T"}):
                continue
            for mask_idx in mask_indices:
                kmer_chars = ["_"] * span
                for idx in mask_idx:
                    kmer_chars[idx] = window[idx]
                kmer = "".join(kmer_chars)
                if rc_equivariant:
                    kmer = canonical_kmer(kmer)
                kmers.append(kmer)
        return gene_name, kmers

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_promoter)(gene, promoter)
        for gene, promoter in tqdm(promoters.items(), desc="Processing promoters")
    )

    # Combine results into dictionary
    kmer_rep = dict(results)
    return kmer_rep


def generate_kmer_count_vector(promoters: dict, k=6, reverse_complement=True):
    """
    Generate count vectors for gapped or contiguous k-mers with optional RC equivariance,
    vectorized for speed.
    """

    span = k
    # kmer representation per gene
    kmer_rep = make_kmer(promoters, k=k, rc_equivariant=reverse_complement, n_jobs=10)
    # All possible k-mers
    ordered_kmer_list = unique_kmers_gapped(
        k=k, span=span, use_rev_comp=reverse_complement
    )

    print(f"Number of unique k-mers: {len(ordered_kmer_list)}")

    kmer_index = {kmer: idx for idx, kmer in enumerate(ordered_kmer_list)}
    kmer_set_size = len(ordered_kmer_list)

    gene_names = list(kmer_rep.keys())

    assert k < 9, "To big!"
    vec_matrix = np.zeros((len(gene_names), kmer_set_size))
    for i, gene in enumerate(tqdm(gene_names)):
        kmers = kmer_rep[gene]
        indices = [kmer_index[k] for k in kmers if k in kmer_index]
        np.add.at(vec_matrix[i], indices, 1)  # increment counts at repeated indices

    # Convert to dict if needed
    count_vectors = {gene: vec_matrix[i] for i, gene in enumerate(gene_names)}

    return count_vectors, kmer_index
