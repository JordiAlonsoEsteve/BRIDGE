import numpy as np
from bridge.kmer_processing.k_mer_utils import generate_kmer_count_vector


def test_recovers_known_kmer_counts_with_rc():
    """Verify that generate_kmer_count_vector correctly recovers known canonical counts."""
    # Sequence: "ACGTACGT" (length 8, k=3 -> 6 total windows)
    promoters = {"gene_1": "ACGTACGT"}
    k = 3

    count_vectors, kmer_index = generate_kmer_count_vector(
        promoters, k=k, reverse_complement=True
    )
    counts = count_vectors["gene_1"]

    # Explicit expected counts for canonical 3-mers
    expected = {
        "ACG": 4.0,  # From "ACG", "CGT", "ACG", "CGT"
        "GTA": 2.0,  # From "GTA", "TAC"
    }

    # Check non-zero k-mer counts
    for kmer, expected_count in expected.items():
        assert counts[kmer_index[kmer]] == expected_count

    # Check that all other k-mers in the feature space are 0
    non_zero_indices = {kmer_index[k] for k in expected}
    all_indices = set(range(len(counts)))
    zero_indices = all_indices - non_zero_indices

    assert np.all(counts[list(zero_indices)] == 0.0)

    # Total counts must equal total sliding windows (8 - 3 + 1 = 6)
    assert np.sum(counts) == 6.0


def test_recovers_known_kmer_counts_without_rc():
    """Verify exact raw k-mer counts when reverse complement equivariance is turned off."""
    promoters = {"gene_1": "ACGTACGT"}
    k = 3

    count_vectors, kmer_index = generate_kmer_count_vector(
        promoters, k=k, reverse_complement=False
    )
    counts = count_vectors["gene_1"]

    expected = {
        "ACG": 2.0,
        "CGT": 2.0,
        "GTA": 1.0,
        "TAC": 1.0,
    }

    for kmer, expected_count in expected.items():
        assert counts[kmer_index[kmer]] == expected_count

    assert np.sum(counts) == 6.0


def test_recovers_homopolymer_counts():
    """Simple check for repeated single-character k-mers."""
    promoters = {"poly_a": "AAAAA"}  # k=3 -> 3 windows of "AAA"

    count_vectors, kmer_index = generate_kmer_count_vector(
        promoters, k=3, reverse_complement=True
    )
    counts = count_vectors["poly_a"]

    assert counts[kmer_index["AAA"]] == 3.0
    assert np.sum(counts) == 3.0
