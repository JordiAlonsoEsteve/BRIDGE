import numpy as np
from Bio import Align
from Bio.Seq import Seq


def calculate_motif_ic(pwm, background=None):
    """
    Calculates the total information content of a motif from its PWM.

    Parameters:
    ----------
    pwm : numpy.ndarray or list of lists
        A 2D array of shape (L, 4) where L is the motif length.
        Each row must sum to 1.0 (representing probabilities for A, C, G, T).
    background : list or numpy.ndarray, optional
        Background frequencies for [A, C, G, T].
        Defaults to uniform distribution [0.25, 0.25, 0.25, 0.25].

    Returns:
    -------
    float
        The total information content of the motif in bits.
    """
    pwm = np.asarray(pwm)
    L, num_alphabet = pwm.shape

    if num_alphabet != 4:
        raise ValueError("PWM must have 4 columns corresponding to A, C, G, T.")

    if background is None:
        background = np.array([0.25, 0.25, 0.25, 0.25])
    else:
        background = np.asarray(background)
        if len(background) != 4 or not np.isclose(np.sum(background), 1.0):
            raise ValueError(
                "Background must be a distribution summing to 1 over 4 bases."
            )

    ic_per_position = np.zeros(L)
    for i in range(L):
        pos_freqs = pwm[i, :]
        nonzero = pos_freqs > 0

        # Calculate individual position IC: sum( P_ij * log2(P_ij / B_j) )
        ic_per_position[i] = np.sum(
            pos_freqs[nonzero] * np.log2(pos_freqs[nonzero] / background[nonzero])
        )

    # Return the sum over all positions
    total_ic = np.sum(ic_per_position)
    return total_ic


def get_alingment(kmer_list, weights):
    """
    Performs optimal sliding alignment on a cluster of k-mers with
    Reverse Complement awareness and coefficient weighting.

    NOTE This is alinging to the kmer with the highest absolute value for that TF! This is not
    necessarily the best!
    """
    weights = np.abs(weights)  # just double cheking in case!
    # Note that we do not separate positive and negative because we are assuming that indeed
    # there is only one TFBM per TF. As it is the only coherent thing w.r.t e.g., JASPAR.

    # SET UP THE ALIGNER
    aligner = Align.PairwiseAligner()
    aligner.mode = "global"

    # Prohibit internal gaps to prevent "breaking" k-mers
    aligner.target_internal_gap_score = -100
    aligner.query_internal_gap_score = -100

    # Requires at least 1 overlap. To prevent trivial non-overlapping alingments
    aligner.target_end_gap_score = -0.2
    aligner.query_end_gap_score = -0.2

    # SELECT ANCHOR (Highest Weight)
    anchor_idx = np.argmax(weights)
    anchor = str(kmer_list[anchor_idx])
    self_score = aligner.align(anchor, anchor)[0].score
    # Profile stores: {position: {base: sum_of_weights}}
    profile = {}

    for seq_str, w in zip(kmer_list, weights):
        seq_obj = Seq(str(seq_str))
        # Check Forward and Reverse Complement
        variants = [str(seq_obj), str(seq_obj.reverse_complement())]

        best_variant = variants[0]
        best_shift = 0
        max_score = -float("inf")

        for variant in variants:
            try:
                alignments = aligner.align(anchor, variant)
                if alignments:
                    aln = alignments[0]
                    # Check if there's an actual alignment path
                    if (
                        len(aln.aligned) > 0
                        and len(aln.aligned[0]) > 0
                        and aln.score > max_score
                    ):
                        max_score = aln.score
                        t_indices, q_indices = aln.aligned
                        # Shift = Target start - Query start
                        best_shift = t_indices[0][0] - q_indices[0][0]
                        best_variant = variant
            except (IndexError, ValueError):
                print("Error!")
                continue
        if max_score < 0.25 * self_score:
            continue
        # Map the chosen variant to the weighted profile
        for i, base in enumerate(best_variant):
            pos = i + best_shift
            if pos not in profile:
                profile[pos] = {"A": 0.0, "C": 0.0, "G": 0.0, "T": 0.0}

            b = base.upper()
            if b in "ACGT":
                profile[pos][b] += w
            else:
                # Handle 'N' by distributing weight equally
                for canonical in "ACGT":
                    profile[pos][canonical] += w / 4

    sorted_positions = sorted(profile.keys())
    bases = ["A", "C", "G", "T"]
    n_pos = len(sorted_positions)

    ppm_matrix = np.zeros((n_pos, 4))
    coverage = np.zeros(n_pos)

    for idx, pos in enumerate(sorted_positions):
        for b_idx, base in enumerate(bases):
            # Sum of weights + pseudocount
            ppm_matrix[idx, b_idx] = profile[pos][base]

        # Coverage here is the sum of weights at this position
        coverage[idx] = sum(profile[pos].values())

    consensus_seq = "".join([bases[np.argmax(row)] for row in ppm_matrix])

    # Normalize to probabilities
    ppm_matrix = ppm_matrix / ppm_matrix.sum(axis=1, keepdims=True)

    # --- 5. GENERATE OUTPUTS ---
    consensus_list = [bases[np.argmax(row)] for row in ppm_matrix]
    consensus_seq = "".join(consensus_list)

    return consensus_seq, ppm_matrix, coverage


def generate_PWM_from_theta(theta, ic_threshold=10) -> tuple[dict, dict]:
    """
    Takes theta matrix (|kmers| x |TFs|), TF names and k-mer names.
    Returns dictionary with PWMs {tf_name: pd.DataFrame}
    """
    tf_names = list(theta.columns)
    kmers = list(theta.index)
    theta_values = theta.values
    pwms = {}
    coverages = {}
    ic_distribution = {}
    for col_idx, tf in enumerate(tf_names):
        # 1. Get non-zero rows for this specific TF column
        # These are the k-mers that the model "picked" for this TF
        coefs = theta_values[:, col_idx]
        active_indices = np.where(coefs != 0)[0]

        if len(active_indices) == 0:
            continue

        # 2. Retrieve the relevant k-mers and their weights (absolute coefficients)
        tf_kmers_list = [kmers[i] for i in active_indices]
        tf_kmers_weights = np.abs(coefs[active_indices])

        # 3. Obtain optimal consensus and PPM
        # Note: We pass the weights into our alignment logic
        _, ppm_df, coverage = get_alingment(tf_kmers_list, tf_kmers_weights)
        ic = calculate_motif_ic(ppm_df)
        ic_distribution[tf] = ic
        if ic < ic_threshold:
            print(f"TF {tf} has low IC ({calculate_motif_ic(ppm_df):.2f}), skipping.")
            continue
        pwms[tf] = ppm_df
        coverages[tf] = np.mean(coverage)

    return pwms, ic_distribution


def generate_meme_file(PWM_dict: dict, meme_path):
    """
    Saves a meme file for PWM based on a dictionary {TF_name: np.array(4, k)}
    """
    # Define standard background (equal frequencies)
    bg = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}

    with open(meme_path, "w") as f:
        # 1. Header Information
        f.write("MEME version 4\n")
        f.write("ALPHABET= ACGT\n")
        f.write("strands: + -\n\n")  # Tells TOMTOM both strands are valid

        # 2. Background frequencies
        f.write("Background letter frequencies\n")
        f.write(f"A {bg['A']} C {bg['C']} G {bg['G']} T {bg['T']}\n\n")

        # 3. Motif Entries
        for tf_name, pwm in PWM_dict.items():
            # pwm shape is (4, k)
            width, n_bases = pwm.shape

            f.write(f"MOTIF {tf_name}\n")
            # alength = alphabet length (4), w = motif width
            f.write(f"letter-probability matrix: alength= {n_bases} w= {width}\n")

            # MEME format requires the matrix to be transposed:
            # Each line in the file is one position (column in your PWM)
            # Rows in the file are A C G T probabilities
            pwm_transpose = pwm

            for position in pwm_transpose:
                # Format each probability to 6 decimal places for precision
                line = "  ".join([f"{prob:.6f}" for prob in position])
                f.write(f" {line}\n")
            f.write("\n")  # Blank line between motifs

    print(f"MEME file successfully saved to: {meme_path}")
