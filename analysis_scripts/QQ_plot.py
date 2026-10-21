import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text


def reverse_complement_seq(seq: str) -> str:
    """Returns the reverse complement of a DNA consensus sequence."""
    if not isinstance(seq, str):
        return ""

    complement = {
        "A": "T",
        "T": "A",
        "C": "G",
        "G": "C",
    }
    return "".join(complement.get(base, base) for base in reversed(seq))


def qq_plot_hits_comparison(df_dict, out_folder, suffix=""):

    # Global Style Settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Publication palette
    color_list = [
        "#B2182B",
        "#D6604D",
        "#F4A582",
        "#01070A",
        "#4393C3",
        "#2166AC",
        "#053061",
        "#7B3294",
        "#C2A5CF",
    ]
    markers = ["o", "x", "^", ".", "v", "s"]

    texts = []
    all_obs_log = []
    all_exp_log = []
    processed_dfs = {}

    for i, (label, df) in enumerate(df_dict.items()):
        # Filter hits and validate
        subset = df[df["Query_Gene_ID"] == df["Target_Gene_ID"]].copy()
        if subset.empty:
            continue

        # Sort: Smallest p-values first
        subset = subset.sort_values("p-value", ascending=True)

        pvals = subset["p-value"].values
        n = len(pvals)
        expected = np.arange(1, n + 1) / (n + 1)

        # Transformations
        obs_log = -np.log10(pvals)
        exp_log = -np.log10(expected)

        all_obs_log.extend(obs_log)
        all_exp_log.extend(exp_log)
        processed_dfs[label] = subset

        # Plotting
        current_color = color_list[i % len(color_list)]
        ax.scatter(
            exp_log,
            obs_log,
            s=15,
            alpha=0.7,
            color=current_color,
            marker=markers[i % len(markers)],
            label=label,
            edgecolors="none",
            zorder=3,
        )

        # Labels for top 2 hits
        for j in range(min(1, len(subset))):
            target_id = subset.iloc[j]["Target_Name"]
            query_id = subset.iloc[j]["Query_Name"]
            motif = subset.iloc[j]["Query_consensus"]
            orientation = subset.iloc[j].get("Orientation", "+")
            if orientation == "-" and pd.notna(motif):
                motif = reverse_complement_seq(str(motif))

            # Formatted label string
            label_text = f"{query_id}-{motif}"  # -{target_id}"

            t = ax.text(
                exp_log[j],
                obs_log[j],
                label_text,
                fontsize=8,
                fontweight="medium",
                color="black",
                zorder=5,
            )
            texts.append(t)

    # Axis Limits & Identity Line
    max_val = max(max(all_obs_log), max(all_exp_log)) if all_obs_log else 1
    limit = max_val * 1.05  # Extra room for top labels

    ax.plot(
        [0, limit], [0, limit], color="#444444", linestyle="--", linewidth=1.2, zorder=1
    )

    # Automatic Label Adjustment
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="->",
                color="#666666",
                mutation_scale=3,
                lw=0.8,
                alpha=0.6,
            ),
            expand_points=(8.0, 8.0),
            expand_text=(2.5, 2.5),
            force_points=8.0,
            force_text=1.0,
            lim=1000,
            avoid_self=True,
            ensure_inside_axes=True,
            only_move={"points": "xy", "text": "xy"},
            precision=0.001,
        )

    # Final Polish
    ax.set_xlabel(r"Expected $-\log_{10}(P)$", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Observed $-\log_{10}(P)$", fontsize=12, labelpad=10)

    # Legend: Bottom Right, No Frame, Transparent
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
        markerscale=1.2,
        handletextpad=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(f"{out_folder}/qq_plot_comparison{suffix}.png", dpi=300)

    return processed_dfs


def qq_plot_all(df_dict, out_folder, suffix=""):
    # Global Style Settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Publication palette
    color_list = [
        "#B2182B",
        "#D6604D",
        "#F4A582",
        "#01070A",
        "#4393C3",
        "#2166AC",
        "#053061",
        "#7B3294",
        "#C2A5CF",
    ]

    markers = ["o", "x", "^", ".", "v", "s"]

    texts = []
    all_obs_log = []
    all_exp_log = []
    processed_dfs = {}

    for i, (label, df) in enumerate(df_dict.items()):
        subset = df.copy()

        if subset.empty:
            continue

        # Sort: Smallest p-values first
        subset = subset.sort_values("p-value", ascending=True)

        pvals = subset["p-value"].values
        n = len(pvals)
        expected = np.arange(1, n + 1) / (n + 1)

        # Transformations
        obs_log = -np.log10(pvals)
        exp_log = -np.log10(expected)

        all_obs_log.extend(obs_log)
        all_exp_log.extend(exp_log)
        processed_dfs[label] = subset

        # Plotting
        current_color = color_list[i % len(color_list)]
        ax.scatter(
            exp_log,
            obs_log,
            s=15,
            alpha=0.7,
            color=current_color,
            marker=markers[i % len(markers)],
            label=label,
            edgecolors="none",
            zorder=3,
        )

        # Labels for top 2 hits
    # for j in range(min(2, len(subset))):
    # query_id = subset.iloc[j]["Query_Name"]
    # target_id = subset.iloc[j]["Target_ID"]
    # motif = subset.iloc[j]["Query_consensus"]

    ## Formatted label string
    # label_text = f"{query_id}-{motif}-{target_id}"

    # t = ax.text(
    # exp_log[j],
    # obs_log[j],
    # label_text,
    # fontsize=8,
    # fontweight="medium",
    # color="black",
    # zorder=5,
    # )
    # texts.append(t)

    # Axis Limits & Identity Line
    max_val = max(max(all_obs_log), max(all_exp_log)) if all_obs_log else 1
    limit = max_val * 1.05  # Extra room for top labels

    ax.plot(
        [0, limit], [0, limit], color="#444444", linestyle="--", linewidth=1.2, zorder=1
    )

    # Automatic Label Adjustment
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="->",
                color="#666666",
                mutation_scale=3,
                lw=0.8,
                alpha=0.6,
            ),
            expand_points=(8.0, 8.0),
            expand_text=(2.5, 2.5),
            force_points=8.0,
            force_text=1.0,
            lim=1000,
            avoid_self=True,
            ensure_inside_axes=True,
            only_move={"points": "xy", "text": "xy"},
            precision=0.001,
        )

    # Final Polish
    ax.set_xlabel(r"Expected $-\log_{10}(P)$", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Observed $-\log_{10}(P)$", fontsize=12, labelpad=10)

    # Legend: Bottom Right, No Frame, Transparent
    ax.legend(
        loc="lower right",
        frameon=False,
        fontsize=10,
        markerscale=1.2,
        handletextpad=0.2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(f"{out_folder}/qq_plot_all{suffix}.png", dpi=300)

    return processed_dfs


def qq_plot_family_comparison(df_dict, out_folder, suffix=""):
    """
    Final publication version for Family comparison:
    - Filters where Query_Family == Target_Family
    - Top 2 hits labeled with Family, Target_Name, and Motif
    - Visual evaluation based on deviation from the diagonal
    - AdjustText for non-overlapping labels
    """

    # Global Style Settings
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Publication palette
    color_list = [
        "#B2182B",
        "#D6604D",
        "#F4A582",
        "#01070A",
        "#4393C3",
        "#2166AC",
        "#053061",
        "#7B3294",
        "#C2A5CF",
    ]
    markers = ["o", "x", "^", ".", "v", "s"]

    texts = []
    all_obs_log = []
    all_exp_log = []
    processed_dfs = {}

    for i, (label, df) in enumerate(df_dict.items()):
        # CRITICAL FILTER: Keep only intra-family matches
        subset = df[df["Query_Family"] == df["Target_Family"]].copy()

        if subset.empty:
            continue

        # Sort: Smallest p-values first
        subset = subset.sort_values("p-value", ascending=True)

        pvals = subset["p-value"].values
        n = len(pvals)
        expected = np.arange(1, n + 1) / (n + 1)

        # Transformations
        obs_log = -np.log10(pvals)
        exp_log = -np.log10(expected)

        all_obs_log.extend(obs_log)
        all_exp_log.extend(exp_log)
        processed_dfs[label] = subset

        # Plotting
        current_color = color_list[i % len(color_list)]
        ax.scatter(
            exp_log,
            obs_log,
            s=15,
            alpha=0.7,
            color=current_color,
            marker=markers[i % len(markers)],
            label=label,
            edgecolors="none",
            zorder=3,
        )

        # Labels for top 2 hits
    # for j in range(min(2, len(subset))):
    # family = subset.iloc[j]["Query_Family"]
    # target_name = subset.iloc[j]["Target_Name"]
    # query_id = subset.iloc[j]["Query_Name"]
    # motif = subset.iloc[j]["Query_consensus"]

    ## Updated formatted label string to reflect the Family context
    # label_text = f"[{family}] {query_id}-{motif}-{target_name}"

    # t = ax.text(
    # exp_log[j],
    # obs_log[j],
    # label_text,
    # fontsize=8,
    # fontweight="medium",
    # color="black",
    # zorder=5,
    # )
    # texts.append(t)

    # Axis Limits & Identity Line
    max_val = max(max(all_obs_log), max(all_exp_log)) if all_obs_log else 1
    limit = max_val * 1.05  # Extra room for top labels

    ax.plot(
        [0, limit], [0, limit], color="#444444", linestyle="--", linewidth=1.2, zorder=1
    )

    # Automatic Label Adjustment
    if texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(
                arrowstyle="->",
                color="#666666",
                mutation_scale=3,
                lw=0.8,
                alpha=0.6,
            ),
            expand_points=(8.0, 8.0),
            expand_text=(2.5, 2.5),
            force_points=8.0,
            force_text=1.0,
            lim=1000,
            avoid_self=True,
            ensure_inside_axes=True,
            only_move={"points": "xy", "text": "xy"},
            precision=0.001,
        )

    ax.set_xlabel(r"Expected $-\log_{10}(P)$", fontsize=12, labelpad=10)
    ax.set_ylabel(r"Observed $-\log_{10}(P)$", fontsize=12, labelpad=10)

    # ax.legend(
    #    loc="lower right",
    #    frameon=False,
    #    fontsize=10,
    #    markerscale=1.2,
    #    handletextpad=0.2,
    # )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(f"{out_folder}/qq_plot_family_comparison{suffix}.png", dpi=300)

    return processed_dfs


if __name__ == "__main__":
    import pandas as pd

    # Maps the internal treatment/file-path key to the display name
    # used in the plot legends (matches the manuscript table, same order).
    ath_treatments = {
        "PTI_TPM": "PTI (TPM)",
        "all_hormone_TPM": "Hormone (TPM)",
        "leaf_CPTT": "Leaf (CPTT)",
        "PTI_log2FC": r"PTI ($\log_2$FC)",
        "all_hormone_log2FC": r"Hormone ($\log_2$FC)",
        "PB_snRNA_hormone_log2FC": r"SN Hormone ($\log_2$FC)",
        "PB_snRNA_hormone_CPTT": "SN Hormone (CPTT)",
    }
    human_treatments = {
        "GTX": "GTEx (TPM)",
        "TabulaSapiens": "Tabula Sapiens (CPTT)",
        "PMBC_CPTT": "PBMC (CPTT)",
        "TCGA": "TCGA (TPM)",
    }

    dfs = {}
    for treatment, display_name in ath_treatments.items():
        df = pd.read_csv(f"results/ath/{treatment}/tomtom_df.csv")
        dfs[display_name] = df
    for treatment, display_name in human_treatments.items():
        df = pd.read_csv(f"results/human/{treatment}/tomtom_df.csv")
        dfs[display_name] = df

    # Runs and outputs the cleaned up plots
    comparison_dfs = qq_plot_hits_comparison(dfs, out_folder="results")
    family_dfs = qq_plot_family_comparison(dfs, out_folder="results")
    qq_plot_all(dfs, out_folder="results")
