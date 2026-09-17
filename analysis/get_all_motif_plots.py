import os

import logomaker as lm
import matplotlib.pyplot as plt
import pandas as pd
from analysis.motif_plots import parse_all_motifs_from_meme


def generate_logos_from_meme(meme_file_path, mapping, output_folder="logo_plots"):
    """
    Parses a .meme file, renames the motifs using a mapping dictionary,
    and saves a sequence logo for each motif element inside the specified folder.
    """
    # 1. Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # 2. Parse the .meme file to get motif data
    motifs = parse_all_motifs_from_meme(meme_file_path)

    # 3. Rename motifs using the mapping dictionary
    # (Fixed syntax to iterate over the list of tuples returned by your parser)
    motifs_renamed = {mapping.get(k, k): v for k, v in motifs}

    print(f"Loaded {len(motifs_renamed)} unique motif(s) after renaming.")

    # 4. Loop through each Transcription Factor (TF) and its matrix
    for tf_name, matrix in motifs_renamed.items():
        print(f"Generating logo for: {tf_name}...")

        # Sanitize the filename to remove characters unsafe for file systems
        safe_filename = tf_name.replace(" ", "_").replace("/", "-").replace(":", "-")

        # 5. Convert the matrix into a pandas DataFrame if it isn't one already
        # Logomaker expects rows as sequence positions and columns as characters (A, C, G, T)
        if not isinstance(matrix, pd.DataFrame):
            # Assumes matrix is a dictionary of lists or a list of dictionaries/lists
            pwm_df = pd.DataFrame(matrix)
        else:
            pwm_df = matrix

        # 6. Transform probabilities into information content (bits)
        # (Change to from_type='counts' if your matrix contains raw counts instead of probabilities)
        try:
            info_df = lm.transform_matrix(
                pwm_df, from_type="probability", to_type="information"
            )
        except (ValueError, KeyError) as e:
            print(
                f"Could not transform matrix for {tf_name}, using raw matrix values instead. Error: {e}"
            )
            info_df = pwm_df

        # 7. Create the Logomaker plot
        fig, ax = plt.subplots(figsize=(10, 3.5))

        # Draw the logo
        logo = lm.Logo(info_df, ax=ax, fade_below=0.5, shade_below=0.5)

        # Clean up the graph styling
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True)
        # ax.set_title(f"Sequence Logo: {tf_name}", fontsize=14, fontweight='bold')
        ax.set_ylabel("Information (bits)", fontsize=12)
        ax.set_xlabel("Position", fontsize=12)

        plt.tight_layout()

        # 8. Save the plot to the designated folder
        output_path = os.path.join(output_folder, f"{safe_filename}_logo.png")
        plt.savefig(output_path, dpi=300)
        plt.close(fig)  # Close the plot to free memory


# --- Execution ---
if __name__ == "__main__":
    # Replace 'your_motifs.meme' with the path to your actual .meme file
    # meme_file = "results/Ath_bulk/all_hormone_TPM/inferred_motifs.meme"
    # mapping = pd.read_csv("data/processed/regulation/Ath/JASPAR_UNIPROT_mapping.csv")
    # mapping from UniProt to symbol
    # mapping_dict = dict(zip(mapping["UniProt_AC"], mapping["Motif_Name"]))
    # generate_logos_from_meme(meme_file, mapping = mapping_dict, output_folder="results/Ath_bulk/all_hormone_TPM/logo_plots")

    meme_file = "results/human/GTX/inferred_motifs.meme"
    mapping = pd.read_csv("data/processed/regulation/human/JASPAR_UNIPROT_mapping.csv")
    # mapping from UniProt to symbol
    mapping_dict = dict(zip(mapping["UniProt_AC"], mapping["Motif_Name"]))
    generate_logos_from_meme(
        meme_file, mapping=mapping_dict, output_folder="results/human/GTX/logo_plots"
    )
