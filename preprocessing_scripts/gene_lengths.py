import re
from collections import defaultdict


def calculate_gene_lengths(gff_file, output_file):
    # Store unique base positions per Gene ID
    gene_bases = defaultdict(set)

    with open(gff_file, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue

            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            start = int(parts[3])
            end = int(parts[4])
            attributes = parts[8]

            # Process every single exon row
            if feature_type == "exon":
                parent_match = re.search(r"Parent=([^;]+)", attributes)
                if parent_match:
                    transcript_id = parent_match.group(1)

                    # STRATEGY: If the ID contains a dot (like AT1G01010.1),
                    # split it and keep everything before the dot to get the true Gene ID.
                    if "." in transcript_id:
                        gene_id = transcript_id.split(".")[0]
                    else:
                        gene_id = transcript_id

                    # Add all base coordinates to the set (collapses overlapping isoforms)
                    gene_bases[gene_id].update(range(start, end + 1))
    # Write the final clean gene-level union exon lengths
    with open(output_file, "w") as out_f:
        out_f.write("Gene_ID\tUnion_Exon_Length_bp\n")
        for gene_id, bases in sorted(gene_bases.items()):
            out_f.write(f"{gene_id}\t{len(bases)}\n")

    print(f"Success! Saved clean gene lengths to: {output_file}")


if __name__ == "__main__":
    calculate_gene_lengths(
        "data/raw/dna_tair10/TAIR10_GFF3_genes.gff",
        "data/raw/dna_tair10/gene_exon_lengths.txt",
    )
    # calculate_gene_lengths(
    #    "data/raw/dna_araport11/Araport11_GFF3_genes_transposons.20250813.gff",
    #    "data/raw/dna_araport11/gene_exon_lengths.txt",
    # )
