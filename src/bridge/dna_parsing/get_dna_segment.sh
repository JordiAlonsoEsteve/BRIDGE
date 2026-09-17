#!/bin/bash
set -e
echo "Bash current working directory: $(pwd)"
# Generates 2 files:
#    gene_sequences.fa with the required sequences, for the output folder.
#    _all_coordinates.bed for the input folder. This is required to construct the annotation files.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default values
gene_to_gene="False"
upstream_bp=""
downstream_bp=""
region_type=""
folder_in=""
folder_out=""
use_strand=""   # default: use -s

while getopts "u:d:r:g:i:o:s:" opt; do
    case $opt in
        u) upstream_bp="$OPTARG" ;;
        d) downstream_bp="$OPTARG" ;;
        r) region_type="$OPTARG" ;;
        g) gene_to_gene="$OPTARG" ;;
        i) folder_in="$OPTARG" ;;
        o) folder_out="$OPTARG" ;;
        s) use_strand="$OPTARG" ;;     # now OPTARG will contain the value
        *)
           echo "Usage: $0 -u <upstream_bp> -d <downstream_bp> -r <TSS/TTS> -g <gene_to_gene> -i <folder_in> -o <folder_out> -s <True/False>"
           exit 1 ;;
    esac
done

# folder_in is always required
if [ -z "$folder_in" ]; then
    echo "Error: input folder (-i) is required."
    exit 1
fi

# folder_out is always required
if [ -z "$folder_out" ]; then
    echo "Error: output folder (-o) is required."   
    exit 1
fi

# If gene_to_gene is not True, other arguments are required
if [ "$gene_to_gene" != "True" ]; then
    if [ -z "$upstream_bp" ] || [ -z "$downstream_bp" ] || [ -z "$region_type" ]; then
        echo "Error: upstream, downstream, and region_type are required when gene_to_gene is False."
        exit 1
    fi
fi


# Display the configuration
echo "Selected config:"
if [ "$gene_to_gene" != "True" ]; then
    echo "  Upstream: $upstream_bp"
    echo "  Downstream: $downstream_bp"
    echo "  Region type: $region_type"
fi
echo "  Gene to gene: $gene_to_gene"
echo "  Input folder: $folder_in"
echo "  Output folder: $folder_out"
echo "  Use strand (-s): $use_strand"



echo "-----------------------------------------------------------------------------------------------------------------"
# Get .fasta file name
FASTA=$(ls "$folder_in"/*.{fa,fas,fasta,fsa,fna} 2>/dev/null | head -n 1)

if [[ -z "$FASTA" ]]; then
    echo "No .fasta file found in $folder_in."
else
    echo "Found: .fasta file: $FASTA"
fi

# Get .gff3 or .gff file name
GFF=$(ls "$folder_in"/*.gff{,3} 2>/dev/null | head -n 1)

if [[ -z "$GFF" ]]; then
    echo "No .gff or .gff3 file found in $folder_in."
else
    echo "Found: .GFF file: $GFF"
fi

# Print wich name is the genome to be processed
NAME=$(basename "$(ls "$folder_in"/*.{fa,fas,fasta,fsa,fna} 2>/dev/null | head -n 1)" | sed 's/\..*//')
echo "Processing genome: $NAME"
echo "-----------------------------------------------------------------------------------------------------------------"

# Index the FASTA file
samtools faidx "$FASTA"

# Extract the chromosome lengths from the index file and save to a text file
cut -f1,2 "$FASTA.fai" > "$folder_in/${NAME}_chromosome_lengths.txt"

#                           #
# Gene-to-Gene extraction   #
#                           #

if [ "$gene_to_gene" == "True" ]; then
# --------------------------------------------
# Step 1: Extract gene coordinates using AWK
# --------------------------------------------

# Just get the gene coordinates
awk 'BEGIN{FS=OFS="\t"}
$3=="gene" { #&& ($9 ~ /Note=protein_coding_gene/ || $9 ~ /gene_biotype=protein_coding/ || $9 ~ /gbkey=Gene/) {
    if (match($9, /(ID|gene_id)=([^;]+)/, id)) gene_id=id[2]; else gene_id="NA"
    print $1, $4-1, $5, gene_id, $7
}' "$GFF" > "${folder_in}/${NAME}_genes.bed"

CHROM_LEN_FILE="${folder_in}/${NAME}_chromosome_lengths.txt"


python "${SCRIPT_DIR}/handle_gtg.py" \
    "${folder_in}/${NAME}_genes.bed" \
    "${folder_in}/${NAME}_coordinates.bed" \
    500


rm "${folder_in}/${NAME}_genes.bed"
rm "${folder_in}/${NAME}_chromosome_lengths.txt"

fi

if [ "$gene_to_gene" != "True" ]; then
    if [[ "$region_type" != "TSS" && "$region_type" != "TTS" ]]; then
        echo "Error: -r must be either 'TSS' or 'TTS'. You provided: $region_type"
        exit 1
    fi

    RAW_BED="${folder_in}/${NAME}_coordinates_raw.bed"

    awk -v upstream="$upstream_bp" -v downstream="$downstream_bp" -v region="$region_type" '
    BEGIN {FS=OFS="\t"}
    $3=="gene" { #&& ($9 ~ /Note=protein_coding_gene/ || $9 ~ /gene_biotype=protein_coding/ || $9 ~ /gbkey=Gene/ || $9 ~ /locus_type=protein_coding_gene/) {
        start_pos = $4
        end_pos   = $5

        if (region == "TSS") {
            if ($7 == "+") {
                bed_start = (start_pos > upstream) ? start_pos - upstream - 1 : 0
                bed_end   = start_pos + downstream
            } else {
                bed_start = (end_pos > downstream) ? end_pos - downstream - 1 : 0
                bed_end   = end_pos + upstream
            }
        } else if (region == "TTS") {
            if ($7 == "+") {
                bed_start = (end_pos > downstream) ? end_pos - downstream - 1 : 0
                bed_end   = end_pos + upstream
            } else {
                bed_start = (start_pos > upstream) ? start_pos - upstream - 1 : 0
                bed_end   = start_pos + downstream
            }
        }

        # Extract gene name/ID
        if (match($9, /(ID|Name)=([^;]+)/, arr)) {
            gene_name = arr[2]
        } else {
            gene_name = "NA"
        }

        print $1, bed_start, bed_end, gene_name, ".", $7
    }' "$GFF" > "$RAW_BED"

# ----------------------------
# Ensure BED coordinates do not exceed chromosome length
# ----------------------------
    awk 'BEGIN{OFS="\t"}
    NR==FNR {chrom_len[$1]=$2; next}
    ($1 in chrom_len) {if ($3 > chrom_len[$1]) $3 = chrom_len[$1]; print}' \
        "${folder_in}/${NAME}_chromosome_lengths.txt" "$RAW_BED" > "${folder_in}/${NAME}_coordinates.bed"

    rm "$RAW_BED"
fi


if [ "$use_strand" == "True" ]; then
bedtools getfasta \
    -fi "$FASTA" \
    -bed "${folder_in}/${NAME}_coordinates.bed" \
    -fo "${folder_out}/gene_sequences.fa" \
    -name \
    -s
else
bedtools getfasta \
    -fi "$FASTA" \
    -bed "${folder_in}/${NAME}_coordinates.bed" \
    -fo "${folder_out}/gene_sequences.fa" \
    -name
fi

rm "$FASTA.fai"

