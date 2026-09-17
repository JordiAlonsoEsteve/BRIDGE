# Instructions to reproduce the figures and results from scratch

While most of the datasets can be comfortably used on a standard laptop with a GPU, Tabula Sapiens, GTEx and TCGA require
considerable VRAM > 7Gb. Furthermore, to get good reproducible results running FISTA at float64 is required (as done in the mansucript).
This is considerably slower in standard hardware. We used H100, which deals with double precision pretty okay.

Furthermore, pre-processing of TabulaSapiens into pseudo-counts will also require considerable ammount of RAM >40Gb (or modifying the scripts provided here).

## Data processing
In this step we make the .fasta, .gff and the RNA-seq data ready for the math. We want to create a folder for each annotation, where the .fasta, .gff, the .txt of putative TFs and, optionally, the gene families in .csv format (for the family aware splitting).

All commands relative to the BRIDGE folder.
### DNA data downloading and processing
#### *Arabidopsis thaliana*
Generate a folder for TAIR10 and Araport11. 
```bash
mkdir -p data/raw/dna_tair10
mkdir data/raw/dna_araport11
```
Download the DNA:
```bash
wget -O data/raw/dna_tair10/TAIR10_chr_all.fas.gz https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas.gz 
gunzip data/raw/dna_tair10/TAIR10_chr_all.fas.gz
cp data/raw/dna_tair10/TAIR10_chr_all.fas data/raw/dna_araport11/
```
Download annotations:
```bash
wget -O data/raw/dna_tair10/TAIR10_GFF3_genes.gff https://www.arabidopsis.org/api/download-files/download?filePath=Maps/gbrowse_data/TAIR10/TAIR10_GFF3_genes.gff
wget -O data/raw/dna_araport11/Araport11_GFF3_genes_transposons.20250813.gff.gz https://www.arabidopsis.org/api/download-files/download?filePath=Genes/Araport11_genome_release/Araport11_GFF3_genes_transposons.20250813.gff.gz
gunzip data/raw/dna_araport11/Araport11_GFF3_genes_transposons.20250813.gff.gz
```
Generate the exon count per gene to be used later for TPM normalization. We only need it for tair10:
```bash
python preprocessing_scripts/gene_lengths.py
```
#### Human
Generate the folder:

```bash
mkdir data/raw/dna_human38
```
Download the DNA:
```bash
wget -O data/raw/dna_human38/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz https://ftp.ensembl.org/pub/release-112/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
gunzip data/raw/dna_human38/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
```
Download the annotation:
```bash
wget -O data/raw/dna_human38/gencode.v38.annotation.gff3.gz https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_38/gencode.v38.annotation.gff3.gz
gunzip data/raw/dna_human38/gencode.v38.annotation.gff3.gz
```
Here we do require some processing to make the pipeline work smoothly, adjusting the annotation format to something similar to the one in Arabidopsis.

The following changes the chr format, selects one particular model HAVANA and selects the first occurence of a particular geneID. 
```bash
awk 'BEGIN {FS="\t"; OFS="\t"} 
/^#/ {next} 
{
    # Standardize to "Chr"
    gsub(/^chr/, "Chr", $1)
    if ($1 !~ /^Chr/) $1 = "Chr" $1

    # Process only HAVANA lines with a geneID
    if ($2 == "HAVANA" && $9 ~ /gene_id=/) {
        
        # Extract the geneID up to the dot
        match($9, /gene_id=([^.;]+)/, arr)
        gid = arr[1]
        
        # Only take the FIRST gene we see for each gid
        if ($3 == "gene" && !seen[gid]++) {
            $3 = "gene"
            $9 = "ID=" gid ";Name=" gid
            print $0
        } 
    }
}' data/raw/dna_human38/gencode.v38.annotation.gff3 > data/raw/dna_human38/my_annotation.gff
```
Now, delete the old version or move it away (no more than one .fasta or .gff per DNA folder!)

```bash
rm data/raw/dna_human38/gencode.v38.annotation.gff3 
```
Great! DNA side is ready!

### RNA-seq donwload and processing
We have 7 datasets, each with its own particularities. Let's start with Arabidopsis.
#### *Arabidopsis thaliana*
##### Hormone data
?????????
##### PTI data
```bash
mkdir -p data/raw/rna_seq_ath/PTI
wget -O data/raw/rna_seq_ath/PTI/PRJEB25079_UncorrectedCounts.csv https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/694/E-MTAB-9694/Files/PRJEB25079_UncorrectedCounts.csv
```
##### Leaf SC data

##### SN Hormone data

#### Human


### Gene regulation data download and processing

## Training

### Run hyperparameter tuning and training

## Plotting

### Reproduce figures & tables