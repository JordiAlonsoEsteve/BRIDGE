# Instructions to reproduce the figures and results from scratch

While most of the datasets can be comfortably used on a standard laptop with a GPU, Tabula Sapiens, GTEx and TCGA require
considerable VRAM > 7Gb. Furthermore, to get good reproducible results running FISTA at float64 is required (as done in the mansucript).
This is considerably slower in standard hardware. We used H100, which deals with double precision pretty okay.

Pre-processing of TabulaSapiens into pseudo-counts will also require considerable ammount of RAM >40Gb (or modifying the scripts provided here).


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

Generate the exon count per gene to be used later for TPM normalization. 
```bash
python preprocessing_scripts/gene_lengths.py
```

### RNA-seq donwload and processing
We have 7 datasets, each with its own particularities. Let's start with Arabidopsis.
#### *Arabidopsis thaliana*

##### Hormone data
?????????

##### PTI data
```bash
mkdir -p data/raw/rna_seq_ath/PTI
mkdir -p data/processed/mRNA/ath/
wget -O data/raw/rna_seq_ath/PTI/PRJEB25079_UncorrectedCounts.csv https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/694/E-MTAB-9694/Files/PRJEB25079_UncorrectedCounts.csv
python preprocessing_scripts/process_PTI.py
```
With this we have available the log(TPM) and log2FC normalizations.

##### Leaf SC data
```bash
mkdir data/raw/rna_seq_ath/leaf
wget -O data/raw/rna_seq_ath/leaf/leaf.expression-matrix-raw.txt.gz https://www.zmbp-resources.uni-tuebingen.de/timmermans/PSCB3_leaf/leaf.expression-matrix-raw.txt.gz
gunzip data/raw/rna_seq_ath/leaf/leaf.expression-matrix-raw.txt.gz
python preprocessing_scripts/process_SC_leaf_ath.py
```
The last python scripts converts it to clean log(CPTT) counts.
##### SN Hormone data
```bash
mkdir data/raw/rna_seq_ath/SN_hormone
```
Download the data from http://119.45.35.29:14545/ into that folder and run

```bash
python preprocessing_scripts/process_SN_hormone_pseudobulk.py
```
which constructs the log(CPTT) and log2FC datasets.


#### Human
##### SC Tabula Sapiens
Careful! 42GB! You will need a lot of RAM to execute that python script!
```bash
mkdir -p data/raw/rna_seq_human/TabulaSapiens
mkdir -p data/processed/mRNA/human/
wget -O data/raw/rna_seq_human/TabulaSapiens/5a495302-b7cd-4bf9-853e-95627b00bb03.h5ad https://datasets.cellxgene.cziscience.com/1c88f927-bcbb-4bb1-9881-281842945a2d.h5ad
python preprocessing_scripts/process_TS.py
```
This yields the log(CPTT) pseudo-bulk .csv for Tabula Sapiens.

##### GTEx
```bash
mkdir data/raw/rna_seq_human/GTEx
wget -O data/raw/rna_seq_human/GTEx/GTEx.gct https://storage.googleapis.com/adult-gtex/bulk-gex/v11/rna-seq/GTEx_Analysis_2026-05-19_v11_RNASeQCv2.4.3_gene_tpm.gct.gz
python preprocessing_scripts/process_GTEx.py
```
Yields log(TPM) GTEx
##### PBMC
```bash
mkdir data/raw/rna_seq_human/PMBC
wget -O data/raw/rna_seq_human/PMBC/filtered_feature_bc_matrix.tar.gz https://cf.10xgenomics.com/samples/cell-exp/4.0.0/SC3_v3_NextGem_SI_PBMC_10K/SC3_v3_NextGem_SI_PBMC_10K_filtered_feature_bc_matrix.tar.gz
tar -xzf data/raw/rna_seq_human/PMBC/filtered_feature_bc_matrix.tar.gz -C  data/raw/rna_seq_human/PMBC/
rm data/raw/rna_seq_human/PMBC/filtered_feature_bc_matrix.tar.gz
python preprocessing_scripts/process_PBMC.py
```
This yields log(CPTT)

##### TCGA
```bash
mkdir data/raw/rna_seq_human/TCGA
wget -O data/raw/rna_seq_human/TCGA/tcga_RSEM_gene_log2_tpm+0.001.csv.gz https://toil-xena-hub.s3.us-east-1.amazonaws.com/download/tcga_RSEM_gene_tpm.gz
gunzip data/raw/rna_seq_human/TCGA/tcga_RSEM_gene_log2_tpm+0.001.csv.gz
python preprocessing_scripts/process_TCGA.py

```

### Regulation data download and processing
We need putative TFs for each species; For arabidopsis we find those here: For human here:

### Gene families
We need the protein sequences of the genes, which we will use to construct a graph to make clusters which will be used as families.

## Training
With all this we can now run the following script to do hyperparameter tuning and the inference. 
```bash
```
Once that completed, we should have the predictive performance results and the optimal $\Theta$ (among other things). These computed results can be found at ... 
With that, is just a matter of running the TOMTOM and plotting

### Reproduce figures & tables
To generate the figures and tables we run
```bash
```
