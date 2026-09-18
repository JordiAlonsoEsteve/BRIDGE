mkdir -p data/raw/dna_tair10
wget -O data/raw/dna_tair10/TAIR10_chr_all.fas.gz https://www.arabidopsis.org/api/download-files/download?filePath=Genes/TAIR10_genome_release/TAIR10_chromosome_files/TAIR10_chr_all.fas.gz 
gunzip data/raw/dna_tair10/TAIR10_chr_all.fas.gz
wget -O data/raw/dna_tair10/TAIR10_GFF3_genes.gff https://www.arabidopsis.org/api/download-files/download?filePath=Maps/gbrowse_data/TAIR10/TAIR10_GFF3_genes.gff
mkdir -p data/raw/rna_seq_ath/PTI
mkdir -p data/processed/mRNA/ath/
wget -O data/raw/rna_seq_ath/PTI/PRJEB25079_UncorrectedCounts.csv https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/694/E-MTAB-9694/Files/PRJEB25079_UncorrectedCounts.csv
wget -O data/raw/dna_tair10/TF_list.txt.gz https://planttfdb.gao-lab.org/download/TF_list/Ath_TF_list.txt.gz
gunzip data/raw/dna_tair10/TF_list.txt.gz
python preprocessing_scripts/gene_lengths.py
python preprocessing_scripts/process_PTI.py