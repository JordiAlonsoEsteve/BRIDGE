![Python](https://img.shields.io/badge/python-3.10+-blue)
# 🧬 Bilinear Regulatory Inference from DNA and Gene Expression (BRIDGE) 🧬
Discover transcription factor binding motifs de novo from RNA-seq — like a Protein Binding Microarray, computed in silico from the data you already have.
![BRIDGE Architecture](images/Overview.png)

## 🔍 What is this?
Transcription factors (TFs) regulate gene expression by binding specific DNA sequences in promoters — but figuring out which TFs bind where, and how that shapes expression, usually requires separate computational approaches: gene regulatory network (GRN) inference for TF-gene relationships, or sequence-to-expression (S2E) models for cis-regulatory motifs. These are typically fit independently per gene or condition, obscuring shared regulatory structure.

BRIDGE unifies both into a single convex optimization problem. It factorizes gene expression as an explicit interaction between promoter k-mers (cis) and TF abundance (trans), recovering both regulatory network structure and de novo TF binding motifs (TFBMs) from the same model — with a guaranteed unique global optimum.

Because it only needs RNA-seq, a genome (FASTA), and an annotation (GFF) — data you likely already have if you're doing transcriptomics — BRIDGE effectively acts as an *in silico* Protein Binding Microarray, TFBMs without prior motif libraries or additional wet-lab assays.

## 🛠️ Installation

Before starting, ensure the following tools are installed:
- [`BEDTools`]
- [`SAMtools`]
- [`BLASTp`]
- [`TOMTOM` (meme suite)]

1. Clone the repository
2. (Recommended) Create and activate a virtual environment. For example:
```bash
conda create -n bridge python=3.11 -y
conda activate bridge
```
3. Install the package in editable mode
```bash
pip install --upgrade pip
pip install -e .
```
This installs the `bridge` package (from `src/bridge`) along with its dependencies, and lets you edit the source code without reinstalling.

## 🔬 Usage Demo: RNA-seq + DNA → Inferred TF-Specific Binding Motifs (.MEME)

## Code overview

## 📊 Manuscript's figure reproduction
See [`REPRODUCE.md`](REPRODUCE.md) for full instructions (data download, preprocessing, and per-figure scripts).
