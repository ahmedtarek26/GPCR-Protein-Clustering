# GPCR Protein clustering

This project groups GPCR proteins using both:
- sequence embeddings (ESM2)
- text information (real metadata + generated descriptions for rows without text data in uniprot)

The goal : put proteins with similar biology closer together.

## Quick Start

1. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
2. All work in notebooks section, separated depending on the task.
3. Main outputs are saved in:
	- `data/processed/`
	- `results/prot2func_gpcr_output_v3/`

## Project Flow

- Data exploration
- Extracting text data
- Load labeled and unlabeled GPCR proteins
- Build sequence and text embeddings
- Fuse distances (sequence + text)
- Cluster proteins with HDBSCAN
- Visualize clusters with UMAP


## Resources

- Data source: UniProt
- Core models: ESM2, Sentence-BERT, Flan-T5
- Core libraries: PyTorch, Transformers, scikit-learn, hdbscan, umap-learn, pandas

### Reference papers

- ESM2 mean-pooling reference paper   
	https://academic.oup.com/bib/article/26/4/bbaf434/8242608
- MSA Transformer (ICML, 2021)  
	https://proceedings.mlr.press/v139/rao21a.html
- Prot2Text: Multimodal Protein Function Generation (AAAI 2024)  
	https://arxiv.org/abs/2311.16453
- Prefix-Tuning: Optimizing Continuous Prompts for Generation (ACL 2021)  
	https://arxiv.org/abs/2101.00190
- LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)  
	https://arxiv.org/abs/2106.09685
- Embedding-based protein sequence alignment with clustering and double dynamic programming (Scientific Reports, 2025)  
	https://www.nature.com/articles/s41598-025-23319-x
