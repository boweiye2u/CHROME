# 🧬CHROME: Chromatin-Structure-Guided Regulatory Modeling

CHROME is a deep learning framework that integrates **physically specific, non-random Hi-C contacts** with **Graph Attention Networks (GATs)** to model cell line–specific regulatory landscapes across multi-megabase chromatin domains. By combining local sequence-based encoders with long-range three-dimensional chromatin structure, CHROME predicts transcription factor binding and histone modification profiles. The learned representations further support downstream variant effect analysis, including eQTL directionality and ClinVar pathogenicity prediction.

---

## 📦 Data and Model Availability

All data and model resources required to reproduce CHROME are available at Zenodo:

👉 **https://doi.org/10.5281/zenodo.17442065**

This repository includes:
- Trained model checkpoints (sequence-only, DNase+sequence, Evo2-based CHROME, and matched baselines)
- Processed non-random chromatin contact sets (GM12878, K562, IMR-90, HepG2 (chr9))
- Training, validation, and test datasets for:
  - ChIP-seq prediction
  - Ablation experiments
  - eQTL analysis
  - ClinVar pathogenicity prediction
- Precomputed Evo2 embeddings

The Zenodo archive provides all data and pretrained models, while this GitHub repository contains the code for preprocessing, training, and evaluation.
---

## 🧬 Non-random Chromatin Contacts

CHROME relies on non-random chromatin contacts identified using **CHROMATIX**, a physics-based polymer model.

- Some precomputed contacts are available at:  
  👉 https://chrompolymerdb.bme.uic.edu/

- CHROMATIX for generating contacts for your own Hi-C data:  
  👉 https://bitbucket.org/aperezrathke/cmx/src/master/

---

## 🚀 Code and Usage

This repository provides code for:
- Data preprocessing
- Model training
- Evaluation pipelines
- eQTL and ClinVar analyses

---





## 🚀 Quick Start (Python 3.9 + Conda)

```bash
# 1️⃣ Clone the repository
git clone https://github.com/boweiye2u/CHROME.git
cd CHROME

# 2️⃣ Create and activate the conda environment
conda create -n chrome python=3.9
conda activate chrome

# 3️⃣ Install dependencies
pip install -r requirements.txt
```

---

## 📊 CHROME Overview

### Architecture and Evaluation

**Figure 1. CHROME architecture and evaluation overview.**

![CHROME architecture overview](figures/1_AB.PNG)
![CHROME performance comparison](figures/1_C.PNG)

(A) illustrates how physically specific, non-random chromatin contacts are identified and constructed from Hi-C data.
(B) shows how CHROME builds structure-aware, signal-centered graphs using these non-random contacts to represent spatially connected genomic loci surrounding each target region.
(C) depicts the node features of the graph—comprising sequence, DNase, or Evo2 embeddings—and the Graph Attention Network (GAT) layers used to predict cell-line-specific ChIP-seq signals.

---


## 🧠 Dependencies (Python 3.9)

Core packages:

```
torch
numpy
torch-geometric
scikit-learn
scipy
h5py
```

---

## 📄 Citation

If you use **CHROME** in your work, please cite the published article:

> **Ye, B., Du, L., Chen, M., Dai, Y., Ma, A., & Liang, J. (2026).**  
> *A chromatin-structure-guided framework for predictive and interpretable regulatory genomics.*  
> **Briefings in Bioinformatics, 27**(4), bbag360.  
> https://doi.org/10.1093/bib/bbag360

- [📖 Published article](https://academic.oup.com/bib/article/27/4/bbag360/8736954)
- [📄 PDF](https://academic.oup.com/bib/article-pdf/27/4/bbag360/69204600/bbag360.pdf)

### BibTeX

```bibtex
@article{ye2026chrome,
    author  = {Ye, Bowei and Du, Lin and Chen, Min and Dai, Yang and Ma, Ao and Liang, Jie},
    title   = {A chromatin-structure-guided framework for predictive and interpretable regulatory genomics},
    journal = {Briefings in Bioinformatics},
    volume  = {27},
    number  = {4},
    pages   = {bbag360},
    year    = {2026},
    month   = {07},
    doi     = {10.1093/bib/bbag360},
    url     = {https://doi.org/10.1093/bib/bbag360}
}
## 📬 Contact

For questions or issues, please open a GitHub issue.

