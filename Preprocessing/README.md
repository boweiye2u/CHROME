# 🧬 CHROME Input and Preprocessing

CHROME takes as input:
- One-hot encoded DNA sequences  
- DNA accessibility profiles (DNase-seq)  
- Optional pretrained sequence embeddings (Evo2)  

DNA sequences are derived from the **hg38 reference genome**, downloaded from the UCSC Genome Browser:  
👉 https://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/

---

## 🧪 Processing DNase-seq

### Dependencies
- `samtools`  
- `deepTools`  

DNase-seq provides cell line–specific chromatin accessibility signals. BAM files are processed using `samtools` and converted to normalized bigWig tracks using **deepTools `bamCoverage` with RPGC normalization**.

Example commands for processing DNase-seq data (HepG2) are provided in **Code 1–4**.

---

## 🧬 Processing Hi-C (Non-random Contacts)

Non-random chromatin contacts are identified using **CHROMATIX**, a physics-based polymer model:

👉 https://bitbucket.org/aperezrathke/cmx/src/master/

Precomputed non-random contact sets for selected cell lines are available at:  
👉 https://chrompolymerdb.bme.uic.edu/

---

## 🧠 Processing Evo2 Embeddings

Sequence embeddings are generated using the **Evo2 7B model**.  

Example code for converting raw DNA sequences into Evo2 embeddings is provided in the `EVO2_embedding` directory.

---

## 📌 Notes

- All inputs are aligned to the **hg38 reference genome**  
- Preprocessed datasets and embeddings are available via Zenodo (see main README)  