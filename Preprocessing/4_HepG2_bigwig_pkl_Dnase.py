#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import pyBigWig
import numpy as np
from multiprocessing import Pool
from scipy.sparse import csr_matrix

# Input and output directories
input_dir = "/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/normalized/"
output_dir = "/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/pkl/"
os.makedirs(output_dir, exist_ok=True)

CHUNK_SIZE = 1000000  # Process data in 1Mb chunks for efficiency

def process_bigwig(bigwig_file):
    """
    Processes a DNase-seq BigWig file and saves the data as a sparse matrix in a pickle file.
    """
    print(f"Processing: {bigwig_file}")
    bw = pyBigWig.open(bigwig_file)
    
    signals = {}
    
    for chrom, length in bw.chroms().items():
        try:
            # Convert chromosome format (e.g., "chrX" → "X", "chr1" → 1)
            chrom_key = "X" if chrom == "chrX" else int(chrom[3:])
        except Exception:
            continue  # Skip invalid chromosomes
        
        # Initialize an empty array for the chromosome
        temp = np.zeros(length, dtype=np.float32)

        # Read signal values from BigWig
        intervals = bw.intervals(chrom)
        for interval in intervals:
            temp[interval[0]:interval[1]] = interval[2]

        # Trim to the nearest 1kb boundary
        seq_length = length // 1000 * 1000
        signals[chrom_key] = csr_matrix(temp[:seq_length])

        print(f"Processed {chrom} ({seq_length} bp) - Mean Signal: {np.mean(signals[chrom_key])}")

    bw.close()

    # Generate output filename
    output_filename = os.path.join(output_dir, os.path.basename(bigwig_file).replace(".bigWig", "_signal.pkl"))
    
    # Save to pickle
    with open(output_filename, "wb") as file:
        pickle.dump(signals, file)

    print(f"Saved: {output_filename}")

if __name__ == "__main__":
    # List all BigWig files
    bigwig_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".bigWig")]
    
    print(f"Found {len(bigwig_files)} BigWig files to process.")

    # Process files in parallel using multiprocessing
    with Pool(processes=28) as pool:  # Adjust based on CPU cores
        pool.map(process_bigwig, bigwig_files)

    print("✅ All BigWig files processed successfully!")
