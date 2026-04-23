#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge BAM files for specific experiments by cell line.
Skips merging if the merged BAM file already exists.
"""

import os
import subprocess

# Paths
base_path = "/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/"
output_dir = "/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/Merged_BAMs"
os.makedirs(output_dir, exist_ok=True)

# Cell lines and their specific experiments
experiments_to_process = {
    "HepG2": "ENCSR149XIL"
}

# Function to merge BAM files
def merge_bams(bam_files, output_file):
    try:
        merge_command = ["samtools", "merge", "-@", "8", output_file] + bam_files
        subprocess.run(merge_command, check=True)
        subprocess.run(["samtools", "index", output_file], check=True)  # Index the merged BAM file
        print(f"Merged BAM file created: {output_file}")
    except Exception as e:
        print(f"Error merging BAM files: {e}")

for cell_line, experiment in experiments_to_process.items():
    print(f"Processing {cell_line} for experiment {experiment}")

    merged_bam_path = os.path.join(output_dir, f"{cell_line}_{experiment}_merged.bam")
    if os.path.exists(merged_bam_path):
        print(f"Merged BAM file already exists for {cell_line} ({experiment}). Skipping.")
        continue

    experiment_path = os.path.join(base_path, cell_line, experiment, "bams")
    if not os.path.exists(experiment_path):
        print(f"Experiment directory does not exist: {experiment_path}. Skipping {cell_line}.")
        continue

    bam_files = [
        os.path.join(experiment_path, file)
        for file in os.listdir(experiment_path)
        if file.endswith(".bam")
    ]

    if not bam_files:
        print(f"No BAM files found for {cell_line} ({experiment}). Skipping.")
        continue

    merge_bams(bam_files, merged_bam_path)

print("\nProcessing completed.")
