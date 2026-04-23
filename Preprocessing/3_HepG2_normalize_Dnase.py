#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

def generate_coverage_tracks(input_path, output_path, genome_size="2559804523", bin_size=1, blacklist_file="black_list.bed"):
    """
    Generates BigWig coverage files from merged BAM files, with normalization and blacklist filtering.

    Parameters:
    - input_path: str, path to the directory containing merged BAM files.
    - output_path: str, path to the directory where BigWig files will be saved.
    - genome_size: str, effective genome size (default "2559804523" for hg38 without chrX and chrM).
    - bin_size: int, bin size for coverage calculation (default 1 bp for high-resolution DNase-seq).
    - blacklist_file: str, path to the blacklist file (e.g., ENCODE blacklist).
    """
    os.makedirs(output_path, exist_ok=True)

    # Process each merged BAM file in the input directory
    for file_name in os.listdir(input_path):
        if file_name.endswith("_merged.bam"):
            bam_file_path = os.path.join(input_path, file_name)
            bigwig_file_name = file_name.replace(".bam", ".bigWig")
            bigwig_file_path = os.path.join(output_path, bigwig_file_name)

            # Run bamCoverage to generate BigWig file with blacklist filtering
            bamcoverage_command = [
                "bamCoverage",
                "--bam", bam_file_path,
                "--outFileName", bigwig_file_path,
                "--outFileFormat", "bigwig",
                "--normalizeUsing", "RPGC",  # Normalizes to Reads Per Genomic Content
                "--effectiveGenomeSize", genome_size,
                "--binSize", str(bin_size),
                "--Offset", "1",  # 1 bp offset for DNase-seq cut site alignment
                "--blackListFileName", blacklist_file,
                "--ignoreForNormalization", "chrX chrM",  # Exclude chrX and chrM from normalization
                "--numberOfProcessors", "24",  # Adjust based on your system
                "--skipNonCoveredRegions"  # Skip regions without coverage
            ]

            print(f"Generating BigWig file for {bam_file_path}...")
            subprocess.run(bamcoverage_command, check=True)
            print(f"BigWig file created: {bigwig_file_path}")

# Paths to the input merged BAM directory and output BigWig directory
input_path = "/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/Merged_BAMs/"
output_path = "/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/normalized/"

# Generate coverage tracks for each merged BAM file with blacklist filtering
generate_coverage_tracks(input_path, output_path, genome_size="2559804523", bin_size=1, blacklist_file="/home/boweiye2/Dropbox/Non-coding-variant/CHROME/black_list.bed")
