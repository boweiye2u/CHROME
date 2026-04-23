#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced script to ensure completeness of downloaded BAM files.
"""

import requests
from collections import defaultdict
import os

def fetch_experiment_data(url):
    headers = {'accept': 'application/json'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        experiments = []
        for experiment in data["@graph"]:
            accession = experiment["accession"]
            frip_score = None
            for metric in experiment.get("quality_metrics", []):
                if "frip" in metric:
                    frip_score = metric["frip"]
                    break
            experiments.append((accession, frip_score))
        return experiments
    else:
        print(f"Failed to retrieve data from {url}: {response.status_code}")
        return []

def fetch_bam_files(accession, target_genome="GRCh38"):
    url = f'https://www.encodeproject.org/experiments/{accession}/?format=json'
    headers = {'accept': 'application/json'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        bam_files = []
        for file in data.get("files", []):
            if (file["file_format"] == "bam" and 
                file["output_type"] == "alignments" and 
                file.get("assembly") == target_genome):  
                bam_files.append({
                    "url": f"https://www.encodeproject.org{file['href']}",
                    "file_size": file.get("file_size", 0)
                })
        return bam_files
    else:
        print(f"Failed to retrieve BAM files for {accession}: {response.status_code}")
        return []

def create_directory_structure(base_path, cell_line, target_name):
    path = os.path.join(base_path, "DNase", cell_line, target_name, 'bams')
    os.makedirs(path, exist_ok=True)
    return path

def is_file_complete(file_path, expected_size):
    """
    Check if a file is complete by comparing its size with the expected size.
    """
    if os.path.exists(file_path):
        actual_size = os.path.getsize(file_path)
        return actual_size == expected_size
    return False

def download_file(url, destination_path, expected_size=None):
    """
    Download a file, ensuring its completeness.
    """
    # Check if file already exists and is complete
    if expected_size and is_file_complete(destination_path, expected_size):
        print(f"File already exists and is complete: {destination_path}")
        return

    # Delete incomplete file if it exists
    if os.path.exists(destination_path):
        print(f"Incomplete file detected, deleting: {destination_path}")
        os.remove(destination_path)

    # Download the file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {destination_path}")
    else:
        print(f"Failed to download {url}: {response.status_code}")

def download_bam_files(bam_dict, base_path):
    """
    Download BAM files, ensuring completeness.
    """
    for cell_line, experiments in bam_dict.items():
        for accession, frip_score in experiments:
            bam_files = fetch_bam_files(accession, target_genome="GRCh38")
            for bam_info in bam_files:
                bam_url = bam_info["url"]
                expected_size = bam_info.get("file_size")
                folder_path = create_directory_structure(base_path, cell_line, accession)
                file_name = bam_url.split('/')[-1]
                destination_path = os.path.join(folder_path, file_name)
                download_file(bam_url, destination_path, expected_size)

# URLs for each DNase-seq dataset
urls = {
    # 'IMR-90': 'https://www.encodeproject.org/search/?type=Experiment&assay_title=DNase-seq&biosample_ontology.term_name=IMR-90&format=json&limit=all',
    # 'GM12878': 'https://www.encodeproject.org/search/?type=Experiment&assay_title=DNase-seq&biosample_ontology.term_name=GM12878&format=json&limit=all',
    # 'K562': 'https://www.encodeproject.org/search/?type=Experiment&assay_title=DNase-seq&biosample_ontology.term_name=K562&format=json&limit=all'
    'HepG2': 'https://www.encodeproject.org/search/?type=Experiment&assay_title=DNase-seq&biosample_ontology.term_name=HepG2&format=json&limit=all'
}

# Fetch the data for each cell line and organize into dictionaries
dnase_bam_dict = defaultdict(list)

for cell_line, url in urls.items():
    experiments = fetch_experiment_data(url)
    dnase_bam_dict[cell_line].extend(experiments)

# Base path for organizing the downloads
base_path = '/mnt/nfs/bowei/CHROME/data/HepG2_Data/Bam/Dnase/'

# Download all BAM files for DNase-seq for each cell line
print("\nDownloading all BAM files for DNase-seq...")
download_bam_files(dnase_bam_dict, base_path)

print("\nAll downloads complete.")
