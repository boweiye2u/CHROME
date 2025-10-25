import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
from evo2 import Evo2
from pyfaidx import Fasta

# Ensure only GPU 0 is used


# Path to the FASTA file
fasta_path = "/mnt/nfs/bowei/CHROME/data/seq/hg38_UCSC.fa"

# Path to save embeddings
save_dir = "/mnt/nfs/bowei/CHROME/data/EVO2_embedding/"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Load the FASTA genome
genome = Fasta(fasta_path)

# Define Group 1 chromosomes
group_chromosomes = ['chr1', 'chr10', 'chr17', 'chr21']

# Sliding window parameters
window_size = 5000
step_size = 1000
chunk_size = 20_000_000  # 20MB per chunk

# Explicitly set device to GPU 0
device = torch.device("cuda:0")

# Load Evo2 model
evo2_model = Evo2('evo2_7b')

# Define the layer to extract embeddings from
layer_name = 'blocks.28.mlp.l3'

# Process each chromosome in Group 1
for chr_name in group_chromosomes:
    chr_length = len(genome[chr_name])
    print(f"Processing {chr_name} (Length: {chr_length} bp)...")

    # Process in 20MB chunks
    for chunk_start in range(0, chr_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, chr_length)
        chunk_id = (chunk_start // chunk_size) + 1
        chunk_filename = f"{chr_name}_chunk{chunk_id}.json"
        chunk_path = os.path.join(save_dir, chunk_filename)

        # Skip chunk if already processed
        if os.path.exists(chunk_path):
            print(f"Skipping {chunk_filename}, already processed.")
            continue

        print(f"Processing {chr_name} Chunk {chunk_id} ({chunk_start+1}-{chunk_end})...")

        # Dictionary to store embeddings
        embeddings_dict = {}

        # Sliding window over the chunk
        for start in range(chunk_start, chunk_end - window_size + 1, step_size):
            end = start + window_size
            sequence = genome[chr_name][start:end].seq.upper()  # Extract and uppercase

            # Tokenize & Convert to tensor (Ensure it's on GPU 0)
            input_ids = torch.tensor(
                evo2_model.tokenizer.tokenize(sequence), dtype=torch.int
            ).unsqueeze(0).to(device)  # Move input tensor to GPU 0

            # Forward pass with Evo2 (outputs remain on GPU 0)
            with torch.no_grad():
                outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])

            # Extract embeddings (Shape: [1, 5000, 4096] → Remove batch dim)
            seq_embedding = embeddings[layer_name].squeeze(0).to(device)  # Shape: (5000, 4096)

            # Apply mean pooling across sequence length
            pooled_embedding = seq_embedding.mean(dim=0).cpu().tolist()  # Convert to list for JSON storage

            # Store in dictionary with key "start-end"
            key = f"{start+1}-{end}"  # Convert to 1-based index
            embeddings_dict[key] = pooled_embedding

            # Optional: Free up GPU memory periodically
            torch.cuda.empty_cache()

            # Print progress every 100,000 bp
            if start % 100000 == 0:
                print(f"{chr_name} Chunk {chunk_id}: Processed {start}/{chunk_end} bp...")

        # Save dictionary as JSON file
        with open(chunk_path, "w") as json_file:
            json.dump(embeddings_dict, json_file)

        print(f"Saved {chunk_filename}")

print("🔥 Processing Completed!")
