import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import json
from evo2 import Evo2
from pyfaidx import Fasta
fasta_path = "/mnt/nfs/bowei/CHROME/data/seq/hg38_UCSC.fa"
save_dir = "/mnt/nfs/bowei/CHROME/data/EVO2_embedding/"
os.makedirs(save_dir, exist_ok=True) 
genome = Fasta(fasta_path)
group_chromosomes = ['chr1', 'chr10', 'chr17', 'chr21']
window_size = 5000
step_size = 1000
chunk_size = 20_000_000  
device = torch.device("cuda:0")
evo2_model = Evo2('evo2_7b')
layer_name = 'blocks.28.mlp.l3'
for chr_name in group_chromosomes:
    chr_length = len(genome[chr_name])
    print(f"Processing {chr_name} (Length: {chr_length} bp)...")
    for chunk_start in range(0, chr_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, chr_length)
        chunk_id = (chunk_start // chunk_size) + 1
        chunk_filename = f"{chr_name}_chunk{chunk_id}.json"
        chunk_path = os.path.join(save_dir, chunk_filename)
        if os.path.exists(chunk_path):
            print(f"Skipping {chunk_filename}, already processed.")
            continue

        print(f"Processing {chr_name} Chunk {chunk_id} ({chunk_start+1}-{chunk_end})...")
        embeddings_dict = {}
        for start in range(chunk_start, chunk_end - window_size + 1, step_size):
            end = start + window_size
            sequence = genome[chr_name][start:end].seq.upper()  
            input_ids = torch.tensor(
                evo2_model.tokenizer.tokenize(sequence), dtype=torch.int
            ).unsqueeze(0).to(device)  
            with torch.no_grad():
                outputs, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
            seq_embedding = embeddings[layer_name].squeeze(0).to(device) 
            pooled_embedding = seq_embedding.mean(dim=0).cpu().tolist() 
            key = f"{start+1}-{end}"  
            embeddings_dict[key] = pooled_embedding
            torch.cuda.empty_cache()
            if start % 100000 == 0:
                print(f"{chr_name} Chunk {chunk_id}: Processed {start}/{chunk_end} bp...")
        with open(chunk_path, "w") as json_file:
            json.dump(embeddings_dict, json_file)
        print(f"Saved {chunk_filename}")

print("Processing Completed!")
