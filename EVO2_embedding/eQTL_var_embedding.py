import os
import pickle
import torch
import json
from tqdm import tqdm
from evo2 import Evo2

# === Config ===
device = torch.device("cuda:0")
evo2_model = Evo2("evo2_7b")
layer_name = "blocks.28.mlp.l3"

# === Input and Output ===
pkl_paths = {
    "gm": "/mnt/nfs/bowei/epcot/data/eQTL/training_data/processed_gm.pkl",
    "imr": "/mnt/nfs/bowei/epcot/data/eQTL/training_data/processed_imr.pkl",
}
output_dir = "/mnt/nfs/bowei/epcot/data/eQTL/training_data"
os.makedirs(output_dir, exist_ok=True)

# === Processing Function ===
def extract_evo2_embedding(sequence: str) -> list:
    input_ids = torch.tensor(evo2_model.tokenizer.tokenize(sequence), dtype=torch.int).unsqueeze(0).to(device)

    with torch.no_grad():
        _, embeddings = evo2_model(input_ids, return_embeddings=True, layer_names=[layer_name])
        seq_embedding = embeddings[layer_name].squeeze(0)  # shape: (5000, 4096)
        pooled = seq_embedding.mean(dim=0)
        return pooled.cpu().float().numpy()  # float32 numpy array


# === Process and Save ===
for tag, path in pkl_paths.items():
    print(f"🔄 Processing {tag} from {path}...")
    with open(path, "rb") as f:
        entries = pickle.load(f)

    for entry in tqdm(entries, desc=f"Embedding {tag}"):
        seq = entry["center_dna_seq"]  # This is a string like 'ATCG...'
        evo2_embed = extract_evo2_embedding(seq)
        entry["EVO2_var_bin_embedding"] = evo2_embed

    # Save back to same file (overwrite)
    with open(path, "wb") as f:
        pickle.dump(entries, f)

    print(f"✅ Saved updated entries to {path}")
