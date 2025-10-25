import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import json
import pickle
import torch
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from layers import Baseline_EVO2

# Paths
data_dir = "/mnt/nfs/bowei/CHROME/data/cellline_specific/IMR/"
embedding_dir = "/mnt/nfs/bowei/CHROME/data/EVO2_embedding/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/cellline_specific/IMR/EVO_MLP/"
os.makedirs(model_save_path, exist_ok=True)

# Dataset class
class BERTEmbeddingDataset(Dataset):
    def __init__(self, file_paths, embeddings):
        self.data = []
        self.embeddings = embeddings

        for path in file_paths:
            with open(path, "rb") as f:
                self.data.extend(pickle.load(f))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        center_region = entry["center_node_region"]
        label_vector = entry["label_vector"]

        # Retrieve BERT embedding for the center region
        embedding = self.embeddings[center_region]
        embedding = torch.tensor(embedding, dtype=torch.float32)  # Convert to tensor

        return embedding, torch.tensor(label_vector, dtype=torch.float32)

def load_embeddings_for_chr(embedding_dir, chromosome):
    chr_path = os.path.join(embedding_dir, f"{chromosome}_merged.json")
    with open(chr_path, "r") as f:
        embeddings = json.load(f)
    return embeddings

def get_available_chunks(data_dir, chromosome, exclude=[]):
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith(chromosome+"_") and f.endswith(".pkl")
    ]
    chunks = [
        os.path.join(data_dir, f)
        for f in files
        if int(f.split("_chunk_")[-1].split(".")[0]) not in exclude
    ]
    return chunks

def train_and_validate(model, criterion, optimizer, scheduler, val_files, embedding_dir, device, log_file):
    best_val_loss = float("inf")
    log = open(log_file, "a")
    for epoch in range(15):  
        model.train()
        epoch_train_loss = 0
        file_count = 0
        for chromosome in [f"chr{i}" for i in range(1, 23) if i not in [8, 9]] + ["chrX"]:
            print(f"Loading embeddings for {chromosome}...")
            embeddings = load_embeddings_for_chr(embedding_dir, chromosome)
            train_files = get_available_chunks(data_dir, chromosome, exclude=[])
            for chr_train_file in train_files:
                print(f"Processing {chr_train_file}...")
                train_dataset = BERTEmbeddingDataset([chr_train_file], embeddings)
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
                file_train_loss = 0
                for embeddings_batch, labels_batch in train_loader:
                    embeddings_batch = embeddings_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(embeddings_batch)
                    loss = criterion(outputs, labels_batch)
                    loss.backward()
                    optimizer.step()
                    file_train_loss += loss.item()

                file_train_loss /= len(train_loader)
                print(f"Epoch {epoch + 1} Train Loss for {chr_train_file}: {file_train_loss:.4f}")
                log.write(f"Epoch {epoch + 1} Train Loss for {chr_train_file}: {file_train_loss:.4f}\n")
                log.flush()
                epoch_train_loss += file_train_loss
                file_count += 1
                if file_count % 20 == 0: 
                    print(f"Performing validation after {file_count} files...")
                    model.eval()
                    val_loss = 0
                    val_embeddings = load_embeddings_for_chr(embedding_dir, "chr8")
                    val_dataset = BERTEmbeddingDataset(val_files, val_embeddings)
                    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

                    with torch.no_grad():
                        for val_embeddings_batch, val_labels_batch in val_loader:
                            val_embeddings_batch = val_embeddings_batch.to(device)
                            val_labels_batch = val_labels_batch.to(device)

                            val_outputs = model(val_embeddings_batch)
                            val_loss += criterion(val_outputs, val_labels_batch).item()

                    val_loss /= len(val_files)
                    print(f"Epoch {epoch + 1} Validation Loss after {file_count} files: {val_loss:.4f}")
                    log.write(f"Epoch {epoch + 1} Validation Loss after {file_count} files: {val_loss:.4f}\n")
                    log.flush()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pt"))
                        print("Best Model Saved.")
                        log.write("Best Model Saved.\n")
                        log.flush()
        epoch_train_loss /= file_count
        print(f"Epoch {epoch + 1}, Training Loss: {epoch_train_loss:.4f}")
        log.write(f"Epoch {epoch + 1}, Training Loss: {epoch_train_loss:.4f}\n")
        log.flush()
        scheduler.step()
    log.close()

def train_model():
    device = torch.device("cuda:0")
    model = Baseline_EVO2(input_dim=4096, hidden_dim=128, output_dim=751).to(device)
    model_path = os.path.join(model_save_path, "best_model.pt")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path))
    else:
        print("No pre-trained model found, starting training from scratch.")
    optimizer = Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
    criterion = BCEWithLogitsLoss()
    scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
    val_files = get_available_chunks(data_dir, "chr8", exclude=[])
    log_file = os.path.join(model_save_path, "training_log.txt")
    print(f"Validation files: {len(val_files)} chunks.")
    train_and_validate(model, criterion, optimizer, scheduler, val_files, embedding_dir, device, log_file)

if __name__ == "__main__":
    train_model()
