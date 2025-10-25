import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import json
import pickle
import torch
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from layers import Baseline_EVO2, GAT_EVO2

data_dir = "/mnt/nfs/bowei/CHROME/data/cellline_specific/IMR/"
embedding_dir = "/mnt/nfs/bowei/CHROME/data/EVO2_embedding/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/cellline_specific/IMR/GAT_EVO/"
os.makedirs(model_save_path, exist_ok=True)

class GATEmbeddingDataset(Dataset):
    def __init__(self, file_paths, embeddings):
        self.data = []
        self.embeddings = embeddings
        self.file_paths = file_paths
        for path in file_paths:
            with open(path, "rb") as f:
                data_chunk = pickle.load(f)
                for entry in data_chunk:
                    entry["file_path"] = path
                self.data.extend(data_chunk)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        file_path = entry["file_path"]
        chromosome = os.path.basename(file_path).split("_")[0] 
        center_region = entry["center_node_region"]
        neighbor_regions = entry["neighbor_node_list"]
        label_vector = entry["label_vector"]
        center_embedding = torch.tensor(self.embeddings[center_region], dtype=torch.float32)
        neighbor_embeddings = torch.stack(
            [torch.tensor(self.embeddings[region], dtype=torch.float32) for region in neighbor_regions],
            dim=0
        )
        node_features = torch.cat([center_embedding.unsqueeze(0), neighbor_embeddings], dim=0)
        num_neighbors = len(neighbor_regions)
        edge_index = torch.tensor(
            [(0, i + 1) for i in range(num_neighbors)] + [(i + 1, 0) for i in range(num_neighbors)],
            dtype=torch.long
        ).t().contiguous()
        return Data(x=node_features, edge_index=edge_index, y=torch.tensor(label_vector, dtype=torch.float32))

def load_embeddings_for_chr(embedding_dir, chromosome):
    chr_path = os.path.join(embedding_dir, f"{chromosome}_merged.json")
    with open(chr_path, "r") as f:
        embeddings = json.load(f)
    return embeddings

def get_available_chunks(data_dir, chromosome, exclude=[]):
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith(chromosome + "_") and f.endswith(".pkl")
    ]
    chunks = [
        os.path.join(data_dir, f)
        for f in files
        if int(f.split("_chunk_")[-1].split(".")[0]) not in exclude
    ]
    return chunks

def train_and_validate(model, criterion, optimizer, scheduler, train_files, val_files, device, log_file, val_embedding, batch_size=16):
    best_val_loss = float("inf")
    chunk_count = 0
    with open(log_file, "w") as log:
        for epoch in range(15):
            model.train()
            train_loss = 0
            for chromosome, files in train_files.items():
                print(f"Training with files from {chromosome}...")
                embeddings = load_embeddings_for_chr(embedding_dir, chromosome)
                for file_path in files:
                    chunk_count += 1
                    print(f"Processing {file_path}...")
                    train_dataset = GATEmbeddingDataset([file_path], embeddings)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                    for batch in train_loader:
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        outputs = model(batch.x, batch.edge_index, batch.batch)
                        if batch.y.dim() > 1:
                            labels = batch.y  
                        else:
                            labels = batch.y.view(outputs.size(0), -1)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    print(f"Epoch {epoch + 1}, File {file_path}, Loss: {loss.item():.4f}")
                    log.write(f"Epoch {epoch + 1}, File {file_path}, Loss: {loss.item():.4f}\n")
                    log.flush()
                    if chunk_count % 20 == 0:
                        model.eval()
                        val_loss = 0
                        with torch.no_grad():
                            for val_file in val_files:
                                val_dataset = GATEmbeddingDataset([val_file], val_embedding)
                                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                                for val_batch in val_loader:
                                    val_batch = val_batch.to(device)
                                    val_outputs = model(val_batch.x, val_batch.edge_index, val_batch.batch)
                                    if val_batch.y.dim() > 1:
                                        val_labels = val_batch.y
                                    else:
                                        val_labels = val_batch.y.view(val_outputs.size(0), -1)
                                    val_loss += criterion(val_outputs, val_labels).item()
                        val_loss /= len(val_files)
                        print(f"Validation Loss after {chunk_count} files: {val_loss:.4f}")
                        log.write(f"Validation Loss after {chunk_count} files: {val_loss:.4f}\n")
                        log.flush()
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pt"))
                            print("Best Model Saved.")
                            log.write("Best Model Saved.\n")
                            log.flush()
            train_loss /= len(train_files)
            print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}")
            log.write(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}\n")
            log.flush()
            scheduler.step()

def train_model():
    device = torch.device("cuda:5")
    pretrained_MLP = Baseline_EVO2(input_dim=4096, hidden_dim=128, output_dim=751)
    state_dict = torch.load(
    "/mnt/nfs/bowei/CHROME/model/cellline_specific/IMR/EVO_MLP/best_model.pt",
    map_location="cuda:7"
    )
    pretrained_MLP.load_state_dict(state_dict)
    pretrained_MLP = pretrained_MLP.to(device)
    pretrained_MLP.eval()
    model = GAT_EVO2(pretrained_MLP).to(device)
    model_path = os.path.join(model_save_path, "best_model.pt")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print("No pre-trained model found, starting training from scratch.")
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    train_files = {f"chr{i}": get_available_chunks(data_dir, f"chr{i}") for i in range(1, 23) if i not in [8, 9]}
    train_files["chrX"] = get_available_chunks(data_dir, "chrX")
    val_files = get_available_chunks(data_dir, "chr8", exclude=[])
    log_file = os.path.join(model_save_path, "training_log.txt")
    val_embedding = load_embeddings_for_chr(embedding_dir, "chr8")
    print(f"validation embedding loaded.")
    train_and_validate(model, criterion, optimizer, scheduler, train_files, val_files, device, log_file, val_embedding )

if __name__ == "__main__":
    train_model()
