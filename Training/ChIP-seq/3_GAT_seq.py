import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import random
import pickle
import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from layers import backbone_CNN, GAT_seq_only

def build_graph_data(chunk_path):
    with open(chunk_path, "rb") as f:
        data_chunk = pickle.load(f)
    graphs = []
    for entry in data_chunk:
        center_seq = torch.tensor(entry["center_node_dna_seq"].toarray(), dtype=torch.float)
        neighbor_seqs = [torch.tensor(seq.toarray(), dtype=torch.float) for seq in entry["neighbor_node_dna_seq_list"]]
        node_features = [center_seq] + neighbor_seqs  
        node_features = torch.stack(node_features, dim=0).permute(0, 2, 1)  
        num_neighbors = len(neighbor_seqs)
        center_idx = 0
        edge_list = [(center_idx, i + 1) for i in range(num_neighbors)] + [(i + 1, center_idx) for i in range(num_neighbors)]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        labels = torch.tensor(entry["label_vector"], dtype=torch.float).unsqueeze(0)  
        graph_data = Data(x=node_features, edge_index=edge_index, y=labels)
        graphs.append(graph_data)
    return graphs

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]

def prepare_data(data_dir, exclude=[]):
    train_files = []
    included_chromosomes = []
    for i in list(range(1, 23)) + ["X"]:
        if i not in exclude:
            included_chromosomes.append(i)
            files = [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.startswith(f"chr{i}_chunk_") and f.endswith(".pkl")
            ]
            train_files.extend(files)
            print(f"chr{i}: {len(files)} chunks")
    print(f"Included Chromosomes: {included_chromosomes}")
    return train_files

def train_and_validate(model, criterion, optimizer, scheduler, train_files, val_files, device, log_path, batch_size=16,sample_fraction=1):
    best_val_loss = float("inf")
    best_model_path = os.path.join(os.path.dirname(log_path), "best_model.pt")
    if os.path.exists(best_model_path):
        print("Loading best model for resuming training...")
        state_dict = torch.load(best_model_path, map_location="cuda:7")
        model.load_state_dict(state_dict)
    else:
        print("No best model found. Starting training from scratch.")
    with open(log_path, "w") as log:
        file_count = 0  
        for epoch in range(15):
            sampled_train_files = random.sample(train_files, int(len(train_files) * sample_fraction))
            random.shuffle(sampled_train_files)
            model.train()
            train_loss = 0
            epoch_start = time.time()
            val_losses = []
            for file_path in sampled_train_files:
                file_count += 1
                print(f"Training unflatten concatenation GAT w CNN (graph label) seq only with {file_path}...")
                graphs = build_graph_data(file_path)
                train_loader = DataLoader(GraphDataset(graphs), batch_size=batch_size, shuffle=True, drop_last=True)
                for batch in train_loader:
                    batch.x = batch.x.permute(0, 2, 1)
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(outputs, batch.y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                print(f"Epoch {epoch + 1}, Chunk {file_path}, Loss: {loss.item():.4f}")
                log.write(f"Epoch {epoch + 1}, Chunk {file_path}, Loss: {loss.item():.4f}\n")
                log.flush()
                if file_count % 20 == 0:
                    print(f"Performing validation after processing {file_count} files...")
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for val_file in val_files:
                            val_graphs = build_graph_data(val_file)
                            val_loader = DataLoader(GraphDataset(val_graphs), batch_size=batch_size, shuffle=False)

                            for val_batch in val_loader:
                                val_batch.x = val_batch.x.permute(0, 2, 1)
                                val_batch = val_batch.to(device)
                                val_outputs = model(val_batch.x, val_batch.edge_index, val_batch.batch)
                                val_loss += criterion(val_outputs, val_batch.y).item()
                    val_loss /= len(val_files)
                    val_losses.append(val_loss)
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Validation Loss after {file_count} files: {val_loss:.4f},  LR = {current_lr:.2e}")
                    log.write(f"Validation Loss after {file_count} files: {val_loss:.4f},  LR = {current_lr:.2e}\n")
                    log.flush()
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), best_model_path)
                        print("Best Model Saved.")
                        log.write("Best Model Saved.\n")
                        log.flush()
            min_val_loss = min(val_losses)
            train_loss /= len(sampled_train_files)
            epoch_duration = time.time() - epoch_start
            log.write(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Time: {epoch_duration:.2f}s\n")
            log.flush()
            scheduler.step(min_val_loss)

def train_model(data_dir, model_save_path, model_type="flattened"):
    device = torch.device("cuda:4")
    embed_dim = 512
    num_classes = 751
    pretrained_cnn = backbone_CNN(nclass=num_classes, seq_length=5000, embed_length=embed_dim)
    state_dict = torch.load(
    "/mnt/nfs/bowei/CHROME/model/ChIP-seq/CNN_seq_baseline/best_model.pt",
    map_location="cuda:7"
    )
    pretrained_cnn.load_state_dict(state_dict)
    pretrained_cnn = pretrained_cnn.to(device)
    pretrained_cnn.eval()  
    model = GAT_seq_only(seq_length=5000, embed_dim=embed_dim,  num_classes=num_classes, pretrained_cnn=pretrained_cnn)
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=1,
    verbose=True, min_lr=1e-7
    )
    train_files = prepare_data(data_dir, exclude=[8, 9]) 
    val_files = prepare_data(data_dir, exclude=list(range(1, 8)) + list(range(9, 23)) + ["X"])  
    log_path = os.path.join(model_save_path, "training_log.txt")
    train_and_validate(model, criterion, optimizer, scheduler, train_files, val_files, device, log_path)
    
if __name__ == "__main__":
    data_dir = "/mnt/nfs/bowei/CHROME/data/ChIP-seq/"
    model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_seq_only/"
    os.makedirs(model_save_path, exist_ok=True)
    train_model(data_dir, model_save_path)
