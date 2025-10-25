import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pickle
import torch
import time
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import StepLR
from layers import backbone_CNN  

data_dir = "/mnt/nfs/bowei/CHROME/data/ChIP-seq/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/CNN_seq_baseline/"
os.makedirs(model_save_path, exist_ok=True)

class SpecificDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for path in file_paths:
            with open(path, "rb") as f:
                self.data.extend(pickle.load(f))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        entry = self.data[idx]
        dna_seq = entry["center_node_dna_seq"].toarray()  
        label_vector = entry["label_vector"]
        return dna_seq, label_vector

def get_available_chunks(data_dir, chromosome, exclude=[]):
    files = [
        f for f in os.listdir(data_dir)
        if f.startswith(chromosome) and f.endswith(".pkl")
    ]
    chunks = [
        os.path.join(data_dir, f)
        for f in files
        if int(f.split("_chunk_")[-1].split(".")[0]) not in exclude
    ]
    return chunks

def train_and_validate(model, criterion, optimizer, scheduler, train_files, val_files, device, log_file, best_val_loss):
    log = open(log_file, "a")
    for epoch in range(15):  
        chunk_count = 0
        for chr_train_file in train_files:
            print(f"Training New Loss center seq only CNN with graph label with {chr_train_file}...")
            train_dataset = SpecificDataset([chr_train_file])
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
            model.train()
            chunk_train_loss = 0
            chunk_start_time = time.time()
            for seqs, labels in train_loader:
                seqs = seqs.float().to(device)
                labels = labels.float().to(device)
                optimizer.zero_grad()
                outputs = model(seqs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                chunk_train_loss += loss.item()
            chunk_train_loss /= len(train_loader)
            chunk_duration = time.time() - chunk_start_time
            print(f"Epoch {epoch + 1}, Chunk {chr_train_file}, Train Loss: {chunk_train_loss:.4f}, Time: {chunk_duration:.2f}s")
            log.write(f"Epoch {epoch + 1}, Chunk {chr_train_file}, Train Loss: {chunk_train_loss:.4f}, Time: {chunk_duration:.2f}s\n")
            log.flush()
            chunk_count += 1
            if chunk_count % 20 == 0:
                print(f"Validating after {chunk_count} chunks...")
                model.eval()
                val_loss = 0
                val_dataset = SpecificDataset(val_files)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=True)
                with torch.no_grad():
                    for val_seqs, val_labels in val_loader:
                        val_seqs = val_seqs.float().to(device)
                        val_labels = val_labels.float().to(device)

                        val_outputs = model(val_seqs)
                        val_loss += criterion(val_outputs, val_labels).item()

                val_loss /= len(val_files)
                print(f"Validation Loss: {val_loss:.4f}")
                log.write(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}\n")
                log.flush()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(model_save_path, "best_model.pt"))
                    print("Best Model Saved.")
                    log.write(f"Best Model Saved after {chunk_count} chunks.\n")
                    log.flush()

        scheduler.step()
    log.close()

def train_model():
    device = torch.device("cuda:5")
    n_labels = 751
    model = backbone_CNN(n_labels, seq_length=5000, embed_length=512).to(device)
    best_model_path = os.path.join(model_save_path, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("✅ Loaded existing best model for continued training.")
    else:
        print("🔁 No saved model found. Starting from scratch.")
    optimizer = Adam(model.parameters(), lr=3e-5, weight_decay=1e-6)
    criterion = BCEWithLogitsLoss()
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    train_files = []
    for i in range(1, 23):
        if i not in [8, 9]:
            train_files.extend(get_available_chunks(data_dir, f"chr{i}"))
    train_files.extend(get_available_chunks(data_dir, "chrX"))
    val_files = get_available_chunks(data_dir, "chr8", exclude=[]) 
    log_file = os.path.join(model_save_path, "training_log.txt")
    print(f"Training files: {len(train_files)} chunks.")
    print(f"Validation files: {len(val_files)} chunks.")
    best_val_loss = float("inf")
    train_and_validate(model, criterion, optimizer, scheduler, train_files, val_files, device, log_file, best_val_loss)

if __name__ == "__main__":
    train_model()
