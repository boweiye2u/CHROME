import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pickle
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from layers import Baseline_EVO2

# Paths
data_dir = "/mnt/nfs/bowei/CHROME/data/ChIP-seq/"
embedding_dir = "/mnt/nfs/bowei/CHROME/data/EVO2_embedding/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/EVO_MLP/"
model_path = os.path.join(model_save_path, "best_model.pt")
metrics_output_path = os.path.join(model_save_path, "test_metrics_chr9.txt")
roc_plot_path = os.path.join(model_save_path, "roc_curve_chr9.png")
predictions_output_path = os.path.join(model_save_path, "predictions_labels_chr9.csv")

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
        embedding = torch.tensor(self.embeddings[entry["center_node_region"]], dtype=torch.float32)
        label_vector = torch.tensor(entry["label_vector"], dtype=torch.float32)
        return embedding, label_vector

def load_embeddings_for_chr(embedding_dir, chromosome):
    path = os.path.join(embedding_dir, f"{chromosome}_merged.json")
    with open(path, "r") as f:
        return json.load(f)

def get_chr_files(data_dir, chromosome):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir)
            if f.startswith(chromosome + "_") and f.endswith(".pkl")]

def test_model():
    device = torch.device("cuda:5")
    model = Baseline_EVO2(input_dim=4096, hidden_dim=128, output_dim=751).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    chr9_files = get_chr_files(data_dir, "chr9")
    embeddings = load_embeddings_for_chr(embedding_dir, "chr9")
    test_dataset = BERTEmbeddingDataset(chr9_files, embeddings)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_preds, all_labels = [], []
    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item()
            all_preds.append(out.sigmoid().cpu().numpy())
            all_labels.append(y.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    binary_preds = (all_preds >= 0.5).astype(int)
    valid_columns = [i for i in range(all_labels.shape[1]) if len(np.unique(all_labels[:, i])) > 1]

    mean_auroc = np.mean([
        roc_auc_score(all_labels[:, i], all_preds[:, i])
        for i in valid_columns
    ])
    valid_labels = all_labels[:, valid_columns]
    valid_binary_preds = binary_preds[:, valid_columns]
    mean_f1 = f1_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)
    mean_precision = precision_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)
    mean_recall = recall_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)

    with open(metrics_output_path, "w") as f:
        f.write(f"Test Loss: {total_loss / len(test_loader):.4f}\n")
        f.write(f"Mean AUROC: {mean_auroc:.4f}\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f}\n")
        f.write(f"Mean Recall: {mean_recall:.4f}\n")
    print(f"Saved metrics to {metrics_output_path}")

    df_preds = pd.DataFrame(all_preds, columns=[f"Predicted_Label_{i}" for i in range(751)])
    df_bin = pd.DataFrame(binary_preds, columns=[f"Binary_Label_{i}" for i in range(751)])
    df_true = pd.DataFrame(all_labels, columns=[f"True_Label_{i}" for i in range(751)])
    df_out = pd.concat([df_preds, df_bin, df_true], axis=1)
    df_out.to_csv(predictions_output_path, index=False)
    print(f"Saved predictions to {predictions_output_path}")

if __name__ == "__main__":
    test_model()
