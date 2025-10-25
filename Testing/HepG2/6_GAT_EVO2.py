import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import json
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from layers import Baseline_EVO2, GAT_EVO2

data_dir = "/mnt/nfs/bowei/CHROME/data/HepG2/"
embedding_dir = "/mnt/nfs/bowei/CHROME/data/EVO2_embedding/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_EVO/"
model_path = os.path.join(model_save_path, "best_model.pt")
metric_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_EVO/"
os.makedirs(metric_save_path, exist_ok=True)
metrics_output_path = os.path.join(metric_save_path, "N_test_metrics_chr9.txt")
roc_plot_path = os.path.join(metric_save_path, "N_roc_curve_chr9.png")
predictions_output_path = os.path.join(metric_save_path, "N_predictions_labels.csv")

class GATEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths, embeddings):
        self.data = []
        self.embeddings = embeddings
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
        center_region = entry["center_node_region"]
        neighbor_regions = entry["neighbor_node_list"]
        label_vector = entry["label_vector"]

        center_embedding = torch.tensor(self.embeddings[center_region], dtype=torch.float32)
        neighbor_embeddings = torch.stack(
            [torch.tensor(self.embeddings[region], dtype=torch.float32) for region in neighbor_regions], dim=0
        )
        node_features = torch.cat([center_embedding.unsqueeze(0), neighbor_embeddings], dim=0)

        edge_index = torch.tensor(
            [(0, i + 1) for i in range(len(neighbor_regions))] + [(i + 1, 0) for i in range(len(neighbor_regions))],
            dtype=torch.long
        ).t().contiguous()

        return Data(x=node_features, edge_index=edge_index, y=torch.tensor(label_vector, dtype=torch.float32))

def load_embeddings_for_chr(embedding_dir, chromosome):
    path = os.path.join(embedding_dir, f"{chromosome}_merged.json")
    with open(path, "r") as f:
        return json.load(f)

def get_available_chunks(data_dir, chromosome):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith(f"{chromosome}_") and f.endswith(".pkl")]

def test_model():
    device = torch.device("cuda:5")
    model = GAT_EVO2(Baseline_EVO2(4096, 128, 751)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    chr9_files = get_available_chunks(data_dir, "chr9")
    embeddings = load_embeddings_for_chr(embedding_dir, "chr9")
    dataset = GATEmbeddingDataset(chr9_files, embeddings)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    criterion = BCEWithLogitsLoss()
    all_preds, all_labels, total_loss = [], [], 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.batch)
            labels = batch.y.view(outputs.size(0), -1).to(device)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(outputs.sigmoid().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)
    valid_columns = [i for i in range(all_labels.shape[1]) if len(np.unique(all_labels[:, i])) > 1]

    mean_auroc = np.mean([roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in valid_columns])
    valid_labels = all_labels[:, valid_columns]
    valid_binary_preds = binary_preds[:, valid_columns]
    mean_f1 = f1_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)
    mean_precision = precision_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)
    mean_recall = recall_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)

    with open(metrics_output_path, "w") as f:
        f.write(f"Test Loss: {total_loss / len(dataloader):.4f}\n")
        f.write(f"Mean AUROC: {mean_auroc:.4f}\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f}\n")
        f.write(f"Mean Recall: {mean_recall:.4f}\n")

    df_preds = pd.DataFrame(all_preds, columns=[f"Predicted_Label_{i}" for i in range(751)])
    df_bin = pd.DataFrame(binary_preds, columns=[f"Binary_Label_{i}" for i in range(751)])
    df_true = pd.DataFrame(all_labels, columns=[f"True_Label_{i}" for i in range(751)])
    df_out = pd.concat([df_preds, df_bin, df_true], axis=1)
    df_out.to_csv(predictions_output_path, index=False)

if __name__ == "__main__":
    test_model()