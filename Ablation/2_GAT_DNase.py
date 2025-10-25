#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch_geometric.data import Data, DataLoader
from layers import backbone_CNN_Dnase, GAT_DNase

data_dir = "/mnt/nfs/bowei/CHROME/data/ablation/"
#   False -> use REAL neighbors
#   True  -> use PERMUTED (degree & distance matched) neighbors
USE_PERMUTED = True

model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_DNase/"
best_model_path = os.path.join(model_save_path, "best_model.pt")

suffix = "perm" if USE_PERMUTED else "real"
metrics_output_path = os.path.join(model_save_path, f"test_metrics_chr9_{suffix}.txt")
roc_plot_path       = os.path.join(model_save_path, f"roc_curve_chr9_{suffix}.png")
predictions_output_path = os.path.join(model_save_path, f"predictions_labels_chr9_{suffix}.csv")
embed_dim = 512
num_classes = 751
def _to_tensor_sparse(x):
    t = torch.tensor(x.toarray(), dtype=torch.float)
    return t

def _stack_center_neighbor(center_seq, center_dnase, neighbor_seqs, neighbor_dnases):
    cseq = _to_tensor_sparse(center_seq) if not torch.is_tensor(center_seq) else center_seq
    cdna = _to_tensor_sparse(center_dnase) if not torch.is_tensor(center_dnase) else center_dnase
    if cdna.ndim == 1:
        cdna = cdna.unsqueeze(0)
    center_node_features = torch.cat([cseq, cdna], dim=0)
    n_features = []
    for s, d in zip(neighbor_seqs, neighbor_dnases):
        ts = _to_tensor_sparse(s) if not torch.is_tensor(s) else s
        td = _to_tensor_sparse(d) if not torch.is_tensor(d) else d
        if td.ndim == 2:
            pass
        elif td.ndim == 1:
            td = td.unsqueeze(0)
        else:
            td = td.squeeze()
            if td.ndim == 1:
                td = td.unsqueeze(0)
        n_features.append(torch.cat([ts, td], dim=0)) 
    node_features = [center_node_features] + n_features
    node_features = torch.stack(node_features, dim=0)  
    return node_features

def _build_edges(num_neighbors):
    if num_neighbors <= 0:
        return torch.empty((2,0), dtype=torch.long)
    edge_list = [(0, i + 1) for i in range(num_neighbors)] + [(i + 1, 0) for i in range(num_neighbors)]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def build_graph_data(chunk_path, use_permuted=False):
    with open(chunk_path, "rb") as f:
        data_chunk = pickle.load(f)
    graphs = []
    for entry in data_chunk:
        labels = torch.tensor(entry["label_vector"], dtype=torch.float)
        center_seq = entry["center_node_dna_seq"]
        center_dnase = entry["center_node_dnase"]
        if use_permuted:
            nb_seq_list = entry.get("perm_neighbor_dna_seq_list", entry.get("perm_neighbor_node_dna_seq_list", []))
            nb_dnase_list = entry.get("perm_neighbor_dnase_list", [])
        else:
            nb_seq_list = entry["neighbor_node_dna_seq_list"]
            nb_dnase_list = entry["neighbor_node_dnase_list"]
        node_features = _stack_center_neighbor(center_seq, center_dnase, nb_seq_list, nb_dnase_list)
        edge_index = _build_edges(len(nb_seq_list))
        graph_data = Data(x=node_features, edge_index=edge_index, y=labels.unsqueeze(0))
        graphs.append(graph_data)
    return graphs

def test_model(test_files, device, use_permuted=False):
    pretrained_cnn = backbone_CNN_Dnase(nclass=num_classes, seq_length=5000, embed_length=embed_dim).to(device)
    model = GAT_DNase(embed_dim=embed_dim, num_classes=num_classes, pretrained_cnn=pretrained_cnn).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    all_labels, all_preds = [], []
    total_loss = 0.0
    total_batches = 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for file_path in test_files:
            graphs = build_graph_data(file_path, use_permuted=use_permuted)
            if len(graphs) == 0:
                continue
            test_loader = DataLoader(graphs, batch_size=16, shuffle=False)
            for batch in test_loader:
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch)  
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                total_batches += 1
                all_preds.append(outputs.sigmoid().cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

    if total_batches == 0:
        print("No batches were processed. Check your input files and parsing.")
        return

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)
    valid_columns = [i for i in range(all_labels.shape[1]) if len(np.unique(all_labels[:, i])) > 1]
    if len(valid_columns) == 0:
        mean_auroc = float("nan")
        mean_f1 = mean_precision = mean_recall = float("nan")
    else:
        valid_auroc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in valid_columns]
        mean_auroc = float(np.mean(valid_auroc_scores))
        valid_labels = all_labels[:, valid_columns]
        valid_binary_preds = binary_preds[:, valid_columns]
        mean_f1 = f1_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)
        mean_precision = precision_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)
        mean_recall = recall_score(valid_labels, valid_binary_preds, average="macro", zero_division=0)

    with open(metrics_output_path, "w") as f:
        f.write(f"Test Loss: {total_loss / max(total_batches,1):.4f}\n")
        f.write(f"Mean AUROC: {mean_auroc:.4f}\n")
        f.write(f"Mean F1 Score: {mean_f1:.4f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f}\n")
        f.write(f"Mean Recall: {mean_recall:.4f}\n")
    print(f"Metrics saved to {metrics_output_path}")
    col_names = [f"Label_{i}" for i in range(num_classes)]
    df_predictions = pd.DataFrame(all_preds, columns=[f"Predicted_{c}" for c in col_names])
    df_binary_preds = pd.DataFrame(binary_preds, columns=[f"Binary_{c}" for c in col_names])
    df_labels = pd.DataFrame(all_labels, columns=[f"True_{c}" for c in col_names])
    df_output = pd.concat([df_predictions, df_binary_preds, df_labels], axis=1)
    df_output.to_csv(predictions_output_path, index=False)
    print(f"Saved predictions and labels to {predictions_output_path}")
  
if __name__ == "__main__":
    test_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("chr9") and f.endswith(".pkl")]
    test_files.sort()
    print(f"Mode: {'PERMUTED' if USE_PERMUTED else 'REAL'} neighbors")
    print(f"Testing on {len(test_files)} files from: {data_dir}")

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    test_model(test_files, device, use_permuted=USE_PERMUTED)
