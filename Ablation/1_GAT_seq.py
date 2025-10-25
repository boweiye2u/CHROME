#!/usr/bin/env python3
# test_gat_cnn_ablation_chr9_dna_only_toggle_f1opt.py
# DNA-sequence only, chr9 ablation. Toggle true-vs-false (perm) neighbors via USE_PERMUTED.
# Adds per-class F1-optimal thresholds and reports both fixed 0.5 and F1-opt metrics.

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from layers import backbone_CNN, GAT_concatenate_CNN_Embedding_flex

# ----------------------------
# Config
# ----------------------------

# chr9 ablation data (with perm_* fields)
data_dir = "/mnt/nfs/bowei/CHROME/data/ablation/"

# False-control toggle:
#   False -> use REAL neighbors
#   True  -> use PERMUTED neighbors (your “false” control)
USE_PERMUTED = True

# Model/artifact paths (match training)
model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_seq_only/"
best_model_path = os.path.join(model_save_path, "best_model.pt")

suffix = "perm" if USE_PERMUTED else "real"
metrics_output_path      = os.path.join(model_save_path, f"test_metrics_chr9_{suffix}.txt")
roc_plot_path            = os.path.join(model_save_path, f"roc_curve_chr9_{suffix}.png")
predictions_output_path  = os.path.join(model_save_path, f"predictions_labels_chr9_{suffix}.csv")
thresholds_output_path   = os.path.join(model_save_path, f"thresholds_f1opt_chr9_{suffix}.csv")

# Model hyperparams (must match training)
embed_dim   = 512
num_classes = 751
seq_len     = 5000

# ----------------------------
# Graph builder
# ----------------------------

def _to_tensor_seq(sparse_seq):
    """
    Convert scipy.sparse [5000,4] one-hot -> torch [4,5000] for the CNN.
    """
    t = torch.tensor(sparse_seq.toarray(), dtype=torch.float)  # [5000,4]
    return t.T  # [4,5000]

def _build_edges(num_neighbors):
    if num_neighbors <= 0:
        return torch.empty((2, 0), dtype=torch.long)
    # Star graph: center(0) <-> neighbors(1..k)
    edge_list = [(0, i + 1) for i in range(num_neighbors)] + [(i + 1, 0) for i in range(num_neighbors)]
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()

def build_graph_data(chunk_path, use_permuted=False):
    with open(chunk_path, "rb") as f:
        data_chunk = pickle.load(f)

    graphs = []
    for entry in data_chunk:
        # Center
        center_seq = _to_tensor_seq(entry["center_node_dna_seq"])  # [4,5000]

        # Neighbors: real vs permuted (“false control”)
        if use_permuted:
            nb_seq_list = entry.get("perm_neighbor_dna_seq_list", [])
        else:
            nb_seq_list = entry.get("neighbor_node_dna_seq_list", [])

        nb_tensors = [_to_tensor_seq(s) for s in nb_seq_list]

        # Stack node features to [N, 4, 5000]
        node_features = torch.stack([center_seq] + nb_tensors, dim=0)

        # Edges (star)
        edge_index = _build_edges(len(nb_tensors))

        # Labels
        labels = torch.tensor(entry["label_vector"], dtype=torch.float)

        # (Optional) skip center-only graphs if your model assumes neighbors present
        if node_features.size(0) <= 1:
            continue

        graphs.append(Data(x=node_features, edge_index=edge_index, y=labels.unsqueeze(0)))

    return graphs

# ----------------------------
# F1-optimal thresholds
# ----------------------------

def pick_f1_opt_thresholds(all_preds, all_labels, valid_cols, grid_points=201):
    """
    For each class in valid_cols, pick the threshold in [0,1] that maximizes F1.
    Returns: thresholds (np.ndarray num_classes,), f1_per_class (np.ndarray), prec_per_class, rec_per_class
    Classes not in valid_cols get threshold=0.5 and f1/prec/rec = np.nan
    """
    n_classes = all_preds.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=np.float32)
    f1s = np.full(n_classes, np.nan, dtype=np.float32)
    precs = np.full(n_classes, np.nan, dtype=np.float32)
    recs = np.full(n_classes, np.nan, dtype=np.float32)

    tgrid = np.linspace(0.0, 1.0, grid_points)

    for i in valid_cols:
        y = all_labels[:, i].astype(int)
        p = all_preds[:, i]
        # Skip degenerate just in case (shouldn't hit because valid_cols filters it)
        if len(np.unique(y)) < 2:
            continue

        best_f1, best_t, best_prec, best_rec = -1.0, 0.5, 0.0, 0.0
        for t in tgrid:
            pb = (p >= t).astype(int)
            tp = int((pb & y).sum())
            fp = int((pb & (1 - y)).sum())
            fn = int(((1 - pb) & y).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if prec == 0.0 and rec == 0.0:
                f1 = 0.0
            else:
                f1 = 2 * prec * rec / (prec + rec)

            if f1 > best_f1:
                best_f1, best_t, best_prec, best_rec = f1, t, prec, rec

        thresholds[i] = best_t
        f1s[i] = best_f1
        precs[i] = best_prec
        recs[i] = best_rec

    return thresholds, f1s, precs, recs

# ----------------------------
# Testing logic
# ----------------------------

def test_model(test_files, device, use_permuted=False, batch_size=16):
    # Build model
    pretrained_cnn = backbone_CNN(nclass=num_classes, seq_length=seq_len, embed_length=embed_dim).to(device)
    model = GAT_concatenate_CNN_Embedding_flex(
        seq_length=seq_len, embed_dim=embed_dim, num_classes=num_classes, pretrained_cnn=pretrained_cnn
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    model.eval()

    all_labels, all_preds = [], []
    total_loss, total_batches = 0.0, 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for file_path in test_files:
            graphs = build_graph_data(file_path, use_permuted=use_permuted)
            if not graphs:
                continue
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

            for batch in loader:
                # x is already [nodes, 4, 5000]; DO NOT permute
                batch.x = batch.x.permute(0, 2, 1).to(device)
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch)  # logits [B, num_classes]
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                total_batches += 1
                all_preds.append(outputs.sigmoid().cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

    if total_batches == 0:
        print("No batches were processed. Check your input files.")
        return

    # Aggregate
    all_preds  = np.concatenate(all_preds, axis=0)     # [N, C]
    all_labels = np.concatenate(all_labels, axis=0)    # [N, C]

    # Valid columns (both classes present)
    valid_cols = [i for i in range(all_labels.shape[1]) if len(np.unique(all_labels[:, i])) > 1]

    # ---- Metrics with fixed 0.5 threshold ----
    binary_05 = (all_preds >= 0.5).astype(int)
    if len(valid_cols) == 0:
        mean_auroc_05 = mean_f1_05 = mean_prec_05 = mean_rec_05 = float("nan")
        roc_auc_map = {}
    else:
        valid_aurocs = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in valid_cols]
        mean_auroc_05 = float(np.mean(valid_aurocs))
        valid_labels = all_labels[:, valid_cols]
        valid_binary_05 = binary_05[:, valid_cols]
        mean_f1_05 = f1_score(valid_labels, valid_binary_05, average="macro", zero_division=0)
        mean_prec_05 = precision_score(valid_labels, valid_binary_05, average="macro", zero_division=0)
        mean_rec_05 = recall_score(valid_labels, valid_binary_05, average="macro", zero_division=0)
        roc_auc_map = {i: v for i, v in zip(valid_cols, valid_aurocs)}

    # ---- Per-class F1-optimal thresholds ----
    th_opt, f1_per_class, prec_per_class, rec_per_class = pick_f1_opt_thresholds(all_preds, all_labels, valid_cols)
    binary_opt = (all_preds >= th_opt).astype(int)

    # Macro metrics with F1-opt thresholds (only on valid cols to be fair)
    if len(valid_cols) == 0:
        mean_f1_opt = mean_prec_opt = mean_rec_opt = float("nan")
    else:
        valid_binary_opt = binary_opt[:, valid_cols]
        valid_labels = all_labels[:, valid_cols]
        mean_f1_opt   = f1_score(valid_labels, valid_binary_opt, average="macro", zero_division=0)
        mean_prec_opt = precision_score(valid_labels, valid_binary_opt, average="macro", zero_division=0)
        mean_rec_opt  = recall_score(valid_labels, valid_binary_opt, average="macro", zero_division=0)

    # Save metrics
    with open(metrics_output_path, "w") as f:
        f.write(f"Mode: {'PERMUTED' if use_permuted else 'REAL'}\n")
        f.write(f"Batches: {total_batches}\n")
        f.write(f"Test Loss: {total_loss / max(total_batches,1):.4f}\n")
        f.write("\n-- Fixed 0.5 threshold --\n")
        f.write(f"Mean AUROC: {mean_auroc_05:.4f}\n")
        f.write(f"Mean F1: {mean_f1_05:.4f}\n")
        f.write(f"Mean Precision: {mean_prec_05:.4f}\n")
        f.write(f"Mean Recall: {mean_rec_05:.4f}\n")
        f.write("\n-- Per-class F1-opt thresholds --\n")
        f.write(f"Mean F1 (opt): {mean_f1_opt:.4f}\n")
        f.write(f"Mean Precision (opt): {mean_prec_opt:.4f}\n")
        f.write(f"Mean Recall (opt): {mean_rec_opt:.4f}\n")
    print(f"Metrics saved to {metrics_output_path}")

    # Save thresholds
    th_df = pd.DataFrame({
        "class": np.arange(num_classes, dtype=int),
        "threshold_f1opt": th_opt,
        "per_class_f1_at_opt": f1_per_class,
        "per_class_precision_at_opt": prec_per_class,
        "per_class_recall_at_opt": rec_per_class,
        "valid": [int(i in valid_cols) for i in range(num_classes)],
    })
    th_df.to_csv(thresholds_output_path, index=False)
    print(f"Per-class thresholds saved to {thresholds_output_path}")

    # Save predictions & labels (include both binary_05 and binary_opt)
    col_names = [f"Label_{i}" for i in range(num_classes)]
    df_pred   = pd.DataFrame(all_preds,  columns=[f"Predicted_{c}" for c in col_names])
    df_bin05  = pd.DataFrame(binary_05,  columns=[f"Binary05_{c}"    for c in col_names])
    df_binopt = pd.DataFrame(binary_opt, columns=[f"BinaryOpt_{c}"   for c in col_names])
    df_true   = pd.DataFrame(all_labels, columns=[f"True_{c}"       for c in col_names])
    pd.concat([df_pred, df_bin05, df_binopt, df_true], axis=1).to_csv(predictions_output_path, index=False)
    print(f"Saved predictions and labels to {predictions_output_path}")

    # ROC plot (same as before; thresholding doesn’t affect ROC)
    plt.figure(figsize=(10, 8))
    for i in valid_cols[:50]:  # limit for readability
        fpr_i, tpr_i, _ = roc_curve(all_labels[:, i], all_preds[:, i])
        auc_i = auc(fpr_i, tpr_i)
        plt.plot(fpr_i, tpr_i, lw=1, label=f"C{i} AUROC={auc_i:.2f}")
    if len(roc_auc_map) > 0:
        macro_auroc = np.mean(list(roc_auc_map.values()))
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray", label=f"Macro AUROC = {macro_auroc:.2f}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"Multi-Label ROC (chr9, {'perm' if use_permuted else 'real'})")
    plt.legend(loc="lower right", ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=150)
    print(f"ROC plot saved to {roc_plot_path}")
    # plt.show()

# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    test_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                  if f.startswith("chr9") and f.endswith(".pkl")]
    test_files.sort()
    print(f"Mode: {'PERMUTED' if USE_PERMUTED else 'REAL'} neighbors")
    print(f"Testing on {len(test_files)} files from: {data_dir}")
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    test_model(test_files, device, use_permuted=USE_PERMUTED, batch_size=16)
