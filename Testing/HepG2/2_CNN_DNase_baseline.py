import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch.utils.data import DataLoader, Dataset
from layers import backbone_CNN_Dnase

data_dir = "/mnt/nfs/bowei/CHROME/data/HepG2/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/CNN_DNase_baseline/"
best_model_path = os.path.join(model_save_path, "best_model.pt")
metric_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/CNN_Dnase/"
os.makedirs(metric_save_path, exist_ok=True)
metrics_output_path = os.path.join(metric_save_path, "N_test_metrics_chr9.txt")
roc_plot_path = os.path.join(metric_save_path, "N_roc_curve_chr9.png")
predictions_output_path = os.path.join(metric_save_path, "N_predictions_labels.csv")

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
        dnase_values = entry["center_node_dnase"].toarray()
        if dnase_values.ndim == 1:
            dnase_values = dnase_values[np.newaxis, :]  
        label_vector = entry["label_vector"]
        combined_input = np.concatenate([dna_seq, dnase_values], axis=0)
        return torch.tensor(combined_input, dtype=torch.float32), torch.tensor(label_vector, dtype=torch.float32)

def compute_valid_auroc(y_true, y_pred):
    valid_auroc_scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            valid_auroc_scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.mean(valid_auroc_scores) if valid_auroc_scores else float("nan")

def test_model(test_files, model_path, device, batch_size=16):
    n_labels = 751
    model = backbone_CNN_Dnase(n_labels, seq_length=5000, embed_length=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.eval()
    test_dataset = SpecificDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    all_labels = []
    all_preds = []
    total_loss = 0
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs = seqs.to(device)
            labels = labels.to(device)
            outputs = model(seqs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.append(outputs.sigmoid().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)
    col_names = [f"Label_{i}" for i in range(n_labels)]
    df_predictions = pd.DataFrame(all_preds, columns=[f"Predicted_{c}" for c in col_names])
    df_binary_preds = pd.DataFrame(binary_preds, columns=[f"Binary_{c}" for c in col_names])
    df_labels = pd.DataFrame(all_labels, columns=[f"True_{c}" for c in col_names])
    df_output = pd.concat([df_predictions, df_binary_preds, df_labels], axis=1)
    df_output.to_csv(predictions_output_path, index=False)
    print(f"Saved predictions and labels to {predictions_output_path}")
    valid_columns = [i for i in range(all_labels.shape[1]) if len(np.unique(all_labels[:, i])) > 1]
    valid_auroc_scores = [roc_auc_score(all_labels[:, i], all_preds[:, i]) for i in valid_columns]
    mean_auroc = np.mean(valid_auroc_scores) if valid_auroc_scores else float("nan")
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
    print(f"Metrics saved to {metrics_output_path}")
    plt.figure(figsize=(10, 8))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_labels):
        if len(np.unique(all_labels[:, i])) > 1:
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_preds[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUROC = {roc_auc[i]:.2f})")
    if len(roc_auc) > 0:
        macro_auroc = np.mean(list(roc_auc.values()))
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2, color="gray", label=f"Macro AUROC = {macro_auroc:.2f}")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Multi-Class ROC Curve (One-vs-Rest)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(roc_plot_path)
    print(f"ROC plot saved to {roc_plot_path}")
    plt.show()
test_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("chr9")]

if __name__ == "__main__":
    print(f"Testing on {len(test_files)} files...")
    test_model(test_files, best_model_path, torch.device("cuda:5" if torch.cuda.is_available() else "cpu"))
