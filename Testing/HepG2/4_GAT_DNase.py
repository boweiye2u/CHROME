import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import pickle
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve, auc
from torch_geometric.data import Data, DataLoader
from layers import backbone_CNN_Dnase, GAT_DNase

data_dir = "/mnt/nfs/bowei/CHROME/data/HepG2/"
model_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_DNase/"
best_model_path = os.path.join(model_save_path, "best_model.pt")
metric_save_path = "/mnt/nfs/bowei/CHROME/model/ChIP-seq/GAT_Nase/"
os.makedirs(metric_save_path, exist_ok=True)
metrics_output_path = os.path.join(metric_save_path, "N_test_metrics_chr9.txt")
roc_plot_path = os.path.join(metric_save_path, "N_roc_curve_chr9.png")
predictions_output_path = os.path.join(metric_save_path, "N_predictions_labels.csv")

def build_graph_data(chunk_path):
    with open(chunk_path, "rb") as f:
        data_chunk = pickle.load(f)
    graphs = []
    for entry in data_chunk:
        center_seq = torch.tensor(entry["center_node_dna_seq"].toarray(), dtype=torch.float)
        center_dnase = torch.tensor(entry["center_node_dnase"].toarray(), dtype=torch.float)
        if center_dnase.ndim == 1:
            center_dnase = center_dnase.unsqueeze(0)
        center_node_features = torch.cat([center_seq, center_dnase], dim=0)
        neighbor_seqs = [torch.tensor(seq.toarray(), dtype=torch.float) for seq in entry["neighbor_node_dna_seq_list"]]
        neighbor_dnase = [torch.tensor(dnase.toarray(), dtype=torch.float) for dnase in entry["neighbor_node_dnase_list"]]
        neighbor_dnase = [dnase if dnase.ndim == 2 else dnase.unsqueeze(0) for dnase in neighbor_dnase]
        neighbor_features = [torch.cat([seq, dnase], dim=0) for seq, dnase in zip(neighbor_seqs, neighbor_dnase)]
        node_features = [center_node_features] + neighbor_features
        node_features = torch.stack(node_features, dim=0).permute(0, 1, 2)
        num_neighbors = len(neighbor_features)
        edge_list = [(0, i + 1) for i in range(num_neighbors)] + [(i + 1, 0) for i in range(num_neighbors)]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        labels = torch.tensor(entry["label_vector"], dtype=torch.float)
        graph_data = Data(x=node_features, edge_index=edge_index, y=labels.unsqueeze(0))
        graphs.append(graph_data)
    return graphs

def test_model(test_files, device):
    embed_dim = 512
    num_classes = 751

    pretrained_cnn = backbone_CNN_Dnase(nclass=num_classes, seq_length=5000, embed_length=embed_dim).to(device)
    model = GAT_DNase(embed_dim=embed_dim, num_classes=num_classes, pretrained_cnn=pretrained_cnn).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_labels, all_preds, total_loss = [], [], 0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for file_path in test_files:
            graphs = build_graph_data(file_path)
            test_loader = DataLoader(graphs, batch_size=16, shuffle=False)
            for batch in test_loader:
                batch.x = batch.x.permute(0, 1, 2).to(device)
                batch = batch.to(device)
                outputs = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(outputs, batch.y)
                total_loss += loss.item()
                all_preds.append(outputs.sigmoid().cpu().numpy())
                all_labels.append(batch.y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    binary_preds = (all_preds >= 0.5).astype(int)
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

    col_names = [f"Label_{i}" for i in range(num_classes)]
    df_predictions = pd.DataFrame(all_preds, columns=[f"Predicted_{c}" for c in col_names])
    df_binary_preds = pd.DataFrame(binary_preds, columns=[f"Binary_{c}" for c in col_names])
    df_labels = pd.DataFrame(all_labels, columns=[f"True_{c}" for c in col_names])
    df_output = pd.concat([df_predictions, df_binary_preds, df_labels], axis=1)
    df_output.to_csv(predictions_output_path, index=False)
    print(f"Saved predictions and labels to {predictions_output_path}")

if __name__ == "__main__":
    test_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("chr9")]
    print(f"Testing on {len(test_files)} files...")
    test_model(test_files, torch.device("cuda:5" if torch.cuda.is_available() else "cpu"))
