import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pickle
import torch
from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from layers import GAT_CNN_eQTL
from collections import Counter
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

class EQTLDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        x = torch.tensor(entry["GAT_CNN_concat_embedding"], dtype=torch.float).squeeze(0)
        y = torch.tensor(int(np.argmax(entry["label"])), dtype=torch.long)
        return x, y

with open("/mnt/nfs/bowei/CHROME/data/eQTL/imr_with_embeddings_dnase.pkl", "rb") as f:
    gm_data = [pickle.load(f) for _ in range(1299)]

label_indices = [int(np.argmax(entry["label"])) for entry in gm_data]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
device = torch.device("cuda:0")

epochs = 10
batch_size = 32

all_train_losses, all_val_losses = [], []
all_aucs, all_f1s, all_accs = [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(gm_data, label_indices)):
    val_labels = [label_indices[i] for i in val_idx]
    print(f"\n🌀 Fold {fold+1} | Class counts in val set: {Counter(val_labels)}")

    train_dataset = EQTLDataset([gm_data[i] for i in train_idx])
    val_dataset = EQTLDataset([gm_data[i] for i in val_idx])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = GAT_CNN_eQTL().to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                probs = torch.softmax(pred, dim=1).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        y_true = all_labels
        y_prob_class1 = all_preds[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_prob_class1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (y_prob_class1 > best_thresh).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob_class1)
        except ValueError as e:
            print(f"AUROC failed: {e}")
            auc = float('nan')

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} | AUROC: {auc:.4f} | "
              f"Optimal F1: {f1:.4f} | Acc: {acc:.4f} | Threshold: {best_thresh:.4f}")

    model_path = f"/mnt/nfs/bowei/CHROME/model/eQTL/Dnase_GAT_embedding/imr_model_fold_{fold+1}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ GM Model for fold {fold+1} saved to {model_path}")

    fold_train_loss = train_loss / len(train_loader)
    fold_val_loss = val_loss / len(val_loader)
    all_train_losses.append(fold_train_loss)
    all_val_losses.append(fold_val_loss)
    all_aucs.append(auc)
    all_f1s.append(f1)
    all_accs.append(acc)

    metrics_path = f"/mnt/nfs/bowei/CHROME/model/eQTL/Dnase_GAT_embedding/imr_metrics_fold_{fold+1}.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Train Loss: {fold_train_loss:.4f}\n")
        f.write(f"Val Loss: {fold_val_loss:.4f}\n")
        f.write(f"AUROC: {auc:.4f}\n")
        f.write(f"Optimal F1 Score: {f1:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Best Threshold: {best_thresh:.4f}\n")
    print(f"📄 Metrics for fold {fold+1} saved to {metrics_path}")

avg_metrics = {
    "Avg Train Loss": np.mean(all_train_losses),
    "Avg Val Loss": np.mean(all_val_losses),
    "Avg AUROC": np.mean(all_aucs),
    "Avg Optimal F1 Score": np.mean(all_f1s),
    "Avg Accuracy": np.mean(all_accs)
}

avg_path = "/mnt/nfs/bowei/CHROME/model/eQTL/Dnase_GAT_embedding/imr_metrics_average.txt"
with open(avg_path, "w") as f:
    for k, v in avg_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

print("\n📈 Averaged Metrics Saved:")
for k, v in avg_metrics.items():
    print(f"{k}: {v:.4f}")
