#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pickle
import torch
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from layers import CNN_Baseline_eQTL  
from cv_saver import CVFoldSaver

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

DEVICE     = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
INPUT_DIR  = "/mnt/nfs/bowei/CHROME/data/ClinVar/enriched_chunks_GAT_CNN_seq_only/"
SAVE_DIR   = "/mnt/nfs/bowei/CHROME/model/ClinVar/CNN_baseline_seq_MLP/"
os.makedirs(SAVE_DIR, exist_ok=True)
CV_OUT_DIR = "/mnt/nfs/bowei/CHROME/model/ClinVar/"
CV_TAG     = "CNN_baseline_seq_MLP"
saver      = CVFoldSaver(CV_OUT_DIR, CV_TAG)

N_SPLITS   = 5
EPOCHS     = 12
BATCH_SIZE = 128
LR         = 1e-3
WD         = 1e-4
EMBED_KEY  = "CNN_VAR_center" 
LABEL_KEY  = "label"          

def load_all_entries(pkl_dir):
    all_entries = []
    for fn in sorted(os.listdir(pkl_dir)):
        if not fn.endswith(".pkl"): continue
        with open(os.path.join(pkl_dir, fn), "rb") as f:
            chunk = pickle.load(f)
        all_entries.extend(chunk)
    return all_entries

data   = load_all_entries(INPUT_DIR)
labels = [int(np.argmax(d[LABEL_KEY])) for d in data]
print(f"Total entries: {len(data)} | Label counts: {Counter(labels)}")

class ClinVarCNNDataset(Dataset):
    def __init__(self, entries): self.entries = entries
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        e = self.entries[idx]
        x = torch.tensor(np.array(e[EMBED_KEY]).squeeze(0), dtype=torch.float) 
        y = torch.tensor(int(np.argmax(e[LABEL_KEY])), dtype=torch.long)       
        return x, y

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_train_losses, all_val_losses = [], []
all_aucs, all_f1s, all_accs = [], [], []

for fold, (train_idx, val_idx) in enumerate(skf.split(data, labels), start=1):
    print(f"\n===== Fold {fold}/{N_SPLITS} =====")
    val_labels = [labels[i] for i in val_idx]
    print(f"Val label counts: {Counter(val_labels)}")

    train_entries = [data[i] for i in train_idx]
    val_entries   = [data[i] for i in val_idx]

    train_loader = DataLoader(ClinVarCNNDataset(train_entries), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(ClinVarCNNDataset(val_entries), batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    model     = CNN_Baseline_eQTL(input_dim=512, hidden_dim=256, output_dim=2, dropout=0.3).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = CrossEntropyLoss()

    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(SAVE_DIR, f"ClinVar_cnn_mlp_seq_center_fold{fold}.pt")
    best_epoch, best_probs, best_labels = -1, None, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward(); optimizer.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        model.eval()
        running = 0.0
        probs_all, labels_all = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
                logits = model(x)
                loss   = criterion(logits, y); running += loss.item()
                probs  = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs_all.append(probs); labels_all.append(y.detach().cpu().numpy())

        val_loss   = running / max(1, len(val_loader))
        probs_all  = np.concatenate(probs_all) if probs_all else np.array([])
        labels_all = np.concatenate(labels_all) if labels_all else np.array([])

        if len(labels_all) > 0:
            try: auc = roc_auc_score(labels_all, probs_all)
            except ValueError: auc = float("nan")
            precision, recall, thresholds = precision_recall_curve(labels_all, probs_all)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_thresh = thresholds[int(np.nanargmax(f1_scores))] if len(thresholds) > 0 else 0.5
            preds_bin  = (probs_all > best_thresh).astype(int)
            f1 = f1_score(labels_all, preds_bin, zero_division=0)
            acc = accuracy_score(labels_all, preds_bin)
        else:
            auc, f1, acc, best_thresh = float("nan"), float("nan"), float("nan"), 0.5

        print(f"Epoch {epoch:02d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
              f"AUROC {auc:.4f} | F1 {f1:.4f} | Acc {acc:.4f} | thr {best_thresh:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            best_probs  = probs_all.copy()
            best_labels = labels_all.copy()
            best_epoch  = epoch

    if best_probs is None or best_labels is None:
        model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
        model.eval(); _pl, _yl = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE); y = y.to(DEVICE)
                logits = model(x)
                _pl.append(torch.softmax(logits, dim=1)[:,1].cpu().numpy())
                _yl.append(y.cpu().numpy())
        best_probs  = np.concatenate(_pl) if _pl else np.array([])
        best_labels = np.concatenate(_yl) if _yl else np.array([])

    saver.save_fold(fold_id=fold, y_true=best_labels, probs=best_probs,
                    best_val_loss=best_val_loss, best_epoch=best_epoch,
                    extra={"script": "B2_seq_CNN.py"})

    all_train_losses.append(train_loss); all_val_losses.append(val_loss)
    all_aucs.append(auc); all_f1s.append(f1); all_accs.append(acc)
    with open(os.path.join(SAVE_DIR, f"fold_{fold}_metrics.txt"), "w") as f:
        f.write(f"Train Loss: {train_loss:.4f}\n")
        f.write(f"Val Loss: {val_loss:.4f}\n")
        f.write(f"AUROC: {auc:.4f}\n")
        f.write(f"F1: {f1:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Best checkpoint: {best_ckpt_path}\n")
    print(f"✔ Saved fold {fold} best model to {best_ckpt_path}")

avg = {
    "Avg Train Loss": float(np.nanmean(all_train_losses)),
    "Avg Val Loss":   float(np.nanmean(all_val_losses)),
    "Avg AUROC":      float(np.nanmean(all_aucs)),
    "Avg F1":         float(np.nanmean(all_f1s)),
    "Avg Acc":        float(np.nanmean(all_accs)),
}
with open(os.path.join(SAVE_DIR, "summary_metrics.txt"), "w") as f:
    for k, v in avg.items(): f.write(f"{k}: {v:.4f}\n")

print("\n==== Averages ====")
for k, v in avg.items(): print(f"{k}: {v:.4f}")
