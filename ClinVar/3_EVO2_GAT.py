#!/usr/bin/env python3
import os, sys, glob, pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve

from layers import EVO_GAT_ClinVar  
from cv_saver import CVFoldSaver

EMB_DIR = "/mnt/nfs/bowei/CHROME/data/ClinVar/GAT_EVO2/"
OUT_DIR = "/mnt/nfs/bowei/CHROME/model/ClinVar/GAT_EVO_embedding/"
os.makedirs(OUT_DIR, exist_ok=True)
CV_OUT_DIR = "/mnt/nfs/bowei/CHROME/model/ClinVar/"
CV_TAG     = "GAT_EVO_embedding"
saver      = CVFoldSaver(CV_OUT_DIR, CV_TAG)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_FOLDS    = 5
EPOCHS       = 50
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 8
MIN_DELTA    = 1e-4
SEED         = 42

FEATURE_KEY = "GAT_EVO_embedding"  
IN_DIM      = 512

def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed()

def load_entries(emb_dir):
    files = sorted(glob.glob(os.path.join(emb_dir, "ClinVar_chunk_*_GAT_embeddings*.pkl")))
    assert files, f"No embedding files found in {emb_dir}"
    entries = []
    for fp in files:
        with open(fp, "rb") as f:
            chunk = pickle.load(f)
        for e in chunk:
            if "label_idx" in e and FEATURE_KEY in e and e[FEATURE_KEY] is not None:
                x = np.asarray(e[FEATURE_KEY], dtype=np.float32)
                if x.ndim == 1 and x.shape[0] == IN_DIM:
                    entries.append(e)
    return entries

class EmbDataset(Dataset):
    def __init__(self, X, y): self.X = X.astype(np.float32); self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i): return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]))

def class_weights(y, num_classes=2):
    y = np.asarray(y); N = y.shape[0]
    cnt = np.bincount(y, minlength=num_classes).astype(np.float32)
    w = np.where(cnt > 0, N / (num_classes * cnt), 0.0)
    return torch.tensor(w, dtype=torch.float32)

def metrics_binary(y_true, logits):
    probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
    y_true = np.asarray(y_true)
    p, r, thr = precision_recall_curve(y_true, probs)
    f1s = 2*(p*r)/(p+r+1e-8)
    best_idx = int(np.argmax(f1s))
    best_t = float(thr[max(0, min(best_idx, len(thr)-1))]) if len(thr) else 0.5
    y_pred = (probs > best_t).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    try: auc = roc_auc_score(y_true, probs)
    except: auc = float("nan")
    return {"acc": acc, "f1": f1, "auroc": auc, "best_thresh": best_t, "probs": probs}

def train_one_fold(fold, X, y, outdir):
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, va) in enumerate(skf.split(np.arange(X.shape[0]), y)):
        if k == fold: train_idx, val_idx = tr, va; break

    mu, sigma = X[train_idx].mean(axis=0), X[train_idx].std(axis=0); sigma[sigma==0] = 1.0
    X_tr = (X[train_idx] - mu) / sigma
    X_va = (X[val_idx]   - mu) / sigma

    train_ds = EmbDataset(X_tr, y[train_idx])
    val_ds   = EmbDataset(X_va, y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    model = EVO_GAT_ClinVar(input_dim=IN_DIM, output_dim=2).to(DEVICE)
    w = class_weights(y[train_idx]).to(DEVICE)
    crit = CrossEntropyLoss(weight=w)
    opt  = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val, best_state, best_epoch, no_imp = 1e9, None, -1, 0
    best_probs, best_labels = None, None

    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        model.eval(); va_loss = 0.0; logits_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = crit(logits, yb)
                va_loss += float(loss.item())
                logits_all.append(logits.cpu())
                y_all.append(yb.cpu().numpy())
        va_loss /= max(1, len(val_loader))
        logits_cat = torch.cat(logits_all, dim=0) if logits_all else torch.empty((0,2))
        y_cat      = np.concatenate(y_all) if y_all else np.empty((0,), dtype=int)
        m = metrics_binary(y_cat, logits_cat)

        print(f"[GAT 512] Fold {fold+1} Ep {ep:02d} | Train {tr_loss:.4f} | Val {va_loss:.4f} | "
              f"AUROC {m['auroc']:.4f} | F1 {m['f1']:.4f} | ACC {m['acc']:.4f}")

        if va_loss + MIN_DELTA < best_val:
            best_val, best_state, best_epoch, no_imp = va_loss, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, ep, 0
            best_probs, best_labels = m["probs"].copy(), y_cat.copy()
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"[GAT 512] ⏹️ Early stop at epoch {ep}")
                break

    if best_state is not None: model.load_state_dict(best_state)

    ckpt = os.path.join(outdir, f"GAT_EVO_embedding_fold_{fold+1}.pt")
    torch.save(model.state_dict(), ckpt)
    print(f"[GAT 512] ✅ Saved → {ckpt}")

    if best_probs is None or best_labels is None:
        model.eval(); logits_all, y_all = [], []
        with torch.no_grad():
            for xb, yb in DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits_all.append(model(xb).cpu())
                y_all.append(yb.cpu().numpy())
        logits_cat = torch.cat(logits_all, dim=0) if logits_all else torch.empty((0,2))
        y_cat      = np.concatenate(y_all) if y_all else np.empty((0,), dtype=int)
        best_probs = torch.softmax(logits_cat, dim=1)[:,1].numpy()
        best_labels= y_cat

    saver.save_fold(fold_id=fold, y_true=best_labels, probs=best_probs,
                    best_val_loss=best_val, best_epoch=best_epoch,
                    extra={"script": "3_EVO2_GAT.py"})

    return {"auroc": roc_auc_score(best_labels, best_probs) if len(best_labels) else float("nan")}

def main():
    entries = load_entries(EMB_DIR)
    y_all = np.array([int(e["label_idx"]) for e in entries], dtype=int)
    X_all = np.stack([np.asarray(e[FEATURE_KEY], dtype=np.float32) for e in entries], axis=0)

    print(f"Loaded entries: {len(entries)}  Dim={X_all.shape[1]}  Label dist: {dict(Counter(y_all))}")

    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []
    for fold, _ in enumerate(skf.split(np.arange(X_all.shape[0]), y_all)):
        m = train_one_fold(fold, X_all, y_all, OUT_DIR)
        fold_metrics.append(m)

    aurocs = [m["auroc"] for m in fold_metrics if m["auroc"] == m["auroc"]]
    with open(os.path.join(OUT_DIR, "GAT_EVO_embedding_metrics_average.txt"), "w") as f:
        f.write(f"Avg AUROC: {np.mean(aurocs):.4f}\n")
    print("\n[GAT 512] 📈 Avg AUROC:", f"{np.mean(aurocs):.4f}" if aurocs else "nan")

if __name__ == "__main__":
    main()
