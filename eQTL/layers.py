import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class GAT_CNN_eQTL(nn.Module):
    def __init__(self, input_dim=1536, hidden_dim=128, output_dim=2, dropout=0.5):
        super(GAT_CNN_eQTL, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CNN_Baseline_eQTL(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=256, output_dim=2, dropout=0.3):
        super(CNN_Baseline_eQTL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class EVO2_GAT_eQTL(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=256, output_dim=2, dropout=0.3):
        super(EVO2_GAT_eQTL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
    
    
class EVO2_MLP_baseline_eQTL(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, output_dim=2, dropout=0.3):
        super(EVO2_MLP_baseline_eQTL, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
