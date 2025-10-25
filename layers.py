import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool


class backbone_CNN(nn.Module):
    def __init__(self, nclass, seq_length, embed_length):
        super(backbone_CNN, self).__init__()
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        sequence_length = seq_length
        n_targets = nclass
        linear_size = embed_length

        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 256, kernel_size=conv_kernel_size1),  # Change input channels to 4
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1, stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2, stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))

        reduce_by1 = 2 * (conv_kernel_size1 - 1)
        reduce_by2 = 2 * (conv_kernel_size2 - 1)
        pool_kernel_size1 = float(pool_kernel_size1)
        pool_kernel_size2 = float(pool_kernel_size2)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by1) / pool_kernel_size1)
                 - reduce_by2) / pool_kernel_size2)
            - reduce_by2)

        self.linear = nn.Linear(512 * self._n_channels, linear_size)
        self.batch_norm = nn.BatchNorm1d(linear_size)
        self.classifier = nn.Linear(linear_size, n_targets)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 512 * self._n_channels)
        x_feat = self.linear(reshape_out)
        predict = self.relu(x_feat)
        predict = self.batch_norm(predict)
        predict = self.classifier(predict)
        return predict
    
    def extract_features_flat(self, x):
        # Extract features from the final convolutional layer
        out = self.conv_net(x)
        return out.view(out.size(0), -1) 
    
    def extract_features_og(self, x):
        # Extract features from the final convolutional layer
        out = self.conv_net(x)
        return out
    def extract_dense_embedding(self, x):
        """
        Extracts the most informative and dense 1D embedding from the input DNA sequence.
        This is equivalent to the layer used just before the final classifier.
        """
        with torch.no_grad():
            out = self.conv_net(x)  # (B, 512, L)
            flat = out.view(out.size(0), -1)  # Flatten to (B, 512 * reduced_L)
            x_feat = self.linear(flat)       # Project to (B, embed_length)
            x_feat = self.relu(x_feat)
            x_feat = self.batch_norm(x_feat)
            return x_feat  # Final 1D dense embedding

class GAT_seq_only(nn.Module):
    """
    Combines embeddings from a pre-trained CNN and a GAT model for final classification.
    """
    def __init__(self, seq_length, embed_dim, num_classes, pretrained_cnn):
        super(GAT_seq_only, self).__init__()
        
        # Pre-trained CNN for feature extraction
        self.pretrained_cnn = pretrained_cnn

        # Selectively unfreeze from the first Conv1d with out_channels = 360 onward
        unfreeze_flag = False
        for layer in self.pretrained_cnn.conv_net:
            if isinstance(layer, nn.Conv1d) and layer.out_channels == 360:
                unfreeze_flag = True
            if unfreeze_flag:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze CNN parameters
        # for param in self.pretrained_cnn.parameters():
        #     param.requires_grad = True

        # Define CNN-specific parameters
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4

        # Calculate the number of reduced channels using CNN logic
        reduce_by1 = 2 * (conv_kernel_size1 - 1)
        reduce_by2 = 2 * (conv_kernel_size2 - 1)
        pool_kernel_size1 = float(pool_kernel_size1)
        pool_kernel_size2 = float(pool_kernel_size2)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (seq_length - reduce_by1) / pool_kernel_size1)
                 - reduce_by2) / pool_kernel_size2)
            - reduce_by2
        )

        # Linear transformation for CNN outputs
        self.cnn_linear = nn.Linear(512 * self._n_channels, embed_dim)
        self.cnn_relu = nn.ReLU()
        self.cnn_batch_norm = nn.BatchNorm1d(embed_dim)

        # GAT layers
        self.gat1 = GATConv(embed_dim, embed_dim, heads=4, concat=True)  # Output: embed_dim * 4
        self.gat2 = GATConv(embed_dim * 4, embed_dim, heads=1, concat=False)  # Output: embed_dim

        # Final classifier
        self.classifier = nn.Linear(embed_dim * 2, num_classes)  # Double the input due to concatenation
    
    def forward(self, x, edge_index, batch):
        """
        Forward pass for the enhanced GAT model using pre-trained CNN for the center node.
        """
        # Identify the center nodes (assuming the center node is the first node in each graph)
        unique_graphs = torch.unique(batch)
        center_node_indices = [torch.where(batch == g)[0][0] for g in unique_graphs]

        # Select center node features
        center_node_features = x[center_node_indices, :, :]  # Shape: (num_graphs, num_channels, seq_length)

        # Generate CNN embedding for the center nodes
        # with torch.no_grad():  # Ensure no gradients are computed for the frozen CNN
        cnn_out_center = self.pretrained_cnn.conv_net(center_node_features)  # Shape: (num_graphs, 512, reduced_length)

        # Flatten CNN output and project to embed_dim
        flatten_out_center = cnn_out_center.view(cnn_out_center.size(0), -1)  # Shape: (num_graphs, 512 * reduced_length)
        cnn_embedding = self.cnn_linear(flatten_out_center)
        cnn_embedding = self.cnn_relu(cnn_embedding)
        cnn_embedding = self.cnn_batch_norm(cnn_embedding)  # Shape: (num_graphs, embed_dim)
        # print(f"CNN Embedding Shape: {cnn_embedding.shape}")

        # Process all nodes for GAT embeddings
        # with torch.no_grad():
        cnn_out_graph = self.pretrained_cnn.conv_net(x)  # Shape: (num_nodes, 512, reduced_length)

        # Flatten CNN features for all graph nodes
        flatten_out_graph = cnn_out_graph.view(cnn_out_graph.size(0), -1)  # Shape: (num_nodes, 512 * reduced_length)

        # Project CNN features for GAT processing
        x = self.cnn_linear(flatten_out_graph)
        x = self.cnn_relu(x)
        x = self.cnn_batch_norm(x)

        # GAT layers
        gat_embedding = self.gat1(x, edge_index)  # Shape: (num_nodes, embed_dim * 4)
        gat_embedding = F.elu(gat_embedding)
        gat_embedding = self.gat2(gat_embedding, edge_index)  # Shape: (num_nodes, embed_dim)
        # print(f"Graph Embedding Shape: {gat_embedding.shape}")

        # Perform graph-level pooling for GAT embeddings
        graph_embeddings = global_mean_pool(gat_embedding, batch)  # Shape: (num_graphs, embed_dim)
        # print(f"Graph Embedding Shape: {graph_embeddings.shape}")

        # Concatenate the CNN embedding (center node) with the GAT graph embedding
        combined_embedding = torch.cat((cnn_embedding, graph_embeddings), dim=1)  # Shape: (num_graphs, embed_dim * 2)
        # print(f"Combined Embedding Shape: {combined_embedding.shape}")

        # Final classification
        out = self.classifier(combined_embedding)  # Shape: (num_graphs, num_classes)
        return out
    
    def extract_dense_embedding(self, x, edge_index, batch):
        """
        Returns the combined CNN + GAT embedding before final classification.
        Output shape: (num_graphs, embed_dim * 2)
        """
        with torch.no_grad():
            # print(f"[input] x shape: {x.shape}")
            # Ensure [B, 4, 5000] format before Conv1d
            if x.shape[1] == 5000 and x.shape[2] == 4:
                x = x.permute(0, 2, 1)  # [B, 4, 5000]
                # print(f"[transposed input] x shape: {x.shape}")

            unique_graphs = torch.unique(batch)
            center_node_indices = [torch.where(batch == g)[0][0] for g in unique_graphs]
            center_node_features = x[center_node_indices, :, :]
            # print(f"[center_node_features] shape: {center_node_features.shape}")

            cnn_out_center = self.pretrained_cnn.conv_net(center_node_features)
            flatten_out_center = cnn_out_center.view(cnn_out_center.size(0), -1)
            cnn_embedding = self.cnn_linear(flatten_out_center)
            cnn_embedding = self.cnn_relu(cnn_embedding)
            cnn_embedding = self.cnn_batch_norm(cnn_embedding)

            cnn_out_graph = self.pretrained_cnn.conv_net(x)
            flatten_out_graph = cnn_out_graph.view(cnn_out_graph.size(0), -1)
            x_proj = self.cnn_linear(flatten_out_graph)
            x_proj = self.cnn_relu(x_proj)
            x_proj = self.cnn_batch_norm(x_proj)

            gat_embedding = self.gat1(x_proj, edge_index)
            gat_embedding = F.elu(gat_embedding)
            gat_embedding = self.gat2(gat_embedding, edge_index)

            graph_embeddings = global_mean_pool(gat_embedding, batch)

            combined_embedding = torch.cat((cnn_embedding, graph_embeddings), dim=1)
            return combined_embedding

class backbone_CNN_Dnase(nn.Module):
    def __init__(self, nclass, seq_length, embed_length):
        super(backbone_CNN_Dnase, self).__init__()
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        sequence_length = seq_length
        n_targets = nclass
        linear_size = embed_length
        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1, stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2, stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))
        reduce_by1 = 2 * (conv_kernel_size1 - 1)
        reduce_by2 = 2 * (conv_kernel_size2 - 1)
        pool_kernel_size1 = float(pool_kernel_size1)
        pool_kernel_size2 = float(pool_kernel_size2)
        self._n_channels = int(
            np.floor(
                (np.floor(
                    (sequence_length - reduce_by1) / pool_kernel_size1)
                 - reduce_by2) / pool_kernel_size2)
            - reduce_by2)
        self.linear = nn.Linear(512 * self._n_channels, linear_size)
        self.batch_norm = nn.BatchNorm1d(linear_size)
        self.classifier = nn.Linear(linear_size, n_targets)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.conv_net(x)
        reshape_out = out.view(out.size(0), 512 * self._n_channels)
        x_feat = self.linear(reshape_out)
        predict = self.relu(x_feat)
        predict = self.batch_norm(predict)
        predict = self.classifier(predict)
        return predict
    
    def extract_features_flat(self, x):
        # Extract features from the final convolutional layer
        out = self.conv_net(x)
        return out.view(out.size(0), -1) 
    
    def extract_features_og(self, x):
        # Extract features from the final convolutional layer
        out = self.conv_net(x)
        return out
    def extract_dense_embedding(self, x):
        with torch.no_grad():
            out = self.conv_net(x)                  # (B, 512, L)
            flat = out.view(out.size(0), -1)        # (B, 512 * L)
            x_feat = self.linear(flat)              # (B, 512)
            x_feat = self.relu(x_feat)
            x_feat = self.batch_norm(x_feat)
            return x_feat

class GAT_DNase(nn.Module):
    """
    GAT model using a partially trainable pre-trained CNN for feature extraction, 
    concatenating center node embedding with GAT output for classification.
    """
    def __init__(self, embed_dim, num_classes, pretrained_cnn, sequence_length=5000):
        super(GAT_DNase, self).__init__()

        self.pretrained_cnn = pretrained_cnn

        # === Unfreeze from Conv1d(360, ...) onwards ===
        unfreeze_flag = False
        for layer in self.pretrained_cnn.conv_net:
            if isinstance(layer, nn.Conv1d) and layer.out_channels == 360:
                unfreeze_flag = True
            if unfreeze_flag:
                for param in layer.parameters():
                    param.requires_grad = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

        # Calculate reduced length after CNN
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        reduce_by1 = 2 * (conv_kernel_size1 - 1)
        reduce_by2 = 2 * (conv_kernel_size2 - 1)
        self._n_channels = int(
            (((sequence_length - reduce_by1) / pool_kernel_size1 - reduce_by2) / pool_kernel_size2) - reduce_by2
        )

        # Linear projection
        self.linear = nn.Linear(512 * self._n_channels, embed_dim)
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        # GAT layers
        self.gat1 = GATConv(embed_dim, embed_dim, heads=4, concat=True)
        self.gat2 = GATConv(embed_dim * 4, embed_dim, heads=1, concat=False)

        # Final classifier (CNN center + GAT-pooled)
        self.classifier = nn.Linear(embed_dim * 2, num_classes)

    def forward(self, x, edge_index, batch):
        # --- Center node CNN embedding ---
        unique_graphs = torch.unique(batch)
        center_node_indices = [torch.where(batch == g)[0][0] for g in unique_graphs]
        center_node_features = x[center_node_indices, :, :]  # (num_graphs, C, L)

        cnn_out_center = self.pretrained_cnn.conv_net(center_node_features)  # (num_graphs, 512, reduced_L)
        flatten_out_center = cnn_out_center.view(cnn_out_center.size(0), -1)
        cnn_embedding = self.linear(flatten_out_center)
        cnn_embedding = self.relu(cnn_embedding)
        cnn_embedding = self.batch_norm(cnn_embedding)

        # --- GAT branch ---
        cnn_out_graph = self.pretrained_cnn.conv_net(x)
        flatten_out_graph = cnn_out_graph.view(cnn_out_graph.size(0), -1)
        x_proj = self.linear(flatten_out_graph)
        x_proj = self.relu(x_proj)
        x_proj = self.batch_norm(x_proj)

        x_gat = self.gat1(x_proj, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = self.dropout(x_gat)
        x_gat = self.gat2(x_gat, edge_index)

        # Global mean pooling
        graph_embeddings = global_mean_pool(x_gat, batch)

        # Concatenate GAT and CNN(center) embeddings
        combined = torch.cat([cnn_embedding, graph_embeddings], dim=1)

        return self.classifier(combined)
    
    def get_embedding(self, x, edge_index, batch):
        # Center node embedding (CNN)
        unique_graphs = torch.unique(batch)
        center_node_indices = [torch.where(batch == g)[0][0] for g in unique_graphs]
        center_node_features = x[center_node_indices, :, :]  # (num_graphs, C, L)

        cnn_out_center = self.pretrained_cnn.conv_net(center_node_features)
        flatten_out_center = cnn_out_center.view(cnn_out_center.size(0), -1)
        cnn_embedding = self.linear(flatten_out_center)
        cnn_embedding = self.relu(cnn_embedding)
        cnn_embedding = self.batch_norm(cnn_embedding)

        # Full graph embedding (GAT)
        cnn_out_graph = self.pretrained_cnn.conv_net(x)
        flatten_out_graph = cnn_out_graph.view(cnn_out_graph.size(0), -1)
        x_proj = self.linear(flatten_out_graph)
        x_proj = self.relu(x_proj)
        x_proj = self.batch_norm(x_proj)

        x_gat = self.gat1(x_proj, edge_index)
        x_gat = F.elu(x_gat)
        x_gat = self.dropout(x_gat)
        x_gat = self.gat2(x_gat, edge_index)

        graph_embeddings = global_mean_pool(x_gat, batch)

        # Combine CNN and GAT embeddings
        combined_embedding = torch.cat([cnn_embedding, graph_embeddings], dim=1)

        return combined_embedding
    
    def extract_center_edge_contribs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        use_source_norm: bool = True,
        eps: float = 1e-9,
    ):
        """
        Return contributions for edges whose DEST is the center node in each graph.
        Contribution per edge e = alpha_e (* ||x_src||_1 if use_source_norm), then
        normalized so contributions into each center sum to 1.
        """
        self.eval()

        # ----- Project all nodes (same path as forward) -----
        cnn_out = self.pretrained_cnn.conv_net(x)                       # [N, 512, L’]
        x_proj  = self.linear(cnn_out.view(cnn_out.size(0), -1))
        x_proj  = self.relu(x_proj)
        x_proj  = self.batch_norm(x_proj)

        # ----- GAT1 -> GAT2 (get attention from last layer) -----
        x1 = self.gat1(x_proj, edge_index)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)                                           # input to gat2

        x2, (ei, alpha2) = self.gat2(x1, edge_index, return_attention_weights=True)
        src, dst = ei[0], ei[1]

        # ---- Make alpha a flat [E] vector (handle [E,1] or [E,H]) ----
        if alpha2.dim() == 2:
            if alpha2.size(-1) == 1:
                alpha2 = alpha2.squeeze(-1)                             # [E]
            else:
                alpha2 = alpha2.mean(dim=-1)                            # [E] (avg heads)
        # ensure 1-D
        alpha2 = alpha2.contiguous().view(-1)

        # ----- Find center node per graph (first node in each batch group) -----
        g_ids = torch.unique(batch, sorted=True)
        centers = torch.stack([torch.where(batch == g)[0][0] for g in g_ids])  # [G]

        # map graph id -> center node id
        center_map = torch.empty(int(batch.max()) + 1, dtype=torch.long, device=batch.device)
        center_map[g_ids] = centers

        # For each edge, what is the center node id of its graph?
        center_dst_for_edge = center_map[batch[dst]]                    # [E]
        is_center_edge = (dst == center_dst_for_edge)
        sel = torch.where(is_center_edge)[0]
        if sel.numel() == 0:
            return None

        # ----- Raw contribution per selected edge -----
        contrib = alpha2.index_select(0, sel)                           # [E_sel]
        if use_source_norm:
            src_strength = x1.index_select(0, src.index_select(0, sel)).abs().sum(dim=1)  # [E_sel]
            contrib = contrib * src_strength

        # ----- Normalize per center node (sum to 1) -----
        dst_sel = dst.index_select(0, sel)                              # [E_sel]
        sum_per_dst = torch.zeros(x.size(0), device=x.device, dtype=contrib.dtype)
        sum_per_dst.index_add_(0, dst_sel, contrib)                     # accum by center id
        contrib_norm = contrib / (sum_per_dst.index_select(0, dst_sel) + eps)

        return {
            "edge_index_center_edges": ei.index_select(1, sel).cpu(),   # [2, E_sel]
            "contrib_raw": contrib.cpu(),                               # [E_sel]
            "contrib_norm": contrib_norm.cpu(),                         # [E_sel]
            "batch_of_dst": batch.index_select(0, dst_sel).cpu(),       # [E_sel]
        }



class Baseline_EVO2(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=128, output_dim=751, dropout=0.5):
        super(Baseline_EVO2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, output_dim)  # Final classifier layer

    def forward(self, embeddings):
        # Forward pass through the network
        x = self.relu(self.fc1(embeddings))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.classifier(x)  # Apply the classifier directly
        return x
    def extract_dense_embedding(self, embeddings):
        """
        Extracts the dense representation before classification.
        Input: embeddings (B, 4096)
        Output: dense embedding (B, hidden_dim)
        """
        with torch.no_grad():
            x = self.relu(self.fc1(embeddings))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            return x  # Shape: (B, hidden_dim)
    
class GAT_EVO2(nn.Module):
    def __init__(self, trained_mlp, hidden_dim=128, num_classes=751, heads=4, dropout=0.5, freeze_mlp=True, unfreeze_fc2=True):
        super(GAT_EVO2, self).__init__()

        # Save the original layer references
        self.fc1 = trained_mlp.fc1
        self.fc2 = trained_mlp.fc2
        self.dropout = trained_mlp.dropout
        self.relu = trained_mlp.relu

        # Compose MLP feature extractor manually
        self.mlp_feature_extractor = nn.Sequential(
            self.fc1,
            self.relu,
            self.dropout,
            self.fc2,
            self.relu,
            self.dropout
        )

        # Freeze all layers by default
        if freeze_mlp:
            for param in self.mlp_feature_extractor.parameters():
                param.requires_grad = False

        # Optionally unfreeze only fc2
        if unfreeze_fc2:
            for param in self.fc2.parameters():
                param.requires_grad = True

        # GAT layers
        embed_dim = self.fc2.out_features
        self.gat1 = GATConv(embed_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout)

        # Pooling and classification
        self.global_pool = global_mean_pool
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.mlp_feature_extractor(x)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        x = self.global_pool(x, batch)
        x = self.classifier(x)
        return x
    
    def extract_dense_embedding(self, x, edge_index, batch):
        """
        Extracts a richer graph-level embedding for representation analysis.
        Output shape: (batch_size, hidden_dim * heads) = (batch_size, 512)
        """
        with torch.no_grad():
            x = self.mlp_feature_extractor(x)            # (num_nodes, 128)
            x = self.gat1(x, edge_index)                 # (num_nodes, 512)
            x = F.elu(x)
            pooled = self.global_pool(x, batch)          # (batch_size, 512)
            return pooled

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
    
class GAT_ClinVar(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=128, output_dim=2, dropout=0.5):
        super().__init__()
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

class CNN_Baseline_ClinVar(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=2, dropout=0.3):
        super(CNN_Baseline_ClinVar, self).__init__()
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
    
class EVO_GAT_ClinVar(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=2, dropout=0.3):
        super(EVO_GAT_ClinVar, self).__init__()
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

class EVO_Baseline_ClinVar(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=2, dropout=0.3):
        super(EVO_Baseline_ClinVar, self).__init__()
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

