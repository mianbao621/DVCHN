import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, HypergraphConv
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import calculate_metrics

# --- 1. 对比学习投影头 (Projection Head) ---
class ProjectionHead(nn.Module):
    """
    将特征映射到对比学习的潜在空间
    结构: Linear -> ELU -> Linear
    """
    def __init__(self, in_dim, hidden_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# --- 2. 双线性解码器 (Bilinear Decoder) ---
class BilinearDecoder(nn.Module):
    """
    解码器: Score = Z_u * W * Z_v
    """
    def __init__(self, feature_dim):
        super(BilinearDecoder, self).__init__()
        # W: [128, 128] 的参数矩阵，用于模拟非对称的生化反应规则
        self.W = nn.Parameter(torch.Tensor(feature_dim, feature_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, edge_index):
        # z: [Num_Nodes, Feature_Dim]
        # edge_index: [2, Num_Edges] (u, v)
        
        inputs = z[edge_index[0]]  # Zu (Drug/miRNA)
        outputs = z[edge_index[1]] # Zv (miRNA/Drug)
        
        # 1. 线性变换: Zu * W
        # [Batch, Dim] * [Dim, Dim] -> [Batch, Dim]
        weighted_inputs = torch.matmul(inputs, self.W)
        
        # 2. 交互点积: (Zu * W) dot Zv
        # sum([Batch, Dim] * [Batch, Dim], dim=1) -> [Batch]
        scores = torch.sum(weighted_inputs * outputs, dim=1)
        
        return scores

# --- 3. 混合编码器 (Hybrid Encoder) ---
class HybridEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(HybridEncoder, self).__init__()
        
        # --- 分支 A: Static GCN (拓扑视图) ---
        # Layer 1: 2048 -> 256
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        # Layer 2: 256 -> 128
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        
        # --- 分支 B: Dynamic HyperGCN (语义视图) ---
        # Layer 1: 2048 -> 256
        self.hyper1 = HypergraphConv(in_channels, hidden_channels)
        # Layer 2: 256 -> 128
        self.hyper2 = HypergraphConv(hidden_channels, out_channels)
        
        # --- 模块组件 ---
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.5)
        
        # --- 自适应融合门控 ---
        # 可学习参数，初始化为 0 (Sigmoid后为 0.5)
        self.fusion_gate = nn.Parameter(torch.FloatTensor([0.0]))

    def forward(self, x, edge_index, hyperedge_index):
        # --- Path 1: GCN ---
        x_s = self.gcn1(x, edge_index)
        x_s = self.activation(x_s)
        x_s = self.dropout(x_s) # 训练时开启 Dropout
        x_s = self.gcn2(x_s, edge_index) # Output: [N, 128]
        
        # --- Path 2: HyperGCN ---
        x_d = self.hyper1(x, hyperedge_index)
        x_d = self.activation(x_d)
        x_d = self.dropout(x_d)
        x_d = self.hyper2(x_d, hyperedge_index) # Output: [N, 128]
        
        # --- Fusion ---
        # 计算融合权重 alpha
        alpha = torch.sigmoid(self.fusion_gate)
        
        # 加权融合: Z = alpha * GCN + (1 - alpha) * Hyper
        z = alpha * x_s + (1 - alpha) * x_d
        
        # 返回: 融合特征 z (用于解码)，以及分视图特征 (用于对比学习)
        return z, x_s, x_d

# --- 4. 整体模型 (Model) ---
class HyperGCN_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(HyperGCN_Model, self).__init__()
        
        self.encoder = HybridEncoder(in_dim, hidden_dim, out_dim)
        self.decoder = BilinearDecoder(out_dim)
        
        # 对比学习投影头
        self.proj_head = ProjectionHead(out_dim, out_dim)
        
    def forward(self, x, edge_index, hyperedge_index, pos_edges, neg_edges):
        # 1. 编码: 获取融合特征 z 和 分视图特征 z_s, z_d
        z, z_s, z_d = self.encoder(x, edge_index, hyperedge_index)
        
        # 2. 解码: 计算正负样本分数 (Task)
        pos_scores = self.decoder(z, pos_edges)
        neg_scores = self.decoder(z, neg_edges)
        
        # 3. 投影: 为对比学习准备特征
        proj_s = self.proj_head(z_s)
        proj_d = self.proj_head(z_d)
        
        return pos_scores, neg_scores, proj_s, proj_d

    def get_task_loss(self, pos_scores, neg_scores):
        """计算分类任务损失 (BCE)"""
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_labels, neg_labels])
        
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss

    def get_contrastive_loss(self, z1, z2, temperature=0.5):
        """
        计算对比学习损失 (InfoNCE)
        目标: 拉近同一节点的 z1(GCN) 和 z2(Hyper) 表示
        """
        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算相似度矩阵 [N, N]
        sim_matrix = torch.matmul(z1, z2.T) / temperature
        
        # 标签: 对角线是正样本 (0->0, 1->1...)
        batch_size = z1.size(0)
        labels = torch.arange(batch_size).to(z1.device)
        
        # Cross Entropy 会自动处理 Softmax 和 Log
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    @torch.no_grad()
    def test(self, x, edge_index, hyperedge_index, pos_edges, neg_edges):
        """测试模式: 只以前向传播计算指标，不计算梯度"""
        # 测试时只需要 z
        z, _, _ = self.encoder(x, edge_index, hyperedge_index)
        
        pos_scores = self.decoder(z, pos_edges).sigmoid()
        neg_scores = self.decoder(z, neg_edges).sigmoid()
        
        pred = torch.cat([pos_scores, neg_scores], dim=0)
        
        y_pos = torch.ones(pos_scores.size(0))
        y_neg = torch.zeros(neg_scores.size(0))
        y = torch.cat([y_pos, y_neg], dim=0)
        
        y, pred = y.cpu().numpy(), pred.cpu().numpy()
        
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        
        # 计算其他指标
        temp = pred.copy()
        temp[temp >= 0.5] = 1
        temp[temp < 0.5] = 0
        acc, sen, pre, spe, F1, mcc = calculate_metrics(y, temp)
        
        return auc, ap, acc, sen, pre, spe, F1, mcc