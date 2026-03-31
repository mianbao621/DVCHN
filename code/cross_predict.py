import torch
import numpy as np
import argparse
import os
import pandas as pd
from torch_geometric.utils import negative_sampling, to_undirected

# 复用你现有的模块
from utils import get_data, set_seed
from model import HyperGCN_Model

# === 配置参数 ===
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--cl_rate', type=float, default=5.0) 
parser.add_argument('--tau', type=float, default=0.1)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def run_discovery():
    print("\n" + "="*60)
    print("NEW DRUG DISCOVERY (Unified Feature Space Strategy)")
    print("="*60)
    print("Logic: Load Dataset 1 (Full) -> Train on Known Edges -> Predict Unknown")
    
    # 1. 加载 Dataset 1 (全量数据)
    # 这一步会自动做 PCA/投影，保证所有特征在同一空间！这是解决问题的关键！
    data = get_data(data_ID=1, output_dim=1024)
    x = data.x.to(device)
    
    # Dataset 1 的 edge_index 就是那已知的 664 条关联
    train_pos_edges = data.edge_index.to(device)
    hyperedge_index = data.hyperedge_index.to(device)
    
    print(f"\n[Info] Training on {train_pos_edges.size(1)} known interactions...")
    print(f"[Info] Feature Space is unified (Dim: {x.shape[1]})")
    
    # 构建训练图
    train_graph_adj = to_undirected(train_pos_edges)
    
    # 2. 模型初始化
    input_dim = x.shape[1] 
    model = HyperGCN_Model(in_dim=input_dim, hidden_dim=256, out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
    
    # 3. 训练 (Train)
    # 使用全部已知数据训练，目标是发现新关联
    epochs = 300 
    warm_up = 100
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 负采样
        train_neg_edges = negative_sampling(train_graph_adj, num_nodes=data.num_nodes, num_neg_samples=train_pos_edges.size(1))
        
        pos_scores, neg_scores, proj_s, proj_d = model(
            x, train_graph_adj, hyperedge_index, train_pos_edges, train_neg_edges
        )
        
        loss_task = model.get_task_loss(pos_scores, neg_scores)
        loss_cl = model.get_contrastive_loss(proj_s, proj_d, temperature=args.tau)
        
        if epoch < warm_up:
            loss = args.cl_rate * loss_cl
        else:
            loss = loss_task + args.cl_rate * loss_cl
            
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
            
    print("Training Finished. Starting Inference...")
    
    # 4. 全量推理 (Inference)
    model.eval()
    with torch.no_grad():
        z, _, _ = model.encoder(x, train_graph_adj, hyperedge_index)
        
        # 获取节点数量信息
        m_file = 'data/dataset1/miRNA_feature.txt'
        if not os.path.exists(m_file): m_file = 'data/dataset1/miRNA_feature_fused.txt'
        
        # 统计数量以便切分 z
        with open(m_file, 'r') as f:
            num_m = sum(1 for _ in f)
        num_d = data.num_nodes - num_m
        
        print(f"  Nodes Split: {num_m} miRNAs, {num_d} Drugs")
        
        Z_miRNA = z[:num_m]      
        W = model.decoder.W
        Z_Drug_Transformed = torch.matmul(z[num_m:], W) 
        
        # 计算全量分数矩阵 [Drugs, miRNAs]
        score_matrix = torch.matmul(Z_Drug_Transformed, Z_miRNA.t()).sigmoid()
        
    # 5. 过滤已知关联 (Masking Known Edges)
    print("Filtering known associations...")
    
    known_mask = torch.zeros_like(score_matrix, dtype=torch.bool)
    
    # 遍历已知边，把它们的分数设为 0
    known_edges_T = train_pos_edges.t()
    for edge in known_edges_T:
        u, v = edge[0].item(), edge[1].item()
        
        # 逻辑：Dataset 1 中 miRNA 在前, Drug 在后
        if u < num_m and v >= num_m: 
            m_idx = u
            d_idx = v - num_m
            known_mask[d_idx, m_idx] = True
        elif v < num_m and u >= num_m: 
            m_idx = v
            d_idx = u - num_m
            known_mask[d_idx, m_idx] = True
            
    # 将已知关联清零
    score_matrix[known_mask] = 0.0
    
    # 6. Top-K 预测并保存
    TOP_K = 500
    print(f"\nExtracting TOP {TOP_K} Novel Predictions...")
    
    flat_scores = score_matrix.flatten()
    topk_values, topk_indices = torch.topk(flat_scores, TOP_K)
    
    results = []
    
    print("-" * 50)
    print(f"{'Rank':<5} | {'Drug ID':<10} | {'miRNA ID':<10} | {'Score':<10}")
    print("-" * 50)
    
    for i in range(TOP_K):
        score = topk_values[i].item()
        idx = topk_indices[i].item()
        
        d_idx = idx // num_m
        m_idx = idx % num_m
        
        results.append({
            'Rank': i + 1,
            'Drug_ID': d_idx,
            'miRNA_ID': m_idx,
            'Score': score
        })
        
        if i < 15:
            print(f"{i+1:<5} | Drug_{d_idx:<5} | miRNA_{m_idx:<5} | {score:.4f}")
            
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("novel_prediction_top500.csv", index=False)
    print("-" * 50)
    print("Full list saved to novel_prediction_top500.csv")

if __name__ == "__main__":
    run_discovery()