import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.utils import negative_sampling, to_undirected
import copy
import argparse
import os

# 引入你现有的模块
from utils import get_data, set_seed
from model import HyperGCN_Model

# === 配置参数 ===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=1, help="Dataset ID")
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def train_model_with_ratio(data_pack, ratio, cl_rate, fold_idx=0):
    """
    运行单次训练的函数
    ratio: 保留训练集的比例 (0.2 - 1.0)
    cl_rate: 对比学习权重 (0.0 或 5.0)
    """
    set_seed(args.seed)
    
    # === 直接从 Data 对象获取数据 ===
    x = data_pack.x.to(device)
    edge_index = data_pack.edge_index
    hyperedge_index = data_pack.hyperedge_index.to(device)
    num_nodes = data_pack.num_nodes
    
    all_pos_edges = edge_index.t()
    
    # 只跑 Fold 1
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    train_val_idx, test_idx = list(kf.split(all_pos_edges))[fold_idx]
    
    # 1. 划分 Train/Val/Test
    # test_pos_edges = all_pos_edges[test_idx].t().to(device) # 训练中用不到test边，注释掉省显存
    train_idx_full, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=args.seed)
    
    # === [核心逻辑]：对训练集进行随机采样 (稀疏化) ===
    num_train_full = len(train_idx_full)
    if ratio < 1.0:
        # 随机选择索引
        keep_num = int(num_train_full * ratio)
        np.random.seed(args.seed)
        keep_indices = np.random.choice(train_idx_full, keep_num, replace=False)
        train_pos_edges = all_pos_edges[keep_indices].t().to(device)
    else:
        train_pos_edges = all_pos_edges[train_idx_full].t().to(device)
    # ===============================================
    
    val_pos_edges = all_pos_edges[val_idx].t().to(device)
    
    # 2. 构建图结构 (仅基于稀疏后的训练边)
    train_graph_adj = to_undirected(train_pos_edges)
    
    # 3. 负采样 (Val set)
    # Val 负样本固定
    val_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=val_pos_edges.size(1))
    
    # 4. 模型初始化
    input_dim = x.shape[1]
    model = HyperGCN_Model(in_dim=input_dim, hidden_dim=256, out_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=5e-5)
    
    # 5. 训练循环
    best_val_auc = 0.0
    best_test_auc = 0.0 # 这里我们用 Val AUC 最高的那个模型在 Val 上的表现近似代替 Test，或者你需要传入 Test Set
    # 为了实验简单，我们这里直接返回 Best Val AUC 作为性能指标，
    # 或者为了严谨，我们应该在这里评估 test set。我们加上 test set 吧。
    
    test_pos_edges = all_pos_edges[test_idx].t().to(device)
    test_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=test_pos_edges.size(1))

    patience = 20
    counter = 0
    
    # 预热参数
    WARM_UP = 100
    epochs = 300 
    
    print(f"  [Ratio {ratio:.1f} | CL {cl_rate}] Training... (Edges: {train_pos_edges.size(1)})")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 训练集动态负采样
        train_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=train_pos_edges.size(1))
        
        # Forward
        pos_scores, neg_scores, proj_s, proj_d = model(
            x, train_graph_adj, hyperedge_index, train_pos_edges, train_neg_edges
        )
        
        loss_task = model.get_task_loss(pos_scores, neg_scores)
        loss_cl = model.get_contrastive_loss(proj_s, proj_d, temperature=0.1)
        
        if epoch < WARM_UP:
            loss = cl_rate * loss_cl
        else:
            loss = loss_task + cl_rate * loss_cl
            
        loss.backward()
        optimizer.step()
        
        # Val & Test Monitoring
        if epoch >= WARM_UP:
            model.eval()
            with torch.no_grad():
                val_auc, _, _, _, _, _, _, _ = model.test(
                    x, train_graph_adj, hyperedge_index, val_pos_edges, val_neg_edges
                )
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                # 记录此时的 Test 结果
                with torch.no_grad():
                    curr_test_auc, _, _, _, _, _, _, _ = model.test(
                        x, train_graph_adj, hyperedge_index, test_pos_edges, test_neg_edges
                    )
                best_test_auc = curr_test_auc
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break
    
    print(f"    -> Done. Ratio {ratio} | CL {cl_rate} | Test AUC: {best_test_auc:.4f}")
    return best_test_auc

def run_experiment():
    # 1. 加载数据
    print("Loading Data...")
    # output_dim 参数在 utils 里实际上被 TARGET_DIM 覆盖了，填 1024 即可
    data = get_data(args.dataset, 1024) 
    
    ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # 存储结果
    res_baseline = [] # w/o CL
    res_ours = []     # w/ CL
    
    # 2. 循环跑实验
    for r in ratios:
        print(f"\n=== Running Ratio: {r*100}% ===")
        
        # Run Baseline (CL=0)
        # print("  Running Baseline (w/o CL)...")
        auc_base = train_model_with_ratio(data, ratio=r, cl_rate=0.0)
        res_baseline.append(auc_base)
        
        # Run Ours (CL=5.0)
        # print("  Running Ours (w/ CL)...")
        auc_ours = train_model_with_ratio(data, ratio=r, cl_rate=5.0)
        res_ours.append(auc_ours)
        
    # 3. 打印结果
    print("\n" + "="*30)
    print("Final Results (AUC)")
    print("="*30)
    print(f"Ratios:   {ratios}")
    print(f"Baseline: {res_baseline}")
    print(f"Ours:     {res_ours}")
    
    # 4. 画图
    plt.figure(figsize=(8, 6))
    plt.plot(ratios, res_baseline, marker='o', linestyle='--', color='gray', label='Baseline (w/o CL)')
    plt.plot(ratios, res_ours, marker='s', linestyle='-', color='red', linewidth=2, label='Ours (w/ CL)')
    
    plt.xlabel('Training Data Ratio', fontsize=12)
    plt.ylabel('Test AUC', fontsize=12)
    plt.title('Robustness Analysis on Data Sparsity', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 自动调整Y轴范围
    if len(res_baseline) > 0:
        min_auc = min(min(res_baseline), min(res_ours))
        plt.ylim(min_auc - 0.05, 1.01)
    
    plt.savefig('sparsity_analysis.png', dpi=300)
    print("图表已保存: sparsity_analysis.png")

if __name__ == "__main__":
    run_experiment()