import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_undirected
from utils import get_data, set_seed
from model import HyperGCN_Model

# === 1. 配置参数 ===
CONFIG = {
    "dataset": 1,
    "p_list": [1, 2, 5, 10],   # 负样本比例
    "seeds": [2023, 2024, 2025], 
    "epochs": 1000,
    "dim": 1024,
    "hidden": 256,
    "out_dim": 128,
    "lr": 0.001,
    "out_csv": "ablation_results.csv",
    "plot_dir": "plots"        # 图片保存文件夹
}

# === 2. 辅助函数 ===
def get_neg_samples(adj_matrix, num_samples, seed, exclude_pairs=set()):
    """采样负样本，严格排除已知正样本"""
    rng = np.random.default_rng(seed)
    rows, cols = adj_matrix.shape
    neg_edges = []
    while len(neg_edges) < num_samples:
        r, c = rng.integers(0, rows), rng.integers(0, cols)
        if adj_matrix[r, c] == 0 and (r, c) not in exclude_pairs:
            neg_edges.append([r, c + rows]) 
            exclude_pairs.add((r, c))
    return torch.tensor(neg_edges, dtype=torch.long).t()

def evaluate_metrics(model, z, pos_edges, neg_edges):
    """同时计算 AUC 和 AUPR"""
    pos_scores = model.decoder(z, pos_edges).sigmoid().cpu().numpy()
    neg_scores = model.decoder(z, neg_edges).sigmoid().cpu().numpy()
    
    y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    y_scores = np.concatenate([pos_scores, neg_scores])
    
    auc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    return auc, aupr

def train_and_eval(data, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg):
    """训练并返回最佳模型在测试集上的指标"""
    device = data.x.device # 获取当前GPU设备
    model = HyperGCN_Model(data.x.size(1), CONFIG["hidden"], CONFIG["out_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=5e-5)
    
    # 【修复重点】：先将 tensor 移动到 GPU，再生成无向图
    train_pos = train_pos.to(device)
    train_neg = train_neg.to(device)
    val_pos = val_pos.to(device)
    val_neg = val_neg.to(device)
    test_pos = test_pos.to(device)
    test_neg = test_neg.to(device)

    # 此时 train_pos 已经在 GPU 上了，生成的 train_graph 也会在 GPU 上
    train_graph = to_undirected(train_pos)
    
    best_val_auc = 0
    final_test_auc, final_test_aupr = 0, 0

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        opt.zero_grad()
        
        # 现在所有输入都在同一个 device 上了，不会报错
        pos_out, neg_out, z_s, z_d = model(data.x, train_graph, data.hyperedge_index, train_pos, train_neg)
        
        # Loss: Warmup阶段只用对比损失，之后联合训练
        loss_task = model.get_task_loss(pos_out, neg_out)
        loss_cl = model.get_contrastive_loss(z_s, z_d)
        loss = 0.1 * loss_cl if epoch <= 100 else loss_task + 0.1 * loss_cl
            
        loss.backward()
        opt.step()

        # 验证与测试 (每10轮)
        if epoch > 100 and epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                z, _, _ = model.encoder(data.x, train_graph, data.hyperedge_index)
                val_auc, _ = evaluate_metrics(model, z, val_pos, val_neg)
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    # 记录此时的测试集指标
                    final_test_auc, final_test_aupr = evaluate_metrics(model, z, test_pos, test_neg)
    
    return final_test_auc, final_test_aupr

def plot_results(df):
    """绘制带误差棒的折线图"""
    os.makedirs(CONFIG["plot_dir"], exist_ok=True)
    metrics = [("auc", "Test AUC"), ("aupr", "Test AUPR")]
    
    for col, ylabel in metrics:
        summary = df.groupby("p")[col].agg(["mean", "std"]).reset_index()
        
        plt.figure(figsize=(6, 4), dpi=150)
        plt.errorbar(
            summary["p"], summary["mean"], yerr=summary["std"], 
            fmt='-o', capsize=5, linewidth=2, markersize=6, color='#1f77b4'
        )
        plt.xlabel("Negative/Positive ratio p")
        plt.ylabel(ylabel)
        plt.xticks(summary["p"])
        plt.grid(True, linestyle='--', alpha=0.5)
        
        filename = f"fig_p_vs_{col}.png"
        save_path = os.path.join(CONFIG["plot_dir"], filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {save_path}")

# === 3. 主程序 ===
def main():
    # 加载数据
    A = np.loadtxt(f"data/dataset{CONFIG['dataset']}/miRNA_SM_adj.txt").astype(int)
    data = get_data(CONFIG['dataset'], CONFIG['dim'])
    
    # 确保转为CPU上的numpy数组
    all_pos = data.edge_index.t().cpu().numpy()
    
    results = []
    print(f"=== 开始消融实验 (p={CONFIG['p_list']}) ===")

    for seed in CONFIG["seeds"]:
        set_seed(seed)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(all_pos)):
            # 数据划分
            train_val_edges = all_pos[train_idx]
            test_edges = torch.tensor(all_pos[test_idx].T, dtype=torch.long)
            
            tr_idx, va_idx = train_test_split(np.arange(len(train_val_edges)), test_size=0.1, random_state=seed)
            train_edges = torch.tensor(train_val_edges[tr_idx].T, dtype=torch.long)
            val_edges = torch.tensor(train_val_edges[va_idx].T, dtype=torch.long)

            # 固定 Val/Test 负样本 (1:1)
            banned = set((r, c) for r, c in np.argwhere(A == 1))
            val_neg = get_neg_samples(A, val_edges.size(1), seed+1, banned.copy())
            test_neg = get_neg_samples(A, test_edges.size(1), seed+2, banned.copy())

            for p in CONFIG["p_list"]:
                # 固定 Train 负样本 (1:p)
                n_neg = int(train_edges.size(1) * p)
                train_neg = get_neg_samples(A, n_neg, seed+p*100, banned.copy())
                
                # 训练
                auc, aupr = train_and_eval(data, train_edges, train_neg, val_edges, val_neg, test_edges, test_neg)
                
                print(f"Seed={seed} Fold={fold} p={p} | AUC={auc:.4f} AUPR={aupr:.4f}")
                results.append({"p": p, "seed": seed, "auc": auc, "aupr": aupr})

    # 保存数据
    df = pd.DataFrame(results)
    df.to_csv(CONFIG["out_csv"], index=False)
    print(f"\n结果已保存至: {CONFIG['out_csv']}")
    
    # 打印统计摘要
    print("\n=== 最终统计 (Mean ± Std) ===")
    print(df.groupby("p")[["auc", "aupr"]].agg(["mean", "std"]))
    
    # 自动画图
    plot_results(df)

if __name__ == "__main__":
    main()