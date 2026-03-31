import torch
import numpy as np
import itertools
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.utils import negative_sampling, to_undirected
import sys
import time

# 复用你现有的模块
from utils import get_data, set_seed
from model import HyperGCN_Model

# ==========================================
# 1. 定义全量搜索空间 (Full Search Space)
# ==========================================
SEARCH_SPACE = {
    # --- A. 模型结构参数 (决定模型的容量) ---
    'hidden_dim': [128, 256, 512],   # 隐藏层大小
    'out_dim':    [64, 128],         # 输出层大小
    
    # --- B. 优化参数 (决定学习速度) ---
    'lr': [0.001, 0.0005],           # 学习率
    
    # --- C. 对比学习参数 (决定辅助任务权重) ---
    'cl_rate': [1.0, 5.0],           # CL 权重
    'tau':     [0.1]                 # 温度系数 (通常0.1就很稳，为了省时间先只搜一个)
}

# === 固定配置 ===
DATASET_ID = 2
# TARGET_DIM = 1024 (由 utils.py 内部决定)
EPOCHS = 150        # 搜索时跑少一点 (正式训练用 300/500)
WARM_UP = 50        # 预热期缩短
PATIENCE = 20       # 早停耐心值
N_SPLITS = 5        # 5折交叉验证
SEED = 2023
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_single_experiment(data, params):
    """
    运行一组特定参数的 5-Fold CV
    params: 字典，包含 hidden_dim, out_dim, lr, cl_rate, tau
    """
    set_seed(SEED)
    
    # 解包参数
    h_dim = params['hidden_dim']
    o_dim = params['out_dim']
    lr = params['lr']
    cl_rate = params['cl_rate']
    tau = params['tau']
    
    # 简单的逻辑检查：隐藏层通常 >= 输出层
    if h_dim < o_dim:
        return -1.0 # 跳过不合理的组合
    
    # 准备数据
    x = data.x.to(DEVICE)
    edge_index = data.edge_index
    hyperedge_index = data.hyperedge_index.to(DEVICE)
    num_nodes = data.num_nodes
    input_dim = x.shape[1]
    
    all_pos_edges = edge_index.t()
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    
    fold_val_aucs = []
    
    # 开始 5-Fold
    for fold, (train_val_idx, _) in enumerate(kf.split(all_pos_edges)):
        # 划分 Train / Val (不需要 Test)
        train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=SEED)
        
        train_pos_edges = all_pos_edges[train_idx].t().to(DEVICE)
        val_pos_edges = all_pos_edges[val_idx].t().to(DEVICE)
        
        train_graph_adj = to_undirected(train_pos_edges)
        val_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=val_pos_edges.size(1))
        
        # === 初始化模型 (使用当前搜索的维度) ===
        model = HyperGCN_Model(
            in_dim=input_dim, 
            hidden_dim=h_dim, 
            out_dim=o_dim
        ).to(DEVICE)
        
        # === 初始化优化器 (使用当前搜索的 LR) ===
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        
        # 训练循环
        best_fold_auc = 0.0
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            
            train_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=train_pos_edges.size(1))
            
            pos_scores, neg_scores, proj_s, proj_d = model(
                x, train_graph_adj, hyperedge_index, train_pos_edges, train_neg_edges
            )
            
            loss_task = model.get_task_loss(pos_scores, neg_scores)
            
            # === 使用当前搜索的 Tau ===
            loss_cl = model.get_contrastive_loss(proj_s, proj_d, temperature=tau)
            
            # === 使用当前搜索的 CL_Rate ===
            if epoch < WARM_UP:
                loss = cl_rate * loss_cl
            else:
                loss = loss_task + cl_rate * loss_cl
                
            loss.backward()
            optimizer.step()
            
            # 验证
            if epoch >= WARM_UP:
                model.eval()
                with torch.no_grad():
                    val_auc, _, _, _, _, _, _, _ = model.test(
                        x, train_graph_adj, hyperedge_index, val_pos_edges, val_neg_edges
                    )
                
                if val_auc > best_fold_auc:
                    best_fold_auc = val_auc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        break
        
        fold_val_aucs.append(best_fold_auc)
        
    return np.mean(fold_val_aucs)

def main():
    print("Loading Data...")
    data = get_data(DATASET_ID, 1024)
    
    # 生成所有参数组合
    keys = SEARCH_SPACE.keys()
    values = SEARCH_SPACE.values()
    combinations = list(itertools.product(*values))
    
    print(f"\n>>> FULL GRID SEARCH START")
    print(f"Total combinations: {len(combinations)}")
    print("-" * 100)
    # 打印表头
    header = " | ".join([f"{k:<8}" for k in keys]) + " | Mean Val AUC"
    print(header)
    print("-" * 100)
    
    best_score = 0.0
    best_params = None
    
    start_time = time.time()
    
    for i, combo in enumerate(combinations):
        # 组合成字典
        params = dict(zip(keys, combo))
        
        try:
            # 运行实验
            mean_auc = run_single_experiment(data, params)
            
            if mean_auc == -1.0:
                # 这是一个被跳过的组合 (Hidden < Out)
                continue
                
            # 打印结果行
            res_str = " | ".join([f"{v:<8}" for v in combo]) + f" | {mean_auc:.4f}"
            print(res_str)
            
            # 更新最佳
            if mean_auc > best_score:
                best_score = mean_auc
                best_params = params
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Skipping combo {params} due to OOM")
                torch.cuda.empty_cache()
            else:
                print(f"Error with {params}: {e}")

    duration = (time.time() - start_time) / 60
    
    print("-" * 100)
    print(f"Grid Search Finished in {duration:.1f} mins.")
    print(f"BEST CONFIGURATION found:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"  Best Val AUC: {best_score:.4f}")
    print("-" * 100)

if __name__ == "__main__":
    main()