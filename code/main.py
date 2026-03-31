import torch
import argparse
import numpy as np
import copy
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.utils import negative_sampling, to_undirected
from utils import get_data, set_seed
from model import HyperGCN_Model

# --- 参数设置 ---
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=1, help="Choose Datasets (1 or 2)")
parser.add_argument('--seed', type=int, default=2028, help="Base Random seed.")
parser.add_argument('--dim', type=int, default=1024, help='Feature Dimension')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--epochs', type=int, default=1000, help='Max Training epochs')
parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
parser.add_argument('--n_splits', type=int, default=5, help='K-Fold Cross Validation')

# --- 对比学习参数 ---
parser.add_argument('--cl_rate', type=float, default=0.1, help='Weight for Contrastive Loss (Lambda)')
parser.add_argument('--tau', type=float, default=0.1, help='Temperature for Contrastive Loss')

# --- [新增] 负样本比例参数 ---
parser.add_argument('--neg_ratio', type=int, default=1, help='Negative Sampling Ratio (e.g., 5 means 1:5)')

args = parser.parse_args()

# 全局指标存储
fold_metrics = {
    'AUC': [], 'AP': [], 'ACC': [], 'SEN': [], 
    'PRE': [], 'SPE': [], 'F1': [], 'MCC': []
}

set_seed(args.seed)

print(f"Loading Data for {args.n_splits}-Fold Cross Validation...")
data = get_data(args.dataset, args.dim)
num_nodes = data.num_nodes
input_dim = data.x.shape[1]
all_pos_edges = data.edge_index.t() 

kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

print(f"Model: HyperGCN + GCN + Attention Fusion + Contrastive Learning")
print(f"Params: CL_Rate={args.cl_rate}, Tau={args.tau}, Neg_Ratio={args.neg_ratio}")
print("-" * 60)

for fold, (train_val_idx, test_idx) in enumerate(kf.split(all_pos_edges)):
    print(f"\n>>> Fold {fold + 1}/{args.n_splits}")
    
    # 1. 数据划分
    test_pos_edges = all_pos_edges[test_idx].t()
    train_idx_final, val_idx = train_test_split(train_val_idx, test_size=0.1, random_state=args.seed)
    train_pos_edges = all_pos_edges[train_idx_final].t()
    val_pos_edges = all_pos_edges[val_idx].t()
    
    # 2. 严格隔离图结构
    train_graph_adj = to_undirected(train_pos_edges)
    
    # 验证集和测试集依然保持 1:1 的负样本 (用于公平评估)
    test_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=test_pos_edges.size(1))
    val_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=val_pos_edges.size(1))
    
    # 3. 初始化模型 (修复了这里缺失的代码)
    # 这里的 hidden_dim=256, out_dim=128 与您之前消融实验的设置保持一致
    model = HyperGCN_Model(input_dim, 128, 128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-5)
    
    # 4. 训练
    best_val_auc = 0.0
    best_model_state = None
    patience_counter = 0
    
    WARM_UP_EPOCHS = 100 

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        # === [修改点] 动态采样: 使用 args.neg_ratio 控制比例 ===
        num_neg = train_pos_edges.size(1) * args.neg_ratio
        train_neg_edges = negative_sampling(train_graph_adj, num_nodes=num_nodes, num_neg_samples=num_neg)
        
        # Forward Pass
        pos_scores, neg_scores, proj_s, proj_d = model(
            data.x, 
            train_graph_adj,        
            data.hyperedge_index,   
            train_pos_edges,        
            train_neg_edges         
        )
        
        # 计算损失
        loss_task = model.get_task_loss(pos_scores, neg_scores)
        loss_cl = model.get_contrastive_loss(proj_s, proj_d, temperature=args.tau)
        
        # 预热逻辑
        if epoch < WARM_UP_EPOCHS:
            loss = args.cl_rate * loss_cl
        else:
            loss = loss_task + args.cl_rate * loss_cl
        
        loss.backward()
        optimizer.step()
        
        # 验证集评估
        model.eval()
        val_auc, val_ap, _, _, _, _, _, _ = model.test(
            data.x, train_graph_adj, data.hyperedge_index, val_pos_edges, val_neg_edges
        )
        
        # 早停逻辑 (预热期不早停)
        if epoch >= WARM_UP_EPOCHS:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= args.patience:
                print(f"    [Early Stopping] Epoch {epoch+1}. Best Val AUC: {best_val_auc:.4f}")
                break
        else:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(model.state_dict())
        
        if (epoch + 1) % 50 == 0:
            status = "WarmUp" if epoch < WARM_UP_EPOCHS else "Train"
            print(f"Epoch {epoch+1} [{status}] | Task: {loss_task.item():.4f} | CL: {loss_cl.item():.4f} | Val AUC: {val_auc:.4f}")
            
    # Testing
    print("    Restoring best model for Testing...")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    
    # 保存特征 (可选)
    # with torch.no_grad():
    #     z, _, _ = model.encoder(data.x, train_graph_adj, data.hyperedge_index)
    #     np.save(f"features_fold{fold}.npy", z.cpu().numpy())

    test_auc, test_ap, acc, sen, pre, spe, F1, mcc = model.test(
        data.x, 
        train_graph_adj, 
        data.hyperedge_index, 
        test_pos_edges.cuda(), 
        test_neg_edges
    )

    print(f"    Fold {fold + 1} Result -> AUC: {test_auc:.4f}, AP: {test_ap:.4f}, F1: {F1:.4f}")

    fold_metrics['AUC'].append(test_auc)
    fold_metrics['AP'].append(test_ap)
    fold_metrics['ACC'].append(acc)
    fold_metrics['SEN'].append(sen)
    fold_metrics['PRE'].append(pre)
    fold_metrics['SPE'].append(spe)
    fold_metrics['F1'].append(F1)
    fold_metrics['MCC'].append(mcc)
    
    del model, optimizer, train_graph_adj
    torch.cuda.empty_cache()

# 最终结果
print("-" * 60)
print(f"Final 5-Fold CV Results (Mean ± Std):")
print("-" * 60)
for key, values in fold_metrics.items():
    print(f"{key:<5}: {np.mean(values):.4f} ± {np.std(values):.4f}")
print("-" * 60)