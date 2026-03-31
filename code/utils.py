import torch
import random
import numpy as np
import os
import pandas as pd
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                             confusion_matrix, accuracy_score, f1_score, matthews_corrcoef)
from sklearn.metrics.pairwise import euclidean_distances

# --- 1. 基础工具 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (1, 1):
        TN, FP, FN, TP = 0, 0, 0, 0
        if y_true[0] == 0: TN = cm[0,0]
        else: TP = cm[0,0]
    else:
        TN, FP, FN, TP = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    F1_score = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc

# --- 2. 辅助函数 ---
def load_txt_robust(path, delimiter=None):
    print(f"  Loading {os.path.basename(path)} with Pandas...")
    try:
        if delimiter:
            df = pd.read_csv(path, sep=delimiter, header=None, engine='python')
        else:
            df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
        df = df.fillna(0.0)
        data = df.values.astype(np.float32)
        print(f"    -> Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"    [Error] Pandas failed to load {path}: {e}")
        return np.loadtxt(path)

def align_rows(matrix, target_rows):
    current_rows = matrix.shape[0]
    if current_rows == target_rows:
        return matrix
    if current_rows < target_rows:
        pad_width = target_rows - current_rows
        return np.pad(matrix, ((0, pad_width), (0, 0)), 'constant')
    return matrix[:target_rows, :]

def calculate_llm_similarity(llm_features):
    sim = cosine_similarity(llm_features)
    return (sim + 1) / 2

# --- 3. 特征投影对齐 (PCA/Random Projection) ---
def align_features(matrix, target_dim, seed=2023):
    scaler = StandardScaler()
    matrix_norm = scaler.fit_transform(matrix)
    current_rows, current_dim = matrix.shape
    
    if current_dim > target_dim:
        # PCA 降维
        pca = PCA(n_components=target_dim, random_state=seed)
        return pca.fit_transform(matrix_norm)
    elif current_dim < target_dim:
        # 随机投影 升维
        np.random.seed(seed)
        projection_matrix = np.random.normal(0, 1/np.sqrt(target_dim), (current_dim, target_dim))
        return np.dot(matrix_norm, projection_matrix)
    else:
        return matrix_norm

def construct_hypergraph_knn(features, k=10):
    sim = cosine_similarity(features) 
    topk_indices = np.argsort(sim, axis=1)[:, -k:] 
    hyperedge_list = []
    for node_idx in range(len(features)):
        neighbors = topk_indices[node_idx]
        for neighbor in neighbors:
            hyperedge_list.append([neighbor, node_idx]) 
    return torch.LongTensor(hyperedge_list).T

# --- 4. Get Data (核心修改) ---
def get_data(data_ID, output_dim):
    
    # === [修改点]：将单路目标维度改为 512 ===
    # 这样最终特征维度 = 512 (原始) + 512 (LLM) = 1024
    TARGET_DIM = 512 
    
    if data_ID == 1:
        base_path = 'data/dataset1'
        delimiter = None 
    elif data_ID == 2:
        base_path = 'data/dataset2'
        delimiter = ',' 
    
    print(f"Loading Dataset {data_ID} from {base_path}...")

    # 加载原始特征
    try:
        miRNA_orig = load_txt_robust(os.path.join(base_path, 'miRNA_feature.txt'), delimiter=delimiter)
        SM_orig = load_txt_robust(os.path.join(base_path, 'SM_feature.txt'), delimiter=delimiter)
    except:
        miRNA_orig = load_txt_robust(os.path.join(base_path, 'miRNA_feature_fused.txt'), delimiter=delimiter)
        SM_orig = load_txt_robust(os.path.join(base_path, 'SM_feature_fused.txt'), delimiter=delimiter)
    
    # 加载 LLM 向量
    llm_delimiter = None
    if data_ID == 1:
        miRNA_llm_vec = load_txt_robust(os.path.join(base_path, 'miRNA_llm_dataset1.txt'), delimiter=llm_delimiter) 
        SM_llm_vec = load_txt_robust(os.path.join(base_path, 'SM_llm_dataset1.txt'), delimiter=llm_delimiter)
    else:
        miRNA_llm_vec = load_txt_robust(os.path.join(base_path, 'miRNA_llm_dataset2.txt'), delimiter=llm_delimiter)
        SM_llm_vec = load_txt_robust(os.path.join(base_path, 'SM_llm_dataset2.txt'), delimiter=llm_delimiter)
        
    association = load_txt_robust(os.path.join(base_path, 'miRNA_SM_adj.txt'), delimiter=',' if data_ID==2 else None)

    # 行数对齐
    target_m_rows = miRNA_orig.shape[0]
    target_s_rows = SM_orig.shape[0]
    miRNA_llm_vec = align_rows(miRNA_llm_vec, target_m_rows)
    SM_llm_vec = align_rows(SM_llm_vec, target_s_rows)
    
    # 计算 LLM 相似度
    miRNA_llm_sim = calculate_llm_similarity(miRNA_llm_vec)
    SM_llm_sim = calculate_llm_similarity(SM_llm_vec)
    
    # === 投影对齐到 512 维 ===
    print(f"  Aligning features to {TARGET_DIM} dim using PCA/Projection...")
    m_v1 = align_features(miRNA_orig, TARGET_DIM)      # 512
    m_v2 = align_features(miRNA_llm_sim, TARGET_DIM)   # 512
    s_v1 = align_features(SM_orig, TARGET_DIM)         # 512
    s_v2 = align_features(SM_llm_sim, TARGET_DIM)      # 512
    
    # === 拼接 ===
    # 最终维度: 512 + 512 = 1024
    m_concat = np.concatenate([m_v1, m_v2], axis=1) 
    s_concat = np.concatenate([s_v1, s_v2], axis=1)

    feature_np = np.concatenate([m_concat, s_concat], axis=0)
    feature = torch.FloatTensor(feature_np)
    
    print("Constructing Hypergraph based on Features...")
    hyperedge_index = construct_hypergraph_knn(feature_np, k=10)
    
    adj = []
    rows, cols = association.shape
    for m in range(rows):
        for s in range(cols):
            if association[m][s] == 1:
                adj.append([m, s + rows])
                
    edge_index = torch.LongTensor(adj).T
    
    data = Data(x=feature, edge_index=edge_index, hyperedge_index=hyperedge_index).cuda()
    
    print(f"Data Loaded. Nodes: {data.num_nodes}, Features Dim: {data.x.shape[1]}")
    return data