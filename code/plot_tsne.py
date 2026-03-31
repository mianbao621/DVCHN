import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import pandas as pd

def count_nodes(dataset_id=1):
    """
    直接读取文件计算 miRNA 和 Drug 的数量
    """
    if dataset_id == 1:
        base_path = 'data/dataset1'
        delimiter = None
    else:
        base_path = 'data/dataset2'
        delimiter = ','
    
    print(f"Counting nodes from {base_path}...")
    
    # 尝试读取 miRNA 文件
    m_path = os.path.join(base_path, 'miRNA_feature.txt')
    if not os.path.exists(m_path):
        m_path = os.path.join(base_path, 'miRNA_feature_fused.txt')
    
    # 尝试读取 Drug 文件
    d_path = os.path.join(base_path, 'SM_feature.txt')
    if not os.path.exists(d_path):
        d_path = os.path.join(base_path, 'SM_feature_fused.txt')
        
    # 读取并计数
    # engine='python' 避免一些分隔符警告
    df_m = pd.read_csv(m_path, sep=delimiter or r'\s+', header=None, engine='python')
    df_d = pd.read_csv(d_path, sep=delimiter or r'\s+', header=None, engine='python')
    
    return len(df_m), len(df_d)

def plot_comparison(dataset_id=1):
    # 1. 获取节点数量 (使用新函数)
    num_miRNAs, num_drugs = count_nodes(dataset_id)
    print(f"Info: {num_miRNAs} miRNAs, {num_drugs} Drugs.")
    
    # 生成标签数组: 0 代表 miRNA, 1 代表 Drug
    labels = np.array([0] * num_miRNAs + [1] * num_drugs)
    
    # 2. 加载特征文件
    print("Loading feature files...")
    # 请确保这两个文件在当前目录下
    file_baseline = "features_cl_0.0.npy"
    file_ours = "features_cl_5.0.npy"
    
    if not os.path.exists(file_baseline) or not os.path.exists(file_ours):
        print(f"Error: 找不到 .npy 文件！\n请检查是否生成了 {file_baseline} 和 {file_ours}")
        return

    feat_baseline = np.load(file_baseline)
    feat_ours = np.load(file_ours)

    # 3. 定义绘图函数
    def run_tsne_and_plot(features, ax, title):
        print(f"Running t-SNE for {title}...")
        # init='random' 通常比 pca 快一点，且对于对比聚类效果足够了
        tsne = TSNE(n_components=2, random_state=2023, init='random', learning_rate='auto')
        z_2d = tsne.fit_transform(features)
        
        # 绘制散点
        # miRNA (Label 0) - 蓝色
        ax.scatter(z_2d[labels==0, 0], z_2d[labels==0, 1], 
                   c='#1f77b4', label='miRNA', s=15, alpha=0.6, edgecolors='none')
        # Drug (Label 1) - 橙色
        ax.scatter(z_2d[labels==1, 0], z_2d[labels==1, 1], 
                   c='#ff7f0e', label='Drug', s=15, alpha=0.6, edgecolors='none')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

    # 4. 画图
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    run_tsne_and_plot(feat_baseline, axes[0], "Baseline (w/o CL)")
    run_tsne_and_plot(feat_ours, axes[1], "Ours (w/ CL)")
    
    plt.tight_layout()
    output_name = "tsne_comparison.png"
    plt.savefig(output_name, dpi=300)
    print(f"Done! 图片已保存为 {output_name}")

if __name__ == "__main__":
    # 如果你是跑 Dataset 2，请把这里改成 dataset_id=2
    plot_comparison(dataset_id=1)