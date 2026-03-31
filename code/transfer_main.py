import pandas as pd
import glob
import os

def generate_consensus_report(file_pattern='*.csv', output_name='consensus_top_discovery.xlsx'):
    # 1. 获取所有文件路径
    files = glob.glob(file_pattern)
    # 排除掉之前可能生成的汇总表
    files = [f for f in files if 'consensus' not in f.lower()]
    
    print(f">>> Found {len(files)} prediction files: {files}")
    
    all_dfs = []
    
    # 2. 读取并标准化数据
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        # 记录每一对关联在当前文件里的排名
        df['rank_in_run'] = df.index + 1 
        df['run_id'] = i + 1
        all_dfs.append(df)
    
    # 合并所有数据
    combined = pd.concat(all_dfs)
    
    # 3. 按 Drug_ID 和 miRNA_ID 进行分组统计
    consensus = combined.groupby(['Drug_ID', 'miRNA_ID']).agg(
        Occurrence=('Score', 'count'),           # 出现频次
        Avg_Score=('Score', 'mean'),             # 平均得分
        Avg_Rank=('rank_in_run', 'mean'),        # 平均排名
        Std_Rank=('rank_in_run', 'std')          # 排名波动情况（越小越稳）
    ).reset_index()
    
    # 4. 核心排序逻辑：优先看出现频次(降序)，再看平均排名(升序)
    consensus = consensus.sort_values(by=['Occurrence', 'Avg_Rank'], ascending=[False, True])
    
    # 5. 过滤掉只出现过 1 次的偶然结果（可选）
    stable_consensus = consensus[consensus['Occurrence'] >= 2]
    
    # 6. 保存为 Excel
    print(f">>> Saving {len(stable_consensus)} consensus pairs to {output_name}...")
    stable_consensus.to_excel(output_name, index=False)
    
    # 打印前 10 名共识
    print("\n" + "="*50)
    print("TOP 10 CONSENSUS PREDICTIONS")
    print("="*50)
    print(stable_consensus.head(10)[['Drug_ID', 'miRNA_ID', 'Occurrence', 'Avg_Score', 'Avg_Rank']])
    print("="*50)

if __name__ == "__main__":
    generate_consensus_report()