#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fig.4-style ablation: AUC vs. reduced training positives
- Loops: removed_ratio(%) x seed x fold
- Saves: auc_results.csv (per run), auc_summary.csv (mean±std), fig4_auc.png/pdf

This script is written to match your current code structure:
- uses utils.get_data / utils.set_seed
- uses model.HyperGCN_Model
- uses KFold + train_test_split + negative_sampling + to_undirected
"""

import os
import argparse
import copy
import math
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch_geometric.utils import negative_sampling, to_undirected

from utils import get_data, set_seed
from model import HyperGCN_Model


def parse_int_list(s: str):
    """Parse '0,10,20' -> [0,10,20]"""
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def parse_seed_list(s: str):
    """Parse '2023,2024' -> [2023,2024]"""
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def subsample_train_edges(edge_index_e2: torch.Tensor, keep_ratio: float, seed: int) -> torch.Tensor:
    """
    edge_index_e2: [E, 2] (each row is a positive edge)
    returns: [E_keep, 2]
    """
    assert edge_index_e2.dim() == 2 and edge_index_e2.size(1) == 2
    E = edge_index_e2.size(0)
    keep = max(1, int(round(E * keep_ratio)))
    g = torch.Generator(device=edge_index_e2.device)
    g.manual_seed(seed)
    perm = torch.randperm(E, generator=g, device=edge_index_e2.device)
    idx = perm[:keep]
    return edge_index_e2[idx]


def build_neg_edges(train_graph_edge_index_2e: torch.Tensor, num_nodes: int, num_neg: int) -> torch.Tensor:
    """
    train_graph_edge_index_2e: [2, E] (undirected training graph)
    returns: [2, num_neg]
    """
    neg = negative_sampling(
        edge_index=train_graph_edge_index_2e,
        num_nodes=num_nodes,
        num_neg_samples=num_neg
    )
    return neg


def train_one_setting(
    data,
    all_pos_edges_e2: torch.Tensor,
    fold_id: int,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    keep_ratio: float,
    seed: int,
    args
):
    """
    Train on one fold with a given keep_ratio.
    Returns test_auc, test_ap
    """
    device = data.x.device
    num_nodes = data.num_nodes
    input_dim = data.x.size(1)

    # Split train/val from trainval edges (keep val/test fixed across ratios)
    trainval_edges = all_pos_edges_e2[trainval_idx]
    test_pos_edges = all_pos_edges_e2[test_idx]

    tr_idx, va_idx = train_test_split(
        np.arange(trainval_edges.size(0)),
        test_size=args.val_ratio,
        random_state=seed + fold_id,
        shuffle=True
    )
    train_pos_edges_full = trainval_edges[torch.tensor(tr_idx, device=device)]
    val_pos_edges = trainval_edges[torch.tensor(va_idx, device=device)]

    # Subsample ONLY training positives
    subsample_seed = seed * 100000 + fold_id * 1000 + int(round((1 - keep_ratio) * 100))
    train_pos_edges = subsample_train_edges(train_pos_edges_full, keep_ratio=keep_ratio, seed=subsample_seed)

    # Build training graph adjacency (no leakage)
    train_graph_adj = to_undirected(train_pos_edges.t().contiguous()).contiguous()  # [2, E_train_undirected]

    # Sample negatives based on training graph
    train_neg_edges = build_neg_edges(train_graph_adj, num_nodes, num_neg=train_pos_edges.size(0)).to(device)
    val_neg_edges = build_neg_edges(train_graph_adj, num_nodes, num_neg=val_pos_edges.size(0)).to(device)
    test_neg_edges = build_neg_edges(train_graph_adj, num_nodes, num_neg=test_pos_edges.size(0)).to(device)

    # Prepare edge shapes for decoder: [2, E]
    train_pos_2e = train_pos_edges.t().contiguous()
    val_pos_2e = val_pos_edges.t().contiguous()
    test_pos_2e = test_pos_edges.t().contiguous()

    # Init model
    model = HyperGCN_Model(in_dim=input_dim, hidden_dim=args.hidden_dim, out_dim=args.out_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        pos_scores, neg_scores, proj_s, proj_d = model(
            data.x, train_graph_adj, data.hyperedge_index, train_pos_2e, train_neg_edges
        )
        task_loss = model.get_task_loss(pos_scores, neg_scores)
        cl_loss = model.get_contrastive_loss(proj_s, proj_d, temperature=args.tau)

        if epoch <= args.warmup_epochs:
            loss = args.cl_rate * cl_loss
        else:
            loss = task_loss + args.cl_rate * cl_loss

        loss.backward()
        optimizer.step()

        # Validate (we can skip validation during warmup to match your original idea)
        if epoch <= args.warmup_epochs and not args.eval_during_warmup:
            continue

        model.eval()
        with torch.no_grad():
            val_auc, val_ap, *_ = model.test(
                data.x, train_graph_adj, data.hyperedge_index, val_pos_2e, val_neg_edges
            )

        # Early stopping starts after warmup (recommended)
        if epoch <= args.warmup_epochs:
            continue

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    # Test
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_auc, test_ap, *_ = model.test(
            data.x, train_graph_adj, data.hyperedge_index, test_pos_2e, test_neg_edges
        )

    return float(test_auc), float(test_ap)


def plot_fig4(summary_csv: str, out_png: str, out_pdf: str):
    """
    Create a Fig.4-like grouped bar chart from summary csv.
    summary csv columns: removed_pct, model, auc_mean, auc_std
    """
    df = pd.read_csv(summary_csv)
    ratios = sorted(df["removed_pct"].unique())
    models = list(df["model"].unique())

    x = np.arange(len(ratios))
    n = len(models)
    group_width = 0.80
    bar_w = group_width / max(1, n)
    offsets = (np.arange(n) - (n - 1) / 2) * bar_w

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)

    for i, m in enumerate(models):
        sub = df[df["model"] == m].set_index("removed_pct").loc[ratios]
        y = sub["auc_mean"].to_numpy()
        yerr = sub["auc_std"].to_numpy() if "auc_std" in sub.columns else None
        ax.bar(x + offsets[i], y, width=bar_w, label=m)
        # If you want error bars, uncomment:
        # if yerr is not None:
        #     ax.errorbar(x + offsets[i], y, yerr=yerr, fmt='none', capsize=2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r}%" for r in ratios])
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.legend(loc="upper right", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png)
    fig.savefig(out_pdf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1, help="Dataset ID (1 or 2)")
    parser.add_argument("--dim", type=int, default=1024, help="Feature Dimension (should match get_data output)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--out_dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Val split ratio from trainval edges")

    # Contrastive learning
    parser.add_argument("--cl_rate", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--warmup_epochs", type=int, default=100)
    parser.add_argument("--eval_during_warmup", action="store_true",
                        help="If set, evaluate val even during warmup. Default: off (closer to your main.py idea).")

    # Ablation settings
    parser.add_argument("--removed_pcts", type=str, default="0,10,20,30,40",
                        help="Comma-separated training reduction percentages. E.g. '0,10,20,30,40'")
    parser.add_argument("--seeds", type=str, default="2023,2024,2025,2026,2027",
                        help="Comma-separated random seeds. E.g. '2023,2024,2025,2026,2027'")

    parser.add_argument("--model_name", type=str, default="MTJL", help="Name shown in Fig.4 legend")
    parser.add_argument("--out_dir", type=str, default="outputs_ratio_auc", help="Output directory")

    args = parser.parse_args()

    removed_pcts = parse_int_list(args.removed_pcts)
    seeds = parse_seed_list(args.seeds)
    if len(removed_pcts) == 0:
        raise ValueError("removed_pcts is empty.")
    if len(seeds) == 0:
        raise ValueError("seeds is empty.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data once (same as your main.py)
    set_seed(seeds[0])
    data = get_data(args.dataset, args.dim)
    device = data.x.device

    all_pos_edges_e2 = data.edge_index.t().contiguous()  # [E,2]
    num_edges = all_pos_edges_e2.size(0)

    rows = []

    for seed in seeds:
        # Fix fold split per seed (so ratios are comparable within the seed)
        set_seed(seed)
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

        for removed_pct in removed_pcts:
            keep_ratio = max(0.0001, 1.0 - removed_pct / 100.0)

            for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(num_edges)), start=1):
                # Make training deterministic per (seed, removed_pct, fold)
                set_seed(seed + removed_pct * 13 + fold_id * 131)

                test_auc, test_ap = train_one_setting(
                    data=data,
                    all_pos_edges_e2=all_pos_edges_e2,
                    fold_id=fold_id,
                    trainval_idx=trainval_idx,
                    test_idx=test_idx,
                    keep_ratio=keep_ratio,
                    seed=seed,
                    args=args
                )

                rows.append({
                    "dataset": args.dataset,
                    "model": args.model_name,
                    "removed_pct": removed_pct,
                    "keep_ratio": keep_ratio,
                    "seed": seed,
                    "fold": fold_id,
                    "auc": test_auc,
                    "aupr": test_ap,
                })

                print(f"[Done] seed={seed} removed={removed_pct}% fold={fold_id}  AUC={test_auc:.4f}  AUPR={test_ap:.4f}")

    # Save per-run results
    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(args.out_dir, "auc_results.csv")
    results_df.to_csv(results_csv, index=False)

    # Aggregate (mean±std across seed×fold)
    summary = (
        results_df
        .groupby(["model", "removed_pct"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            aupr_mean=("aupr", "mean"),
            aupr_std=("aupr", "std"),
        )
        .sort_values(["removed_pct", "model"])
    )
    summary_csv = os.path.join(args.out_dir, "auc_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Plot Fig.4-like chart
    out_png = os.path.join(args.out_dir, "fig4_auc.png")
    out_pdf = os.path.join(args.out_dir, "fig4_auc.pdf")
    plot_fig4(summary_csv, out_png, out_pdf)

    print("\nSaved:")
    print(f"  - {results_csv}")
    print(f"  - {summary_csv}")
    print(f"  - {out_png}")
    print(f"  - {out_pdf}")


if __name__ == "__main__":
    main()
