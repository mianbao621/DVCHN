#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Table-2 style independent set evaluation WITHOUT leakage (Method 1).

You only need:
- association matrix: data/dataset{1,2}/miRNA_SM_adj.txt (0/1)
- similarity matrices / features are loaded by your existing utils.get_data()

What it does:
1) Randomly pick N_pos positives (A==1) as an independent set (never used in training graph).
2) Randomly pick N_neg negatives (A==0) as an independent set.
3) Train your model on the remaining positives (with optional val split) using your main.py-like loop.
4) Score the 2N pairs (N_pos+N_neg), rank them:
   - "#Top positives": how many positives appear in the top N_pos scores
   - "#Bottom negatives": how many negatives appear in the bottom N_neg scores
5) Repeat over multiple seeds and report mean±std.

Run example:
python independent_table2.py --dataset 1 --dim 1024 --epochs 1000 --cl_rate 0.1 --warmup_epochs 100 \
  --n_pos 45 --n_neg 45 --seeds 2023,2024,2025,2026,2027 --out_dir outputs_independent
"""

import os
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import get_data, set_seed
from model import HyperGCN_Model


def parse_int_list(s: str):
    if s is None or str(s).strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def load_association(dataset: int):
    base = f"data/dataset{dataset}"
    path = os.path.join(base, "miRNA_SM_adj.txt")
    # dataset1 uses whitespace, dataset2 uses comma (per your utils.py)
    delimiter = None if dataset == 1 else ","
    A = np.loadtxt(path, delimiter=delimiter)
    A = A.astype(int)
    return A, path


def sample_independent_pairs(A: np.ndarray, n_pos: int, n_neg: int, seed: int):
    """
    Sample independent positives from A==1 and independent negatives from A==0 (bipartite).
    Returns:
      ind_pos_pairs: list[(m, s)]
      ind_neg_pairs: list[(m, s)]
    """
    rng = np.random.default_rng(seed)

    pos_coords = np.argwhere(A == 1)
    if pos_coords.shape[0] < n_pos:
        raise ValueError(f"Not enough positive pairs in matrix: have {pos_coords.shape[0]}, need {n_pos}")
    rng.shuffle(pos_coords)
    ind_pos = [tuple(x) for x in pos_coords[:n_pos]]

    # Negatives: sample without replacement from A==0
    num_m, num_s = A.shape
    ind_neg = []
    used = set(ind_pos)
    used_neg = set()
    # n_neg is small (e.g., 45), simple rejection sampling is fine
    while len(ind_neg) < n_neg:
        m = int(rng.integers(0, num_m))
        s = int(rng.integers(0, num_s))
        if A[m, s] != 0:
            continue
        if (m, s) in used or (m, s) in used_neg:
            continue
        used_neg.add((m, s))
        ind_neg.append((m, s))

    return ind_pos, ind_neg


def pairs_to_edge_index(pairs, num_m):
    """
    Convert (m, s) into graph edge indices (miRNA index = m, SM index = s+num_m).
    Returns torch.LongTensor [2, E]
    """
    rows = [p[0] for p in pairs]
    cols = [p[1] + num_m for p in pairs]
    e = torch.tensor([rows, cols], dtype=torch.long)
    return e


def filter_training_positives(all_pos_edges_e2: torch.Tensor, ind_pos_edges_2e: torch.Tensor):
    """
    Remove independent positives from all positives to avoid leakage.
    all_pos_edges_e2: [E,2]
    ind_pos_edges_2e: [2, N]
    Returns remaining_pos_edges_e2: [E_remain,2]
    """
    # Build a set of tuples to remove
    ind = set(zip(ind_pos_edges_2e[0].tolist(), ind_pos_edges_2e[1].tolist()))
    keep = []
    for a, b in all_pos_edges_e2.tolist():
        if (a, b) not in ind:
            keep.append([a, b])
    if len(keep) == 0:
        raise ValueError("After removing independent positives, no training positives remain.")
    return torch.tensor(keep, dtype=torch.long, device=all_pos_edges_e2.device)


def sample_bipartite_negatives(A: np.ndarray, num_m: int, num_s: int, num_samples: int, seed: int, banned_pairs=None):
    """
    Sample negatives strictly from A==0 (bipartite), avoiding banned_pairs.
    Returns list[(m,s)].
    """
    rng = np.random.default_rng(seed)
    banned_pairs = banned_pairs or set()
    out = []
    used = set()
    while len(out) < num_samples:
        m = int(rng.integers(0, num_m))
        s = int(rng.integers(0, num_s))
        if A[m, s] != 0:
            continue
        if (m, s) in banned_pairs or (m, s) in used:
            continue
        used.add((m, s))
        out.append((m, s))
    return out


@torch.no_grad()
def score_edges(model, data, graph_adj_2e, edges_2e):
    model.eval()
    z, _, _ = model.encoder(data.x, graph_adj_2e, data.hyperedge_index)
    scores = model.decoder(z, edges_2e)  # sigmoid scores
    return scores.view(-1).detach().cpu().numpy()


def train_model(data, train_graph_adj_2e, train_pos_2e, train_neg_2e, val_pos_2e, val_neg_2e, args):
    device = data.x.device
    model = HyperGCN_Model(in_dim=data.x.size(1), hidden_dim=args.hidden_dim, out_dim=args.out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_auc = -1.0
    best_state = None
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        pos_scores, neg_scores, proj_s, proj_d = model(
            data.x, train_graph_adj_2e, data.hyperedge_index, train_pos_2e, train_neg_2e
        )
        task_loss = model.get_task_loss(pos_scores, neg_scores)
        cl_loss = model.get_contrastive_loss(proj_s, proj_d, temperature=args.tau)

        if epoch <= args.warmup_epochs:
            loss = args.cl_rate * cl_loss
        else:
            loss = task_loss + args.cl_rate * cl_loss

        loss.backward()
        opt.step()

        # Validation (start after warmup)
        if epoch <= args.warmup_epochs:
            continue

        model.eval()
        with torch.no_grad():
            val_auc, val_ap, *_ = model.test(
                data.x, train_graph_adj_2e, data.hyperedge_index, val_pos_2e, val_neg_2e
            )

        if val_auc > best_val_auc:
            best_val_auc = float(val_auc)
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_auc


def evaluate_independent_table2(model, data, train_graph_adj_2e, ind_pos_2e, ind_neg_2e):
    pos_scores = score_edges(model, data, train_graph_adj_2e, ind_pos_2e)
    neg_scores = score_edges(model, data, train_graph_adj_2e, ind_neg_2e)

    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    y_pred = np.concatenate([pos_scores, neg_scores])

    # Ranking counts (Table-2 style)
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    order_desc = np.argsort(-y_pred)  # high -> low
    top_idx = order_desc[:n_pos]
    top_pos_count = int(y_true[top_idx].sum())

    order_asc = np.argsort(y_pred)  # low -> high
    bottom_idx = order_asc[:n_neg]
    bottom_neg_count = int((1 - y_true[bottom_idx]).sum())

    auc = float(roc_auc_score(y_true, y_pred))
    ap = float(average_precision_score(y_true, y_pred))
    return top_pos_count, bottom_neg_count, auc, ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=int, default=1, choices=[1,2])
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--out_dim", type=int, default=128)

    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--weight_decay", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50)

    ap.add_argument("--cl_rate", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--warmup_epochs", type=int, default=100)

    ap.add_argument("--n_pos", type=int, default=45)
    ap.add_argument("--n_neg", type=int, default=45)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seeds", type=str, default="2023,2024,2025,2026,2027")
    ap.add_argument("--model_name", type=str, default="Ours")
    ap.add_argument("--out_dir", type=str, default="outputs_independent")
    args = ap.parse_args()

    seeds = parse_int_list(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load association to sample independent set (NO leakage into training graph)
    A, assoc_path = load_association(args.dataset)
    num_m, num_s = A.shape

    # Load graph data (features, hypergraph, full edge_index) using your existing pipeline
    set_seed(seeds[0])
    data = get_data(args.dataset, args.dim)
    device = data.x.device

    all_pos_edges_e2 = data.edge_index.t().contiguous()  # [E,2] (miRNA -> SM+num_m)
    rows = []

    for seed in seeds:
        set_seed(seed)
        # Sample independent positives/negatives from the *matrix* (bipartite)
        ind_pos_pairs, ind_neg_pairs = sample_independent_pairs(A, args.n_pos, args.n_neg, seed=seed)

        ind_pos_2e = pairs_to_edge_index(ind_pos_pairs, num_m).to(device)
        ind_neg_2e = pairs_to_edge_index(ind_neg_pairs, num_m).to(device)

        # Remove independent positives from training positives
        remain_pos_e2 = filter_training_positives(all_pos_edges_e2, ind_pos_2e)  # [E_remain,2]

        # Split train/val positives from remaining positives
        idx = np.arange(remain_pos_e2.size(0))
        tr_idx, va_idx = train_test_split(idx, test_size=args.val_ratio, random_state=seed, shuffle=True)
        train_pos_e2 = remain_pos_e2[torch.tensor(tr_idx, device=device)]
        val_pos_e2 = remain_pos_e2[torch.tensor(va_idx, device=device)]

        # Build training graph adjacency from TRAIN positives only (prevents leakage)
        train_graph_adj_2e = torch.stack(
            [train_pos_e2[:,0], train_pos_e2[:,1]], dim=0
        ).contiguous()  # [2,E_train]
        # Undirected message passing
        train_graph_adj_2e = torch.cat([train_graph_adj_2e, train_graph_adj_2e.flip(0)], dim=1).contiguous()

        # Negatives sampled strictly from A==0 (also avoids accidentally sampling positives as negatives)
        # Note: using A here does NOT leak training info into model; it only ensures negatives are truly 0 in the matrix.
        banned = set(ind_pos_pairs)  # don't sample ind positives as negatives
        train_neg_pairs = sample_bipartite_negatives(A, num_m, num_s, len(train_pos_e2), seed=seed+11, banned_pairs=banned)
        val_neg_pairs   = sample_bipartite_negatives(A, num_m, num_s, len(val_pos_e2),  seed=seed+22, banned_pairs=banned)

        train_neg_2e = pairs_to_edge_index(train_neg_pairs, num_m).to(device)
        val_neg_2e   = pairs_to_edge_index(val_neg_pairs, num_m).to(device)

        # Convert positives to [2,E] for model forward/test
        train_pos_2e = train_pos_e2.t().contiguous()
        val_pos_2e   = val_pos_e2.t().contiguous()

        # Train
        model, best_val_auc = train_model(
            data=data,
            train_graph_adj_2e=train_graph_adj_2e,
            train_pos_2e=train_pos_2e,
            train_neg_2e=train_neg_2e,
            val_pos_2e=val_pos_2e,
            val_neg_2e=val_neg_2e,
            args=args
        )

        # Evaluate Table-2 counts on independent set
        top_pos, bottom_neg, ind_auc, ind_ap = evaluate_independent_table2(
            model, data, train_graph_adj_2e, ind_pos_2e, ind_neg_2e
        )

        rows.append({
            "model": args.model_name,
            "dataset": args.dataset,
            "seed": seed,
            "n_pos": args.n_pos,
            "n_neg": args.n_neg,
            "top_pos_in_topN": top_pos,
            "bottom_neg_in_bottomN": bottom_neg,
            "ind_auc": ind_auc,
            "ind_aupr": ind_ap,
            "best_val_auc": best_val_auc,
            "assoc_path": assoc_path
        })

        print(f"[seed={seed}] TopPos@{args.n_pos}={top_pos}  BottomNeg@{args.n_neg}={bottom_neg}  "
              f"Ind-AUC={ind_auc:.4f}  Ind-AUPR={ind_ap:.4f}  (best val AUC={best_val_auc:.4f})")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.out_dir, "table2_independent_runs.csv")
    df.to_csv(out_csv, index=False)

    # Summary (mean±std)
    summ = df.agg({
        "top_pos_in_topN": ["mean", "std"],
        "bottom_neg_in_bottomN": ["mean", "std"],
        "ind_auc": ["mean", "std"],
        "ind_aupr": ["mean", "std"],
    })
    out_sum = os.path.join(args.out_dir, "table2_independent_summary.csv")
    summ.to_csv(out_sum)

    print("\nSaved:")
    print(f"  - {out_csv}")
    print(f"  - {out_sum}")

    # Print Table-2 style view
    top_mean = df["top_pos_in_topN"].mean()
    bot_mean = df["bottom_neg_in_bottomN"].mean()
    print("\nTable-2 style (averaged over seeds):")
    print(f"  {args.n_pos} positive samples | # in Top-{args.n_pos}: {top_mean:.2f}")
    print(f"  {args.n_neg} negative samples | # in Bottom-{args.n_neg}: {bot_mean:.2f}")


if __name__ == "__main__":
    main()
