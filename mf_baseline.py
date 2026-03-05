import argparse, random, math
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def reindex_ids(df, user_col="user_id", item_col="item_id"):
    u_vals = df[user_col].astype(str).unique()
    i_vals = df[item_col].astype(str).unique()
    u2idx = {u:i for i,u in enumerate(u_vals)}
    i2idx = {it:i for i,it in enumerate(i_vals)}
    df["u"] = df[user_col].astype(str).map(u2idx).astype(int)
    df["i"] = df[item_col].astype(str).map(i2idx).astype(int)
    return df, len(u2idx), len(i2idx)

def leave_one_out_split(df, has_ts: bool):
    # returns train_df, test_df (one interaction per user in test if possible)
    test_rows = []
    train_rows = []
    by_user = df.groupby("u")
    for u, g in by_user:
        if len(g) < 2:
            # too few interactions: keep in train only (or you can drop)
            train_rows.append(g)
            continue
        if has_ts:
            g_sorted = g.sort_values("timestamp")
            test = g_sorted.iloc[-1:]
            train = g_sorted.iloc[:-1]
        else:
            idx = np.random.randint(0, len(g))
            test = g.iloc[idx:idx+1]
            train = g.drop(g.index[idx])
        test_rows.append(test)
        train_rows.append(train)
    train_df = pd.concat(train_rows, ignore_index=True)
    test_df  = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(columns=df.columns)
    return train_df, test_df

class BPRDataset(Dataset):
    def __init__(self, train_pairs, num_items, user_pos_items):
        self.train_pairs = train_pairs  # list of (u, pos_i)
        self.num_items = num_items
        self.user_pos = user_pos_items  # dict u -> set(pos items in train)
        self.users = [u for u,_ in train_pairs]
        self.pos_items = [i for _,i in train_pairs]

    def __len__(self):
        return len(self.train_pairs)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]
        # negative sampling
        while True:
            neg = random.randrange(self.num_items)
            if neg not in self.user_pos[u]:
                break
        return u, pos, neg

class MF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super().__init__()
        self.P = nn.Embedding(num_users, dim)  # user emb
        self.Q = nn.Embedding(num_items, dim)  # item emb
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)

    def score(self, u, i):
        pu = self.P(u)
        qi = self.Q(i)
        return (pu * qi).sum(dim=-1)

def bpr_loss(pos_scores, neg_scores):
    # -log sigma(pos-neg)
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

@torch.no_grad()
def evaluate_recall_ndcg(model: MF, test_pos, train_pos, num_items, K=10, num_neg=99, device="cpu"):
    """
    test_pos: dict u -> test_item (single)
    train_pos: dict u -> set(items in train) (avoid sampling those as negatives if possible)
    Evaluate with 1 positive + num_neg negatives per user.
    """
    model.eval()
    recalls = []
    ndcgs = []
    users = list(test_pos.keys())
    for u in users:
        pos_i = test_pos[u]
        # sample negatives
        negs = []
        tried = 0
        while len(negs) < num_neg and tried < num_neg * 20:
            j = random.randrange(num_items)
            tried += 1
            if j == pos_i:
                continue
            if j in train_pos.get(u, set()):
                continue
            negs.append(j)
        # if not enough negatives (rare), fill anyway without filtering train_pos
        while len(negs) < num_neg:
            j = random.randrange(num_items)
            if j != pos_i:
                negs.append(j)

        items = [pos_i] + negs
        u_tensor = torch.tensor([u]*len(items), dtype=torch.long, device=device)
        i_tensor = torch.tensor(items, dtype=torch.long, device=device)

        scores = model.score(u_tensor, i_tensor).cpu().numpy()
        # rank descending
        rank_idx = np.argsort(-scores)
        topk = rank_idx[:K]
        # position of positive (which is index 0 in items list)
        pos_rank = np.where(rank_idx == 0)[0][0]  # 0-based rank among 100 items
        hit = 1.0 if pos_rank < K else 0.0
        recalls.append(hit)
        if hit > 0:
            # NDCG@K for single positive: 1/log2(rank+2)
            ndcgs.append(1.0 / math.log2(pos_rank + 2))
        else:
            ndcgs.append(0.0)
    return float(np.mean(recalls)), float(np.mean(ndcgs))

def bytes_per_round_full_sync(num_items, dim, dtype_bytes=4, up_and_down=True):
    base = num_items * dim * dtype_bytes
    return base * (2 if up_and_down else 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--num_neg_eval", type=int, default=99)
    args = ap.parse_args()

    seed_all(args.seed)
    df = pd.read_csv(args.data)
    has_ts = "timestamp" in df.columns
    df, num_users, num_items = reindex_ids(df, "user_id", "item_id")
    if has_ts:
        # make sure timestamp is numeric or sortable
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0)

    train_df, test_df = leave_one_out_split(df, has_ts)

    # build user->pos sets for train
    train_pos = defaultdict(set)
    for u, i in zip(train_df["u"].values, train_df["i"].values):
        train_pos[int(u)].add(int(i))

    # build test dict: one item per user
    test_pos = {}
    for u, i in zip(test_df["u"].values, test_df["i"].values):
        test_pos[int(u)] = int(i)

    train_pairs = list(zip(train_df["u"].astype(int).tolist(), train_df["i"].astype(int).tolist()))
    dataset = BPRDataset(train_pairs, num_items, train_pos)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MF(num_users, num_items, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Users={num_users} Items={num_items} Interactions={len(df)} "
          f"Train={len(train_df)} TestUsers={len(test_pos)} Device={device}")

    # train
    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for u, pos, neg in loader:
            u = u.to(device)
            pos = pos.to(device)
            neg = neg.to(device)
            pos_s = model.score(u, pos)
            neg_s = model.score(u, neg)
            loss = bpr_loss(pos_s, neg_s)

            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        rec, ndcg = evaluate_recall_ndcg(model, test_pos, train_pos, num_items,
                                         K=args.K, num_neg=args.num_neg_eval, device=device)
        print(f"Epoch {ep:02d} loss={np.mean(losses):.4f} Recall@{args.K}={rec:.4f} NDCG@{args.K}={ndcg:.4f}")

    # system metric (for later FL)
    b = bytes_per_round_full_sync(num_items, args.dim, dtype_bytes=4, up_and_down=True)
    print(f"bytes/round (full item sync, fp32, up+down) = {b/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()