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

def reindex_from_train(train_df, test_df):
    # build mapping from train+test to avoid unseen ids
    u_vals = pd.concat([train_df["user_id"].astype(str), test_df["user_id"].astype(str)]).unique()
    i_vals = pd.concat([train_df["item_id"].astype(str), test_df["item_id"].astype(str)]).unique()
    u2idx = {u:i for i,u in enumerate(u_vals)}
    i2idx = {it:i for i,it in enumerate(i_vals)}

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["u"] = train_df["user_id"].astype(str).map(u2idx).astype(int)
    train_df["i"] = train_df["item_id"].astype(str).map(i2idx).astype(int)
    test_df["u"]  = test_df["user_id"].astype(str).map(u2idx).astype(int)
    test_df["i"]  = test_df["item_id"].astype(str).map(i2idx).astype(int)
    return train_df, test_df, len(u2idx), len(i2idx)

class BPRDataset(Dataset):
    def __init__(self, train_pairs, num_items, user_pos_items):
        self.users = [u for u,_ in train_pairs]
        self.pos_items = [i for _,i in train_pairs]
        self.num_items = num_items
        self.user_pos = user_pos_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]
        while True:
            neg = random.randrange(self.num_items)
            if neg not in self.user_pos[u]:
                break
        return u, pos, neg

class MF(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super().__init__()
        self.P = nn.Embedding(num_users, dim)
        self.Q = nn.Embedding(num_items, dim)
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)

    def score(self, u, i):
        return (self.P(u) * self.Q(i)).sum(dim=-1)

def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()

@torch.no_grad()
def evaluate(model, test_pos, train_pos, num_items, K=10, num_neg=99, device="cpu"):
    model.eval()
    recalls, ndcgs = [], []
    users = list(test_pos.keys())
    for u in users:
        pos_i = test_pos[u]
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
        while len(negs) < num_neg:
            j = random.randrange(num_items)
            if j != pos_i:
                negs.append(j)

        items = [pos_i] + negs
        u_tensor = torch.tensor([u]*len(items), dtype=torch.long, device=device)
        i_tensor = torch.tensor(items, dtype=torch.long, device=device)
        scores = model.score(u_tensor, i_tensor).cpu().numpy()
        rank = np.argsort(-scores)
        pos_rank = int(np.where(rank == 0)[0][0])
        hit = 1.0 if pos_rank < K else 0.0
        recalls.append(hit)
        if hit:
            ndcgs.append(1.0 / math.log2(pos_rank + 2))
        else:
            ndcgs.append(0.0)
    return float(np.mean(recalls)), float(np.mean(ndcgs))

def bytes_per_round(num_items, dim, dtype_bytes=4):
    return 2 * num_items * dim * dtype_bytes  # down + up

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--num_neg_eval", type=int, default=99)
    args = ap.parse_args()

    seed_all(args.seed)

    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)

    train_df, test_df, num_users, num_items = reindex_from_train(train_df, test_df)

    # build train pos sets
    train_pos = defaultdict(set)
    for u, i in zip(train_df["u"].values, train_df["i"].values):
        train_pos[int(u)].add(int(i))

    # build test dict (assume 1 row per user, but if not, last one wins)
    test_pos = {}
    for u, i in zip(test_df["u"].values, test_df["i"].values):
        test_pos[int(u)] = int(i)

    train_pairs = list(zip(train_df["u"].astype(int).tolist(), train_df["i"].astype(int).tolist()))
    dataset = BPRDataset(train_pairs, num_items, train_pos)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MF(num_users, num_items, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Fixed split | Users={num_users} Items={num_items} TrainRows={len(train_df)} TestUsers={len(test_pos)} Device={device}")

    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for u, pos, neg in loader:
            u = u.to(device); pos = pos.to(device); neg = neg.to(device)
            loss = bpr_loss(model.score(u, pos), model.score(u, neg))
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        rec, ndcg = evaluate(model, test_pos, train_pos, num_items, K=args.K, num_neg=args.num_neg_eval, device=device)
        print(f"Epoch {ep:02d} loss={np.mean(losses):.4f} Recall@{args.K}={rec:.4f} NDCG@{args.K}={ndcg:.4f}")

    b = bytes_per_round(num_items, args.dim, dtype_bytes=4)
    print(f"bytes/round (full item sync, fp32, up+down) = {b/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()