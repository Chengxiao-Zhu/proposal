# ncf_fixed_split.py
# Fixed train/test split NCF with NEGATIVE SAMPLING RATIO (neg_k) = default 4
# Usage:
#   python .\ncf_fixed_split.py --train .\data\processed\train.csv --test .\data\processed\test.csv --epochs 5 --dim 64 --neg_k 4
#
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
    u_vals = pd.concat([train_df["user_id"].astype(str), test_df["user_id"].astype(str)]).unique()
    i_vals = pd.concat([train_df["item_id"].astype(str), test_df["item_id"].astype(str)]).unique()
    u2idx = {u:i for i,u in enumerate(u_vals)}
    i2idx = {it:i for i,it in enumerate(i_vals)}
    train_df = train_df.copy(); test_df = test_df.copy()
    train_df["u"] = train_df["user_id"].astype(str).map(u2idx).astype(int)
    train_df["i"] = train_df["item_id"].astype(str).map(i2idx).astype(int)
    test_df["u"]  = test_df["user_id"].astype(str).map(u2idx).astype(int)
    test_df["i"]  = test_df["item_id"].astype(str).map(i2idx).astype(int)
    return train_df, test_df, len(u2idx), len(i2idx)

class NegKDataset(Dataset):
    """
    For each positive (u, pos_i), sample neg_k negatives.
    Returns: u_pos, i_pos, neg_items[neg_k]
    """
    def __init__(self, train_pairs, num_items, user_pos_items, neg_k=4):
        self.users = [u for u,_ in train_pairs]
        self.pos_items = [i for _,i in train_pairs]
        self.num_items = num_items
        self.user_pos = user_pos_items
        self.neg_k = neg_k

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.pos_items[idx]
        negs = []
        # sample neg_k items not in user's positive set
        while len(negs) < self.neg_k:
            neg = random.randrange(self.num_items)
            if neg not in self.user_pos[u]:
                negs.append(neg)
        # return tensors to make collation stable
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(negs, dtype=torch.long),
        )

class NCF(nn.Module):
    """
    Simple NCF:
      - embedding for user/item
      - concat -> MLP -> score (logit)
    """
    def __init__(self, num_users, num_items, dim, hidden=(128, 64)):
        super().__init__()
        self.P = nn.Embedding(num_users, dim)
        self.Q = nn.Embedding(num_items, dim)
        layers = []
        in_dim = dim * 2
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]
        self.mlp = nn.Sequential(*layers)
        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)

    def forward(self, u, i):
        x = torch.cat([self.P(u), self.Q(i)], dim=-1)
        return self.mlp(x).squeeze(-1)  # logits

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
        logits = model(u_tensor, i_tensor).cpu().numpy()
        scores = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
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
    return 2 * num_items * dim * dtype_bytes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--num_neg_eval", type=int, default=99)
    ap.add_argument("--neg_k", type=int, default=4)  # <-- key change
    args = ap.parse_args()

    seed_all(args.seed)

    train_df = pd.read_csv(args.train)
    test_df  = pd.read_csv(args.test)
    train_df, test_df, num_users, num_items = reindex_from_train(train_df, test_df)

    train_pos = defaultdict(set)
    for u, i in zip(train_df["u"].values, train_df["i"].values):
        train_pos[int(u)].add(int(i))
    test_pos = {}
    for u, i in zip(test_df["u"].values, test_df["i"].values):
        test_pos[int(u)] = int(i)

    train_pairs = list(zip(train_df["u"].astype(int).tolist(), train_df["i"].astype(int).tolist()))
    dataset = NegKDataset(train_pairs, num_items, train_pos, neg_k=args.neg_k)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NCF(num_users, num_items, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    print(f"Fixed split | NCF(neg_k={args.neg_k}) | Users={num_users} Items={num_items} "
          f"TrainRows={len(train_df)} TestUsers={len(test_pos)} Device={device}")

    for ep in range(1, args.epochs+1):
        model.train()
        losses = []
        for u_pos, i_pos, negs in loader:
            # u_pos: [B], i_pos: [B], negs: [B, neg_k]
            u_pos = u_pos.to(device)
            i_pos = i_pos.to(device)
            negs = negs.to(device)

            # expand users for negatives
            u_neg = u_pos.unsqueeze(1).expand_as(negs)  # [B, neg_k]

            # concat positives + negatives
            u = torch.cat([u_pos, u_neg.reshape(-1)], dim=0)
            i = torch.cat([i_pos, negs.reshape(-1)], dim=0)

            # labels
            y_pos = torch.ones_like(u_pos, dtype=torch.float, device=device)
            y_neg = torch.zeros(u_neg.numel(), dtype=torch.float, device=device)
            y = torch.cat([y_pos, y_neg], dim=0)

            logits = model(u, i)
            loss = bce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        rec, ndcg = evaluate(model, test_pos, train_pos, num_items,
                             K=args.K, num_neg=args.num_neg_eval, device=device)
        print(f"Epoch {ep:02d} loss={np.mean(losses):.4f} Recall@{args.K}={rec:.4f} NDCG@{args.K}={ndcg:.4f}")

    b = bytes_per_round(num_items, args.dim, dtype_bytes=4)
    print(f"bytes/round (full item sync, fp32, up+down) = {b/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()