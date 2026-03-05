import os
import argparse
import random
from collections import defaultdict, Counter

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def read_triplets_first_n(path, max_lines):
    # returns list of (u, i)
    pairs = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            u, i = parts[0], parts[1]
            pairs.append((u, i))
            if len(pairs) >= max_lines:
                break
    return pairs

def filter_pairs(pairs, min_user, min_item_users, max_iter=5):
    """
    Iterative filtering:
      - drop users with < min_user interactions
      - drop items with < min_item_users distinct users
    Repeat until stable or max_iter.
    """
    cur = pairs
    for _ in range(max_iter):
        # user counts
        ucnt = Counter(u for u, _ in cur)

        # item -> set(users) count (distinct users per item)
        item_users = defaultdict(set)
        for u, i in cur:
            item_users[i].add(u)
        icnt_users = {i: len(us) for i, us in item_users.items()}

        # keep sets
        keep_users = {u for u, c in ucnt.items() if c >= min_user}
        keep_items = {i for i, c in icnt_users.items() if c >= min_item_users}

        nxt = [(u, i) for (u, i) in cur if (u in keep_users and i in keep_items)]
        if len(nxt) == len(cur):
            break
        cur = nxt
    return cur

def subsample_pairs(pairs, max_rows, seed):
    if len(pairs) <= max_rows:
        return pairs
    rnd = random.Random(seed)
    # sample indices without loading huge extra structures
    return rnd.sample(pairs, max_rows)

def write_interactions_csv(pairs, out_path):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("user_id,item_id\n")
        for u, i in pairs:
            f.write(f"{u},{i}\n")

def leave_one_out_split(pairs, seed):
    """
    No timestamp in MSD triplets here => random leave-one-out per user.
    Returns train_pairs, test_pairs (one test per user if user has >=2 interactions)
    """
    rnd = random.Random(seed)
    by_user = defaultdict(list)
    for u, i in pairs:
        by_user[u].append(i)

    train = []
    test = []
    for u, items in by_user.items():
        if len(items) < 2:
            # too few interactions: keep all in train
            for i in items:
                train.append((u, i))
            continue
        idx = rnd.randrange(len(items))
        test_item = items[idx]
        test.append((u, test_item))
        for j, i in enumerate(items):
            if j != idx:
                train.append((u, i))
    return train, test

def stats(pairs):
    users = set(u for u, _ in pairs)
    items = set(i for _, i in pairs)
    ucnt = Counter(u for u, _ in pairs)
    icnt = Counter(i for _, i in pairs)
    avg_u = sum(ucnt.values()) / max(1, len(users))
    avg_i = sum(icnt.values()) / max(1, len(items))
    return {
        "rows": len(pairs),
        "users": len(users),
        "items": len(items),
        "avg_interactions_per_user": avg_u,
        "avg_interactions_per_item": avg_i,
        "min_user_interactions": min(ucnt.values()) if ucnt else 0,
        "min_item_interactions": min(icnt.values()) if icnt else 0,
        "median_user_interactions": sorted(ucnt.values())[len(ucnt)//2] if ucnt else 0,
        "median_item_interactions": sorted(icnt.values())[len(icnt)//2] if icnt else 0,
    }

def write_stats(path, header, d):
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{header}\n")
        for k, v in d.items():
            f.write(f"- {k}: {v}\n")
        f.write("\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triplets", default="train_triplets.txt")
    ap.add_argument("--max_lines_read", type=int, default=1_000_000)   # 先读100万行
    ap.add_argument("--min_user", type=int, default=20)                # 默认 user>=20
    ap.add_argument("--min_item_users", type=int, default=10)          # 默认 item>=10 users
    ap.add_argument("--max_rows_keep", type=int, default=1_000_000)    # 过滤后最多保留100万行
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default=r".\processed")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    # 1) read subset
    raw = read_triplets_first_n(args.triplets, args.max_lines_read)
    raw_stats = stats(raw)

    # 2) filter iteratively
    filt = filter_pairs(raw, args.min_user, args.min_item_users)
    filt_stats = stats(filt)

    # 3) subsample (if still huge)
    kept = subsample_pairs(filt, args.max_rows_keep, args.seed)
    kept_stats = stats(kept)

    # 4) split
    train, test = leave_one_out_split(kept, args.seed)
    train_stats = stats(train)
    test_stats = stats(test)

    # 5) write files
    interactions_path = os.path.join(args.outdir, "interactions_filtered.csv")
    train_path = os.path.join(args.outdir, "train.csv")
    test_path = os.path.join(args.outdir, "test.csv")
    stats_path = os.path.join(args.outdir, "stats.txt")

    # overwrite stats file
    if os.path.exists(stats_path):
        os.remove(stats_path)

    write_interactions_csv(kept, interactions_path)
    write_interactions_csv(train, train_path)
    write_interactions_csv(test, test_path)

    write_stats(stats_path, "RAW (first N lines)", raw_stats)
    write_stats(stats_path, f"FILTERED (min_user={args.min_user}, min_item_users={args.min_item_users})", filt_stats)
    write_stats(stats_path, f"KEPT (after subsample max_rows_keep={args.max_rows_keep})", kept_stats)
    write_stats(stats_path, "TRAIN (leave-one-out)", train_stats)
    write_stats(stats_path, "TEST (leave-one-out, 1 per user when possible)", test_stats)

    print("Done.")
    print("Wrote:")
    print(" -", interactions_path)
    print(" -", train_path)
    print(" -", test_path)
    print(" -", stats_path)

if __name__ == "__main__":
    main()