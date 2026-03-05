import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="train_triplets.txt")
    ap.add_argument("--outfile", default="interactions.csv")
    ap.add_argument("--max_lines", type=int, default=1_000_000)  # 先取100万行
    args = ap.parse_args()

    users = set()
    items = set()
    rows = 0

    with open(args.infile, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.outfile, "w", encoding="utf-8", newline="") as fout:
        fout.write("user_id,item_id\n")
        for line in fin:
            # format: user_id \t song_id \t play_count
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            u, i = parts[0], parts[1]
            fout.write(f"{u},{i}\n")
            users.add(u)
            items.add(i)
            rows += 1
            if rows >= args.max_lines:
                break

    print(f"Saved {args.outfile}")
    print(f"rows={rows} unique_users={len(users)} unique_items={len(items)}")

if __name__ == "__main__":
    main()