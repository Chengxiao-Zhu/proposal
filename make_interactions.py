import pandas as pd

# MovieLens-100K u.data: user  item  rating  timestamp (tab-separated)
df = pd.read_csv("u.data", sep="\t", header=None, names=["user_id","item_id","rating","timestamp"])

# 最稳：先不过滤，直接当成隐式反馈
out = df[["user_id","item_id","timestamp"]]
out.to_csv("interactions.csv", index=False)
print("Saved interactions.csv with rows:", len(out))
print("Users:", out["user_id"].nunique(), "Items:", out["item_id"].nunique())