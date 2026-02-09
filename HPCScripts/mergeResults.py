import glob
import pandas as pd

tmp_files = sorted(glob.glob("results/_tmp_*.csv"))
assert len(tmp_files) > 0, "No temp files found!"

dfs = [pd.read_csv(f, index_col=0) for f in tmp_files]
final_df = pd.concat(dfs, axis=0)

final_df.to_csv("results/BiasOut1LR_ALL.csv")
print("Wrote results/BiasOut1LR_ALL.csv")
