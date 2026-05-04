import pandas as pd

work = pd.DataFrame({"PLOT_ID": ["1", "2"], "CIgreen_mean": [float('nan'), float('nan')]})
ts_means = pd.DataFrame({"PLOT_ID": ["1", "2"], "CIgreen_mean": [0.5, 0.6]})

work = work.merge(ts_means, on="PLOT_ID", how="left")
print(work.columns)

candidate_cols = ["CIgreen_mean"]
try:
    work[candidate_cols]
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")
