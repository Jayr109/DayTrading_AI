"""Helper: append bars from JSON file into existing CSV (deduplicates by datetime index)."""
import sys, json, pandas as pd
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text())
symbol = sys.argv[2]   # e.g. MNQ1
tf     = sys.argv[3]   # e.g. 5M

bars = data if isinstance(data, list) else data.get("bars", [])
df_new = pd.DataFrame(bars)
df_new.columns = [c.lower() for c in df_new.columns]
df_new["datetime"] = pd.to_datetime(df_new["time"], unit="s", utc=True)
df_new = df_new.drop(columns=["time"]).set_index("datetime").sort_index()

out = Path(__file__).parent.parent / "data" / "raw" / f"{symbol}_{tf}.csv"

if out.exists():
    df_existing = pd.read_csv(out, index_col="datetime", parse_dates=True)
    df_existing.index = pd.to_datetime(df_existing.index, utc=True)
    df_combined = pd.concat([df_existing, df_new])
    df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
else:
    df_combined = df_new

df_combined.to_csv(out)
print(f"Saved {len(df_combined)} bars -> {out}  [{df_combined.index[0]} - {df_combined.index[-1]}]  (+{len(df_new)} new)")
