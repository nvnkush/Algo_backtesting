import sys
import pandas as pd

# -------------------------------
# Add backtester folder to sys.path
# -------------------------------
sys.path.append(r"D:\Algo_trading\backtester")

from data_handler import DataHandler
from rank_handler import RankingHandler

# -------------------------------
# CONFIG
# -------------------------------
stock_folder = r"D:\Algo_trading\Data\Day"
nifty_file = r"D:\Algo_trading\Data\Day\NIFTY50.csv"
top_n = 10  # number of top tickers to select
momentum_period = 5
volume_period = 5

# -------------------------------
# LOAD STOCK DATA
# -------------------------------
handler = DataHandler(stock_folder)
panel = handler.get_panel(use_union_dates=True)

# Make all dates tz-naive
panel = panel.copy()
panel.index = panel.index.set_levels(
    [panel.index.levels[0].tz_localize(None), panel.index.levels[1]]
)

# -------------------------------
# LOAD NIFTY50 DATA
# -------------------------------
nifty = pd.read_csv(nifty_file, parse_dates=["date"], index_col="date")

# Ensure tz-naive
nifty.index = nifty.index.tz_localize(None)

# If volume column missing, create dummy
if "volume" not in nifty.columns:
    nifty["volume"] = 0

# Convert to MultiIndex format
nifty_panel = nifty.copy()
nifty_panel["ticker"] = "NIFTY50"
nifty_panel = nifty_panel[["close", "volume", "ticker"]]
nifty_panel = nifty_panel.reset_index().set_index(["date", "ticker"])

# Drop duplicates in nifty_panel
nifty_panel = nifty_panel[~nifty_panel.index.duplicated(keep="last")]

# If NIFTY50 already exists in panel, drop it
if "NIFTY50" in panel.index.get_level_values("ticker"):
    panel = panel.drop("NIFTY50", level="ticker")

# Merge clean
panel = pd.concat([panel, nifty_panel]).sort_index()

# -------------------------------
# INITIALIZE RANKING HANDLER
# -------------------------------
ranker = RankingHandler(panel)

# -------------------------------
# 1. Momentum
# -------------------------------
momentum = ranker.momentum(momentum_period)
print(f"\nMomentum ({momentum_period}-day) last 5 rows:")
print(momentum.tail())

# -------------------------------
# 2. Volume rank
# -------------------------------
vol_rank = ranker.volume_rank(volume_period)
print(f"\nVolume rank ({volume_period}-day rolling) last 5 rows:")
print(vol_rank.tail())

# -------------------------------
# 3. Relative Strength vs NIFTY50
# -------------------------------
rs = ranker.relative_strength("NIFTY50")
print("\nRelative Strength vs NIFTY50 last 5 rows:")
print(rs.tail())

# -------------------------------
# 4. Top N tickers by momentum (latest date)
# -------------------------------
top_mom = ranker.get_top_n(momentum, n=top_n)
print(f"\nTop {top_n} tickers by {momentum_period}-day momentum:")
print(top_mom)

# -------------------------------
# 5. Top N tickers by relative strength (latest date)
# -------------------------------
top_rs = ranker.get_top_n(rs, n=top_n)
print(f"\nTop {top_n} tickers by relative strength vs NIFTY50:")
print(top_rs)
