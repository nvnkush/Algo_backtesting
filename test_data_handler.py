import sys
import pandas as pd

# Add backtester folder to sys.path
sys.path.append(r"D:\Algo_trading\backtester")

from data_handler import DataHandler
from rank_handler import RankingHandler

# --- CONFIG ---
data_folder = r"D:\Algo_trading\Data\Day"

# --- Load data ---
handler = DataHandler(data_folder)

# Get panel (union of all tickers’ dates)
panel = handler.get_panel(use_union_dates=True)

# --- Initialize rank handler ---
ranker = RankingHandler(panel)

# --- 1. Momentum (5-day %) ---
momentum = ranker.momentum(period=5)
print("\nMomentum (last 5 rows):")
print(momentum.tail())

# --- 2. Volume rank (5-day rolling avg) ---
vol_rank = ranker.volume_rank(period=5)
print("\nVolume rank (last 5 rows):")
print(vol_rank.tail())

# --- 3. Custom metric (e.g., daily return volatility over 10 days) ---
def custom_volatility(panel):
    close = panel['close'].unstack(level=1)
    return close.pct_change().rolling(10).std()

custom_metric = ranker.custom_metric(custom_volatility)
print("\nCustom metric (10-day volatility, last 5 rows):")
print(custom_metric.tail())
