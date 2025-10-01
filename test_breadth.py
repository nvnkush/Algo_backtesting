import sys
import pandas as pd

# Add backtester folder to sys.path
sys.path.append(r"D:\Algo_trading\backtester")

from data_handler import DataHandler
from breadth import BreadthCalculator   # ✅ match file name

# --- CONFIG ---
data_folder = r"D:\Algo_trading\Data\Day"

# --- Load data ---
handler = DataHandler(data_folder)
handler.summary()

# --- Initialize breadth calculator ---
calc = BreadthCalculator(handler)

# --- 1. Basic breadth calculation ---
breadth = calc.compute(sma_period=50, use_common_dates=True)
print("\nBreadth (last 5 rows):")
print(breadth.tail())
print(f"\nBreadth shape: {breadth.shape}")

# --- 2. Multiple SMA breadth ---
breadth_multi = calc.compute_multiple_sma([20, 50, 200], use_common_dates=True)
print("\nBreadth with multiple SMAs (last 5 rows):")
print(breadth_multi.tail())
print(f"\nBreadth multi shape: {breadth_multi.shape}")

# --- 3. Latest summary ---
summary = calc.get_breadth_summary()
print("\nLatest Breadth Summary:")
print(summary)
