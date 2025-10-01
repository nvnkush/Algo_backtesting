from backtester.data_handler import DataHandler
from backtester.breadth import BreadthCalculator
from backtester.backtester import Backtester
from backtester.kpi_report import KPIReport
from Strategies.sma_crossover import SMACrossover

# Step 1: Load all stock CSVs
folder = r"D:\Algo_trading\Data\Day"
dh = DataHandler(folder)

# Step 2: Get panel (MultiIndex DataFrame)
panel = dh.get_panel()

# Step 3: Compute breadth
bc = BreadthCalculator(dh)
breadth = bc.compute()

# Step 4: Initialize original SMA strategy
strat = SMACrossover(short=20, long=50)  # original periods
signals = strat.generate_signals(panel, breadth)

# Step 5: Inspect signals
print("Signals head:")
print(signals.head(20))
print("Number of trades:", (signals['signal'] != 0).sum())

# Step 6: Run Backtester
bt = Backtester(panel)
equity_curve = bt.run(signals)

# Step 7: KPI
report = KPIReport(equity_curve)
summary = report.summary()

# Step 8: Display
print("Equity Curve head:")
print(equity_curve.head(20))
print("\nKPI Summary:")
for k, v in summary.items():
    print(f"{k}: {v}")
