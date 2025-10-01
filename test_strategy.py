from backtester.data_handler import DataHandler
from backtester.breadth import BreadthCalculator
from Strategies.sma_crossover import SMACrossover

# Step 1: Load all stock CSVs
folder = r"D:\Algo_trading\Data\Day"
dh = DataHandler(folder)

# Step 2: Get panel (MultiIndex DataFrame)
panel = dh.get_panel()

# Step 3: Compute breadth indicators
bc = BreadthCalculator(dh)
breadth = bc.compute()

# Step 4: Initialize strategy
strat = SMACrossover(short=20, long=50)

# Step 5: Generate signals
signals = strat.generate_signals(panel, breadth)

# Step 6: Inspect first few signals
print(signals.head())
