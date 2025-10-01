import pandas as pd
import numpy as np

class Backtester:
    def __init__(self, panel, cash=1_000_000, slippage=0.001, commission=0.0):
        self.panel = panel
        self.cash_start = cash
        self.slippage = slippage
        self.commission = commission

    def run(self, signals):
        dates = sorted(set(self.panel.index.get_level_values(0)))
        cash = self.cash_start
        positions = {}
        equity = []

        for dt in dates:
            todays_signals = signals.loc[signals.index.get_level_values(0) == dt]                 if dt in signals.index.get_level_values(0) else None
            todays_prices = self.panel.loc[dt]["close"] if dt in self.panel.index.get_level_values(0) else {}

            if todays_signals is not None:
                for t, row in todays_signals.groupby("ticker"):
                    signal = row["signal"].iloc[0]
                    price = todays_prices.get(t, np.nan)
                    if np.isnan(price): continue

                    if signal == 1 and t not in positions:
                        qty = cash / (10 * price)  # allocate 10% of cash
                        cost = qty * price * (1 + self.slippage)
                        cash -= cost
                        positions[t] = (qty, price)
                    elif signal == -1 and t in positions:
                        qty, entry = positions.pop(t)
                        cash += qty * price * (1 - self.slippage)

            mtm = cash
            for t, (qty, entry) in positions.items():
                price = todays_prices.get(t, entry)
                mtm += qty * price
            equity.append((dt, mtm))

        return pd.DataFrame(equity, columns=["date", "equity"]).set_index("date")
