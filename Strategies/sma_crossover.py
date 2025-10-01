import pandas as pd
from Strategies.base_strategy import Strategy

class SMACrossover(Strategy):
    def __init__(self, short=20, long=50):
        super().__init__(f"SMA_{short}_{long}")
        self.short = short
        self.long = long

    def generate_signals(self, panel, breadth):
        close = panel["close"].unstack(level=1)
        short_sma = close.rolling(self.short).mean()
        long_sma = close.rolling(self.long).mean()
        sigs = (short_sma > long_sma).astype(int).diff().fillna(0)
        signals = []
        for dt in sigs.index:
            for t in sigs.columns:
                if sigs.loc[dt, t] == 1:
                    signals.append((dt, t, 1))
                elif sigs.loc[dt, t] == -1:
                    signals.append((dt, t, -1))
        return pd.DataFrame(signals, columns=["date","ticker","signal"]).set_index(["date","ticker"])
