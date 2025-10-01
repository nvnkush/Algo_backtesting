import numpy as np

class KPIReport:
    def __init__(self, equity_curve):
        self.eq = equity_curve

    def summary(self):
        eq = self.eq.copy()
        eq["ret"] = eq["equity"].pct_change().fillna(0)
        total_ret = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
        ann_ret = (1 + total_ret) ** (252 / len(eq)) - 1
        vol = eq["ret"].std() * (252 ** 0.5)
        sharpe = ann_ret / vol if vol != 0 else np.nan
        dd = (eq["equity"] / eq["equity"].cummax() - 1).min()
        return {
            "Total Return": total_ret,
            "Annualized Return": ann_ret,
            "Sharpe": sharpe,
            "Max Drawdown": dd
        }
