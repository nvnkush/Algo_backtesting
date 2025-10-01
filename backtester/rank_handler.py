import pandas as pd

class RankingHandler:

    def __init__(self, panel: pd.DataFrame):
        if not isinstance(panel, pd.DataFrame):
            raise ValueError("panel must be a pandas DataFrame")
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("panel must have a MultiIndex [date, ticker]")

        self.panel = panel.copy()

    # ------------------------------
    # Momentum
    # ------------------------------
    def momentum(self, period: int = 5) -> pd.DataFrame:
        close = self.panel['close'].unstack(level=1)
        # Safely compute pct_change
        return close.pct_change(period).fillna(0)

    # ------------------------------
    # Rolling Volume Rank
    # ------------------------------
    def volume_rank(self, period: int = 5) -> pd.DataFrame:
        volume = self.panel['volume'].unstack(level=1)
        return volume.rolling(period, min_periods=1).mean().fillna(0)

    # ------------------------------
    # Custom metric
    # ------------------------------
    def custom_metric(self, func) -> pd.DataFrame:
        return func(self.panel)

    # ------------------------------
    # Relative Strength vs benchmark
    # ------------------------------
    def relative_strength(self, benchmark: str = "NIFTY50", method: str = "cumulative") -> pd.DataFrame:
        close = self.panel['close'].unstack(level=1)

        if benchmark not in close.columns:
            raise ValueError(f"Benchmark '{benchmark}' not found in panel")

        # Only keep dates where benchmark exists
        close = close.loc[close[benchmark].notna()]

        # Compute daily returns
        returns = close.pct_change().fillna(0)

        if method == "daily":
            # Daily relative strength = ticker return - benchmark return
            rs = returns.subtract(returns[benchmark], axis=0)
        elif method == "cumulative":
            # Cumulative relative strength = cumulative returns vs benchmark
            cum_returns = (1 + returns).cumprod()
            rs = cum_returns.div(cum_returns[benchmark], axis=0)
        else:
            raise ValueError("method must be 'daily' or 'cumulative'")

        # Drop benchmark column itself
        rs = rs.drop(columns=[benchmark])
        return rs

    # ------------------------------
    # Get top N tickers
    # ------------------------------
    def get_top_n(self, metric: pd.DataFrame, n: int = 10, date: str = None) -> pd.Series:
        """
        Return top N tickers for a given metric on a specific date
        """
        # Drop rows where all tickers are NaN
        metric_clean = metric.dropna(how='all')
        if metric_clean.empty:
            print("Warning: Metric has no valid data to rank!")
            return pd.Series(dtype=float)

        # Determine date
        if date is None:
            date_idx = metric_clean.index[-1]
        else:
            date_idx = pd.to_datetime(date)
            if date_idx not in metric_clean.index:
                print(f"Warning: Date {date} not found. Using latest available date.")
                date_idx = metric_clean.index[-1]

        # Get row and drop NaNs
        row = metric_clean.loc[date_idx].dropna()
        if row.empty:
            print(f"Warning: No valid tickers to rank on {date_idx}!")
            return pd.Series(dtype=float)

        # Return top N
        return row.sort_values(ascending=False).head(n)
