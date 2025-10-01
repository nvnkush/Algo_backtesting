import pandas as pd

def sma(series: pd.Series, period: int):
    return series.rolling(period).mean()

def ema(series: pd.Series, period: int):
    return series.ewm(span=period).mean()

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(period).mean()
    down = -delta.clip(upper=0).rolling(period).mean()
    rs = up / down
    return 100 - (100 / (1 + rs))
