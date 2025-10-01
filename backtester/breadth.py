import pandas as pd
import numpy as np
from typing import Optional, Union, List

class BreadthCalculator:
    def __init__(self, data_handler):
        """
        Initialize BreadthCalculator with a DataHandler instance.
        
        Args:
            data_handler: DataHandler instance containing stock data
        """
        self.dh = data_handler
        
        # Validate that we have data
        if not self.dh.data:
            raise ValueError("DataHandler contains no data")
    
    def _get_close_data(self, use_common_dates: bool = True) -> pd.DataFrame:
        """
        Extract close prices for all tickers aligned by dates.
        
        Args:
            use_common_dates: If True, use only dates common to all stocks.
                            If False, use union of all dates (with NaN for missing data).
        
        Returns:
            DataFrame with dates as index and tickers as columns
        """
        if use_common_dates:
            target_dates = self.dh.dates
            if target_dates.empty:
                print("Warning: No common dates found, using union of all dates")
                target_dates = self.dh.get_union_index()
        else:
            target_dates = self.dh.get_union_index()
        
        close_data = {}
        missing_close = []
        
        for ticker, df in self.dh.data.items():
            if "close" not in df.columns:
                missing_close.append(ticker)
                continue
            
            # Align data to target dates
            aligned_close = df["close"].reindex(target_dates)
            close_data[ticker] = aligned_close
        
        if missing_close:
            print(f"Warning: 'close' column missing for tickers: {missing_close}")
        
        if not close_data:
            raise ValueError("No valid close price data found")
        
        return pd.DataFrame(close_data, index=target_dates)
    
    def compute(self, 
                sma_period: int = 50, 
                use_common_dates: bool = True,
                min_stocks_for_calculation: int = 1) -> pd.DataFrame:
        """
        Compute market breadth indicators.
        
        Args:
            sma_period: Period for simple moving average calculation
            use_common_dates: Whether to use only common dates across all stocks
            min_stocks_for_calculation: Minimum number of stocks needed for valid calculation
        
        Returns:
            DataFrame with breadth indicators
        """
        if sma_period < 1:
            raise ValueError("sma_period must be positive")
        
        # Get aligned close prices
        close = self._get_close_data(use_common_dates)
        
        if close.empty:
            return pd.DataFrame()
        
        # Calculate returns
        returns = close.pct_change()
        
        # Count valid returns per day (exclude NaN)
        valid_returns = returns.notna().sum(axis=1)
        
        # Filter days with insufficient data
        sufficient_data_mask = valid_returns >= min_stocks_for_calculation
        
        # Calculate advancers and decliners (excluding NaN)
        adv = (returns > 0).sum(axis=1)
        dec = (returns < 0).sum(axis=1)
        unchanged = (returns == 0).sum(axis=1)
        
        # Calculate advance ratio (handle division by zero)
        total_movers = adv + dec
        adv_ratio = np.where(total_movers > 0, adv / total_movers, np.nan)
        
        # Calculate SMA and percentage above SMA
        sma = close.rolling(window=sma_period, min_periods=max(1, sma_period//2)).mean()
        
        # Count stocks above SMA (excluding NaN comparisons)
        above_sma = (close > sma).sum(axis=1)
        valid_sma_comparisons = ((close.notna()) & (sma.notna())).sum(axis=1)
        
        pct_above_sma = np.where(
            valid_sma_comparisons > 0, 
            above_sma / valid_sma_comparisons, 
            np.nan
        )
        
        # Create result DataFrame
        result = pd.DataFrame({
            "advancers": adv,
            "decliners": dec,
            "unchanged": unchanged,
            "total_stocks": valid_returns,
            "adv_ratio": adv_ratio,
            f"pct_above_sma_{sma_period}": pct_above_sma,
            "valid_for_sma": valid_sma_comparisons
        }, index=close.index)
        
        # Apply minimum stock filter
        result.loc[~sufficient_data_mask, :] = np.nan
        
        return result
    
    def compute_multiple_sma(self, 
                           sma_periods: List[int] = [20, 50, 200],
                           use_common_dates: bool = True) -> pd.DataFrame:
        """
        Compute breadth indicators with multiple SMA periods.
        
        Args:
            sma_periods: List of SMA periods to calculate
            use_common_dates: Whether to use only common dates across all stocks
        
        Returns:
            DataFrame with breadth indicators for multiple SMA periods
        """
        if not sma_periods:
            raise ValueError("sma_periods cannot be empty")
        
        # Get base indicators with first SMA period
        result = self.compute(sma_periods[0], use_common_dates)
        
        if len(sma_periods) == 1:
            return result
        
        # Add additional SMA periods
        close = self._get_close_data(use_common_dates)
        
        for period in sma_periods[1:]:
            if period < 1:
                continue
                
            sma = close.rolling(window=period, min_periods=max(1, period//2)).mean()
            above_sma = (close > sma).sum(axis=1)
            valid_comparisons = ((close.notna()) & (sma.notna())).sum(axis=1)
            
            pct_above_sma = np.where(
                valid_comparisons > 0,
                above_sma / valid_comparisons,
                np.nan
            )
            
            result[f"pct_above_sma_{period}"] = pct_above_sma
        
        return result
    
    def get_breadth_summary(self, date: Optional[str] = None) -> dict:
        """
        Get a summary of breadth indicators for a specific date.
        
        Args:
            date: Date string (YYYY-MM-DD) or None for latest date
        
        Returns:
            Dictionary with breadth indicators
        """
        breadth = self.compute()
        
        if breadth.empty:
            return {}
        
        if date is None:
            # Get latest date with data
            latest_idx = breadth.dropna().index[-1] if not breadth.dropna().empty else breadth.index[-1]
        else:
            try:
                latest_idx = pd.to_datetime(date)
                if latest_idx not in breadth.index:
                    raise KeyError(f"Date {date} not found in data")
            except (ValueError, KeyError) as e:
                raise ValueError(f"Invalid date {date}: {e}")
        
        row = breadth.loc[latest_idx]
        
        return {
            "date": latest_idx.strftime("%Y-%m-%d"),
            "advancers": int(row["advancers"]) if pd.notna(row["advancers"]) else None,
            "decliners": int(row["decliners"]) if pd.notna(row["decliners"]) else None,
            "unchanged": int(row["unchanged"]) if pd.notna(row["unchanged"]) else None,
            "advance_ratio": round(row["adv_ratio"], 3) if pd.notna(row["adv_ratio"]) else None,
            "pct_above_sma_50": round(row["pct_above_sma_50"], 3) if pd.notna(row["pct_above_sma_50"]) else None,
            "total_stocks": int(row["total_stocks"]) if pd.notna(row["total_stocks"]) else None
        }

# Example usage and testing
if __name__ == "__main__":
    # Assuming you have a DataHandler instance
    # handler = DataHandler("your_data_folder")
    # calc = BreadthCalculator(handler)
    
    # # Basic breadth calculation
    # breadth = calc.compute()
    # print("Breadth Indicators:")
    # print(breadth.tail())
    
    # # Multiple SMA periods
    # breadth_multi = calc.compute_multiple_sma([20, 50, 200])
    # print("\nMultiple SMA Breadth:")
    # print(breadth_multi.tail())
    
    # # Get summary for latest date
    # summary = calc.get_breadth_summary()
    # print("\nLatest Breadth Summary:")
    # print(summary)
    
    pass