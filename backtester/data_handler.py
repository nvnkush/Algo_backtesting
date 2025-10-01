import os
import glob
import pandas as pd
from typing import Dict, Optional, List

class DataHandler:
    def __init__(self, folder: str):
        """
        Initialize DataHandler with a folder containing CSV files.
        
        Args:
            folder: Path to folder containing CSV files with stock data
        """
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Folder '{folder}' does not exist")
        
        self.folder = folder
        self.data = self._load_data()
        self.tickers = sorted(self.data.keys())
        self.dates = self._get_common_index()

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV files from the folder."""
        files = glob.glob(os.path.join(self.folder, "*.csv"))
        
        if not files:
            raise ValueError(f"No CSV files found in folder '{self.folder}'")
        
        data = {}
        failed_files = []
        
        for file_path in files:
            try:
                ticker = os.path.splitext(os.path.basename(file_path))[0]
                
                # Read CSV
                df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
                
                if df.empty:
                    print(f"Warning: Empty file {file_path}")
                    continue
                
                # Ensure datetime index and drop timezone if present
                df.index = pd.to_datetime(df.index, errors="coerce")
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                df = df.sort_index()
                
                # Remove duplicated dates if any
                if df.index.duplicated().any():
                    print(f"Warning: Duplicate dates found in {file_path}, keeping first occurrence")
                    df = df[~df.index.duplicated(keep='first')]
                
                data[ticker] = df
                
            except Exception as e:
                failed_files.append((file_path, str(e)))
                print(f"Error loading {file_path}: {e}")
        
        if failed_files:
            print(f"Failed to load {len(failed_files)} files")
        
        if not data:
            raise ValueError("No valid data files were loaded")
        
        return data

    def _get_common_index(self) -> pd.DatetimeIndex:
        """Get intersection of all date indices (truly common dates)."""
        if not self.data:
            return pd.DatetimeIndex([])
        
        all_indices = list(self.data.values())
        common_index = all_indices[0].index
        
        for df in all_indices[1:]:
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            common_index = common_index.intersection(idx)
        
        return common_index.sort_values()
    
    def get_union_index(self) -> pd.DatetimeIndex:
        """Get union of all date indices (all dates from all files)."""
        if not self.data:
            return pd.DatetimeIndex([])
        
        all_indices = []
        for df in self.data.values():
            idx = df.index
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            all_indices.append(idx)

        union_index = all_indices[0]
        for idx in all_indices[1:]:
            union_index = union_index.union(idx)
        
        return union_index.sort_values()

    def get_panel(self, use_union_dates: bool = False, 
                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a panel DataFrame with MultiIndex (date, ticker).
        """
        if not self.data:
            return pd.DataFrame()
        
        # Choose date index
        target_dates = self.get_union_index() if use_union_dates else self.dates
        
        if target_dates.empty:
            print("Warning: No common dates found across all files")
            target_dates = self.get_union_index()
        
        frames = []
        for ticker, df in self.data.items():
            # Select columns if specified
            if columns:
                available_cols = [col for col in columns if col in df.columns]
                if available_cols != columns:
                    missing_cols = set(columns) - set(available_cols)
                    print(f"Warning: Columns {missing_cols} not found in {ticker}")
                df_subset = df[available_cols] if available_cols else df
            else:
                df_subset = df
            
            # Reindex to align dates
            df_aligned = df_subset.reindex(target_dates)
            df_aligned["ticker"] = ticker
            frames.append(df_aligned)
        
        if not frames:
            return pd.DataFrame()
        
        # Concatenate and create MultiIndex
        panel = pd.concat(frames, ignore_index=False)
        panel.index.name = "date"
        
        return panel.reset_index().set_index(["date", "ticker"]).sort_index()
    
    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Get data for a specific ticker."""
        if ticker not in self.data:
            raise KeyError(f"Ticker '{ticker}' not found. Available tickers: {self.tickers}")
        return self.data[ticker].copy()
    
    def get_date_range(self) -> tuple:
        """Get the overall date range across all data."""
        if not self.data:
            return None, None
        
        min_date = min(df.index.min() for df in self.data.values())
        max_date = max(df.index.max() for df in self.data.values())
        return min_date, max_date
    
    def summary(self) -> None:
        """Print a summary of loaded data."""
        print(f"DataHandler Summary:")
        print(f"  Folder: {self.folder}")
        print(f"  Tickers loaded: {len(self.tickers)}")
        print(f"  Tickers: {', '.join(self.tickers)}")
        
        if self.data:
            min_date, max_date = self.get_date_range()
            print(f"  Date range: {min_date.date()} to {max_date.date()}")
            print(f"  Common dates: {len(self.dates)}")
            print(f"  Union dates: {len(self.get_union_index())}")
            
            print(f"\n  Data shapes:")
            for ticker in self.tickers:
                shape = self.data[ticker].shape
                print(f"    {ticker}: {shape[0]} rows, {shape[1]} columns")
