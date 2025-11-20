"""
Data fetching module for production S&P 500 predictions
Handles data acquisition, caching, and validation
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Optional

import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Handles fetching and caching of market data from Yahoo Finance"""
    
    def __init__(self, ticker: str = config.TICKER_SYMBOL):
        self.ticker = ticker
        self.cache_dir = config.CACHE_DIR
        self._setup_cache_dir()
    
    def _setup_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"Created cache directory: {self.cache_dir}")
    
    def _get_cache_path(self) -> str:
        """Get path for cached data file"""
        return os.path.join(self.cache_dir, f"{self.ticker.replace('^', '')}_data.csv")
    
    def _is_cache_valid(self) -> bool:
        """Check if cached data exists and is not expired"""
        cache_path = self._get_cache_path()
        
        if not os.path.exists(cache_path):
            logger.info("No cache found")
            return False
        
        # Check cache age
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_path))
        is_valid = cache_age < timedelta(hours=config.CACHE_EXPIRY_HOURS)
        
        if is_valid:
            logger.info(f"Cache is valid (age: {cache_age})")
        else:
            logger.info(f"Cache expired (age: {cache_age})")
        
        return is_valid
    
    def _load_from_cache(self) -> pd.DataFrame:
        """Load data from cache"""
        cache_path = self._get_cache_path()
        logger.info(f"Loading data from cache: {cache_path}")
        
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df
    
    def _save_to_cache(self, df: pd.DataFrame):
        """Save data to cache"""
        cache_path = self._get_cache_path()
        df.to_csv(cache_path)
        logger.info(f"Saved data to cache: {cache_path}")
    
    def fetch_data(self, 
                   period: str = config.DATA_PERIOD,
                   interval: str = config.DATA_INTERVAL,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch market data from Yahoo Finance or cache
        
        Args:
            period: Time period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            use_cache: Whether to use cached data if available
        
        Returns:
            DataFrame with OHLCV data
        """
        # Try loading from cache first
        if use_cache and self._is_cache_valid():
            try:
                return self._load_from_cache()
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}. Fetching fresh data.")
        
        # Fetch fresh data from Yahoo Finance
        logger.info(f"Fetching data for {self.ticker} (period={period}, interval={interval})")
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            df = ticker_obj.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError(f"No data returned for ticker {self.ticker}")
            
            # Validate required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            logger.info(f"Successfully fetched {len(df)} rows of data")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            
            # Save to cache
            if use_cache:
                self._save_to_cache(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            raise
    
    def prepare_data_for_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare fetched data for model inference
        
        Args:
            df: Raw dataframe from yfinance
        
        Returns:
            DataFrame ready for feature engineering
        """
        # Create a copy to avoid modifying original
        prepared_df = df.copy()
        
        # Keep only columns needed for prediction
        columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in columns_to_keep if col in prepared_df.columns]
        prepared_df = prepared_df[available_columns]
        
        # Create Target column (required by compute_features)
        # This is typically your label from training - adjust based on your training logic
        # Example: predict if next day's close is higher than today's
        prepared_df['Target'] = (prepared_df['Close'].shift(-1) > prepared_df['Close']).astype(int)
        
        # Drop the last row since it has NaN target
        prepared_df = prepared_df.iloc[:-1]
        
        # Sort by date (yfinance should already do this, but be explicit)
        prepared_df = prepared_df.sort_index()
        
        logger.info(f"Prepared {len(prepared_df)} rows for inference")
        
        return prepared_df


def get_latest_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Convenience function to get latest market data ready for inference
    
    Args:
        use_cache: Whether to use cached data
    
    Returns:
        DataFrame ready for feature engineering and prediction
    """
    fetcher = DataFetcher()
    raw_data = fetcher.fetch_data(use_cache=use_cache)
    prepared_data = fetcher.prepare_data_for_inference(raw_data)
    return prepared_data


# Example usage and testing
if __name__ == "__main__":
    print("=== Testing Data Fetcher ===\n")
    
    # Test 1: Fetch fresh data
    print("Test 1: Fetching fresh data...")
    df = get_latest_data(use_cache=False)
    print(f"✅ Fetched {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nLast 5 rows:\n{df.tail()}")
    
    # Test 2: Use cached data
    print("\n" + "="*50)
    print("Test 2: Using cached data...")
    df_cached = get_latest_data(use_cache=True)
    print(f"✅ Loaded {len(df_cached)} rows from cache")
    
    # Test 3: Validate data quality
    print("\n" + "="*50)
    print("Test 3: Data quality check...")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nStatistics:\n{df.describe()}")