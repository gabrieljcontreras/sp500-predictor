"""
Production configuration for S&P 500 prediction system
"""
import os

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Data configuration
TICKER_SYMBOL = "^GSPC"  # S&P 500 index
DATA_PERIOD = "5y"  # How much historical data to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
DATA_INTERVAL = "1d"  # Data granularity (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

# Feature engineering configuration
HORIZONS = [2, 5, 60, 250, 1000]  # Must match training configuration!

# Predictor columns (must match training)
BASE_PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]

# Cache configuration (optional - for performance)
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_EXPIRY_HOURS = 1  # Refresh data every hour

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "predictions.log"