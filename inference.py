"""
Production inference module for S&P 500 predictions
Refactored from your original code with production enhancements
"""
import os
import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Tuple, Optional

import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles loading and validation of trained models"""
    
    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
    
    def load(self):
        """Load model with comprehensive error handling"""
        abs_path = os.path.abspath(self.model_path)
        logger.info(f"Loading model from: {abs_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {abs_path}")
        
        try:
            self.model = joblib.load(self.model_path)
            
            # Try to extract feature names from model
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
                logger.info(f"Model expects {len(self.feature_names)} features: {self.feature_names}")
            else:
                logger.warning("Model doesn't store feature names. Will use config-based features.")
            
            logger.info("✅ Model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Failed to load model from {abs_path}: {e}") from e
    
    def get_model(self):
        """Get loaded model, loading if necessary"""
        if self.model is None:
            self.load()
        return self.model
    
    def get_feature_names(self) -> Optional[List[str]]:
        """Get expected feature names from model"""
        if self.model is None:
            self.load()
        return self.feature_names


class FeatureEngineer:
    """Handles feature engineering for model input"""
    
    def __init__(self, horizons: List[int] = config.HORIZONS):
        self.horizons = horizons
    
    def compute_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Compute rolling average and trend features
        
        Args:
            df: DataFrame with OHLCV data and Target column
        
        Returns:
            Tuple of (DataFrame with features, list of new feature names)
        """
        df = df.copy()
        new_predictors = []
        
        logger.info(f"Computing features for {len(self.horizons)} horizons: {self.horizons}")
        
        for horizon in self.horizons:
            # Rolling average ratio
            rolling_averages = df["Close"].rolling(horizon).mean()
            ratio_column = f"Close_Ratio_{horizon}"
            df[ratio_column] = df["Close"] / rolling_averages
            
            # Trend feature (shifted target rolling sum)
            trend_column = f"Trend_{horizon}"
            if "Target" not in df.columns:
                raise KeyError("compute_features requires a 'Target' column in the input DataFrame")
            df[trend_column] = df["Target"].shift(1).rolling(horizon).sum()
            
            new_predictors += [ratio_column, trend_column]
        
        logger.info(f"Created {len(new_predictors)} new features")
        return df, new_predictors


class PredictionEngine:
    """Handles model predictions with validation and error handling"""
    
    def __init__(self, model_loader: ModelLoader):
        self.model_loader = model_loader
    
    def predict(self, df: pd.DataFrame, predictors: List[str]) -> pd.Series:
        """
        Make predictions on DataFrame
        
        Args:
            df: DataFrame with all required features
            predictors: List of feature column names to use
        
        Returns:
            Series with predictions (index preserved from valid rows)
        """
        model = self.model_loader.get_model()
        
        # Validate predictor columns exist
        missing = [p for p in predictors if p not in df.columns]
        if missing:
            raise KeyError(
                f"Missing predictor columns: {missing}\n"
                f"Available columns: {list(df.columns)}\n"
                f"Required predictors: {predictors}"
            )
        
        # Extract features
        features = df[predictors]
        
        # Handle NaN values from rolling windows
        valid_mask = features.notna().all(axis=1)
        num_valid = valid_mask.sum()
        num_invalid = (~valid_mask).sum()
        
        logger.info(f"Prediction data: {num_valid} valid rows, {num_invalid} rows with NaNs")
        
        if not valid_mask.any():
            raise ValueError(
                "No rows without NaNs in predictor columns. "
                "This usually means insufficient historical data for rolling windows."
            )
        
        # Make predictions only on valid rows
        features_valid = features.loc[valid_mask]
        
        logger.info(f"Making predictions on {len(features_valid)} rows...")
        preds = model.predict(features_valid)
        
        # Return as Series with original index
        preds_series = pd.Series(preds, index=features_valid.index, name="Predictions")
        
        logger.info(f"✅ Generated {len(preds_series)} predictions")
        logger.info(f"Prediction distribution: {preds_series.value_counts().to_dict()}")
        
        return preds_series


class InferencePipeline:
    """Complete end-to-end inference pipeline"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.feature_engineer = FeatureEngineer()
        self.prediction_engine = PredictionEngine(self.model_loader)
        
        # Load model at initialization
        self.model_loader.load()
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete inference pipeline
        
        Args:
            df: Raw DataFrame with OHLCV data and Target
        
        Returns:
            DataFrame with original data + predictions
        """
        logger.info("="*60)
        logger.info("Starting inference pipeline")
        logger.info("="*60)
        
        # Step 1: Feature engineering
        logger.info("Step 1: Feature engineering...")
        df_with_features, new_predictors = self.feature_engineer.compute_features(df)
        
        # Step 2: Determine predictors
        logger.info("Step 2: Determining predictors...")
        
        # Use model's feature names if available, otherwise use config
        model_features = self.model_loader.get_feature_names()
        if model_features:
            predictors = model_features
            logger.info(f"Using model's feature names: {len(predictors)} features")
        else:
            predictors = config.BASE_PREDICTORS + new_predictors
            logger.info(f"Using config-based features: {len(predictors)} features")
        
        # Step 3: Make predictions
        logger.info("Step 3: Making predictions...")
        predictions = self.prediction_engine.predict(df_with_features, predictors)
        
        # Step 4: Combine results
        logger.info("Step 4: Combining results...")
        result_df = df_with_features.copy()
        result_df['Predictions'] = predictions
        
        # Add metadata
        result_df['Prediction_Date'] = pd.Timestamp.now()
        
        logger.info("="*60)
        logger.info("✅ Inference pipeline completed successfully")
        logger.info("="*60)
        
        return result_df


# Convenience functions for external use
def load_model() -> object:
    """Load and return the trained model"""
    loader = ModelLoader()
    return loader.load()


def make_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make predictions on a DataFrame
    
    Args:
        df: DataFrame with OHLCV data and Target column
    
    Returns:
        DataFrame with predictions added
    """
    pipeline = InferencePipeline()
    return pipeline.run(df)


# Testing
if __name__ == "__main__":
    print("=== Testing Inference Pipeline ===\n")
    
    # Create synthetic test data
    print("Creating synthetic test data...")
    dates = pd.date_range(end=pd.Timestamp.today(), periods=1500, freq='D')
    test_df = pd.DataFrame({
        'Open': np.random.uniform(3000, 5000, len(dates)),
        'High': np.random.uniform(3100, 5100, len(dates)),
        'Low': np.random.uniform(2900, 4900, len(dates)),
        'Close': np.random.uniform(3000, 5000, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates)),
        'Target': np.random.randint(0, 2, len(dates))
    }, index=dates)
    
    print(f"✅ Created test data with {len(test_df)} rows\n")
    
    # Test the pipeline
    try:
        print("Running inference pipeline...")
        pipeline = InferencePipeline()
        results = pipeline.run(test_df)
        
        print(f"\n✅ Pipeline successful!")
        print(f"\nResults shape: {results.shape}")
        print(f"\nColumns: {results.columns.tolist()}")
        print(f"\nPredictions summary:")
        print(results['Predictions'].describe())
        print(f"\nLast 10 predictions:")
        print(results[['Close', 'Predictions']].tail(10))
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()