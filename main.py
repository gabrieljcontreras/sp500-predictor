"""
Main entry point for S&P 500 prediction system
Combines data fetching and inference for production use
"""
import sys
import argparse
import pandas as pd
from datetime import datetime
import logging

import config
from data_fetcher import get_latest_data, DataFetcher
from inference import InferencePipeline, make_predictions

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def predict_latest(use_cache: bool = True, save_results: bool = True) -> pd.DataFrame:
    """
    Fetch latest data and generate predictions
    
    Args:
        use_cache: Whether to use cached data
        save_results: Whether to save predictions to CSV
    
    Returns:
        DataFrame with predictions
    """
    try:
        logger.info("="*80)
        logger.info("STARTING S&P 500 PREDICTION PIPELINE")
        logger.info("="*80)
        
        # Step 1: Fetch latest data
        logger.info("\n📊 STEP 1: Fetching market data...")
        df = get_latest_data(use_cache=use_cache)
        logger.info(f"✅ Data fetched: {len(df)} rows from {df.index.min()} to {df.index.max()}")
        
        # Step 2: Run inference
        logger.info("\n🤖 STEP 2: Running model inference...")
        results = make_predictions(df)
        logger.info(f"✅ Predictions generated for {results['Predictions'].notna().sum()} data points")
        
        # Step 3: Display results
        logger.info("\n📈 STEP 3: Results Summary")
        logger.info("-" * 80)
        
        # Get latest predictions (most recent dates with valid predictions)
        latest_results = results[results['Predictions'].notna()].tail(10)
        
        logger.info(f"\nLatest 10 predictions:")
        for idx, row in latest_results.iterrows():
            prediction = "📈 UP" if row['Predictions'] == 1 else "📉 DOWN"
            logger.info(f"  {idx.date()} | Close: ${row['Close']:.2f} | Prediction: {prediction}")
        
        # Prediction distribution
        pred_counts = results['Predictions'].value_counts()
        logger.info(f"\nOverall prediction distribution:")
        for pred, count in pred_counts.items():
            direction = "UP" if pred == 1 else "DOWN"
            percentage = (count / len(results['Predictions'].dropna())) * 100
            logger.info(f"  {direction}: {count} ({percentage:.1f}%)")
        
        # Step 4: Save results
        if save_results:
            logger.info("\n💾 STEP 4: Saving results...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"predictions_{timestamp}.csv"
            results.to_csv(output_file)
            logger.info(f"✅ Results saved to: {output_file}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


def predict_from_csv(csv_path: str, save_results: bool = True) -> pd.DataFrame:
    """
    Generate predictions from a CSV file (for testing or batch processing)
    
    Args:
        csv_path: Path to CSV file with OHLCV data
        save_results: Whether to save predictions to CSV
    
    Returns:
        DataFrame with predictions
    """
    try:
        logger.info(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        
        # Ensure Target column exists
        if 'Target' not in df.columns:
            logger.warning("Target column not found. Creating synthetic target...")
            df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
            df = df.iloc[:-1]  # Remove last row with NaN target
        
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Run inference
        results = make_predictions(df)
        
        # Save results
        if save_results:
            output_file = csv_path.replace('.csv', '_predictions.csv')
            results.to_csv(output_file)
            logger.info(f"✅ Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process CSV: {e}")
        raise


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description='S&P 500 Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch latest data and predict
  python main.py
  
  # Force refresh data (ignore cache)
  python main.py --no-cache
  
  # Predict from CSV file
  python main.py --csv my_data.csv
  
  # Don't save results to file
  python main.py --no-save
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='Path to CSV file with OHLCV data (optional)',
        default=None
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force refresh data from Yahoo Finance (ignore cache)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save predictions to CSV file'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        help=f'Stock ticker symbol (default: {config.TICKER_SYMBOL})',
        default=config.TICKER_SYMBOL
    )
    
    parser.add_argument(
        '--period',
        type=str,
        help=f'Data period to fetch (default: {config.DATA_PERIOD})',
        default=config.DATA_PERIOD
    )
    
    args = parser.parse_args()
    
    try:
        if args.csv:
            # Predict from CSV file
            results = predict_from_csv(args.csv, save_results=not args.no_save)
        else:
            # Override config if custom ticker/period provided
            if args.ticker != config.TICKER_SYMBOL or args.period != config.DATA_PERIOD:
                logger.info(f"Using custom ticker: {args.ticker}, period: {args.period}")
                fetcher = DataFetcher(ticker=args.ticker)
                raw_data = fetcher.fetch_data(period=args.period, use_cache=not args.no_cache)
                df = fetcher.prepare_data_for_inference(raw_data)
                results = make_predictions(df)
                
                if not args.no_save:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"predictions_{args.ticker}_{timestamp}.csv"
                    results.to_csv(output_file)
                    logger.info(f"✅ Results saved to: {output_file}")
            else:
                # Standard prediction with latest data
                results = predict_latest(
                    use_cache=not args.no_cache,
                    save_results=not args.no_save
                )
        
        return results
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()