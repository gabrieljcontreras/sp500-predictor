"""
FastAPI wrapper for S&P 500 prediction system
Exposes prediction functionality as REST API endpoints
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import pandas as pd
import logging

from data_fetcher import get_latest_data, DataFetcher
from inference import make_predictions, InferencePipeline
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="S&P 500 Prediction API",
    description="Machine learning predictions for S&P 500 stock movements",
    version="1.0.0"
)

# Enable CORS (allows frontend from different domain to access API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (loaded once at startup)
pipeline = None


# Response models
class PredictionResponse(BaseModel):
    """Single prediction response"""
    date: str
    close_price: float
    prediction: int = Field(..., description="0 for DOWN, 1 for UP")
    prediction_label: str = Field(..., description="UP or DOWN")


class HistoricalPrediction(BaseModel):
    """Historical prediction with actual data"""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    prediction: Optional[int]
    prediction_label: Optional[str]


class BatchPredictionResponse(BaseModel):
    """Batch predictions response"""
    total_predictions: int
    date_range: Dict[str, str]
    predictions: List[HistoricalPrediction]
    summary: Dict[str, int]


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None
    timestamp: str


# Startup event - load model once
@app.on_event("startup")
async def startup_event():
    """Load model and initialize pipeline at startup"""
    global pipeline
    try:
        logger.info("Loading model and initializing pipeline...")
        pipeline = InferencePipeline()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# Health check endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        model_loaded=pipeline is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy" if pipeline is not None else "unhealthy",
        model_loaded=pipeline is not None,
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


# Prediction endpoints
@app.get("/predict/latest", response_model=PredictionResponse)
async def predict_latest(use_cache: bool = True):
    """
    Get the latest prediction for S&P 500
    
    - **use_cache**: Whether to use cached data (default: true)
    
    Returns the most recent prediction with current market data
    """
    try:
        logger.info(f"Fetching latest prediction (use_cache={use_cache})")
        
        # Fetch latest data
        df = get_latest_data(use_cache=use_cache)
        
        # Make predictions
        results = pipeline.run(df)
        
        # Get the most recent prediction
        latest = results[results['Predictions'].notna()].tail(1)
        
        if latest.empty:
            raise HTTPException(
                status_code=404,
                detail="No valid predictions available. Insufficient historical data."
            )
        
        # Format response
        pred_value = int(latest['Predictions'].iloc[0])
        response = PredictionResponse(
            date=latest.index[0].isoformat(),
            close_price=float(latest['Close'].iloc[0]),
            prediction=pred_value,
            prediction_label="UP" if pred_value == 1 else "DOWN"
        )
        
        logger.info(f"✅ Latest prediction: {response.prediction_label}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_latest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    use_cache: bool = True,
    limit: int = 30,
    ticker: Optional[str] = None,
    period: Optional[str] = None
):
    """
    Get batch predictions for multiple days
    
    - **use_cache**: Whether to use cached data (default: true)
    - **limit**: Number of recent predictions to return (default: 30, max: 365)
    - **ticker**: Stock ticker symbol (default: ^GSPC for S&P 500)
    - **period**: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)
    
    Returns predictions for the specified number of recent trading days
    """
    try:
        # Validate limit
        limit = min(max(1, limit), 365)
        
        logger.info(f"Batch prediction request: limit={limit}, ticker={ticker}, period={period}")
        
        # Fetch data
        if ticker or period:
            fetcher = DataFetcher(ticker=ticker or config.TICKER_SYMBOL)
            raw_data = fetcher.fetch_data(
                period=period or config.DATA_PERIOD,
                use_cache=use_cache
            )
            df = fetcher.prepare_data_for_inference(raw_data)
        else:
            df = get_latest_data(use_cache=use_cache)
        
        # Make predictions
        results = pipeline.run(df)
        
        # Get recent predictions
        recent = results[results['Predictions'].notna()].tail(limit)
        
        if recent.empty:
            raise HTTPException(
                status_code=404,
                detail="No valid predictions available"
            )
        
        # Format predictions
        predictions_list = []
        for idx, row in recent.iterrows():
            pred_value = int(row['Predictions']) if pd.notna(row['Predictions']) else None
            predictions_list.append(
                HistoricalPrediction(
                    date=idx.isoformat(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    prediction=pred_value,
                    prediction_label="UP" if pred_value == 1 else "DOWN" if pred_value == 0 else None
                )
            )
        
        # Calculate summary
        pred_counts = recent['Predictions'].value_counts()
        summary = {
            "up": int(pred_counts.get(1, 0)),
            "down": int(pred_counts.get(0, 0))
        }
        
        response = BatchPredictionResponse(
            total_predictions=len(predictions_list),
            date_range={
                "start": recent.index.min().isoformat(),
                "end": recent.index.max().isoformat()
            },
            predictions=predictions_list,
            summary=summary
        )
        
        logger.info(f"✅ Returning {len(predictions_list)} predictions")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh-data")
async def refresh_data(background_tasks: BackgroundTasks):
    """
    Force refresh market data from Yahoo Finance
    
    This endpoint triggers a data refresh in the background
    """
    try:
        logger.info("Triggering data refresh...")
        
        # Run data fetch in background
        def fetch_fresh_data():
            try:
                get_latest_data(use_cache=False)
                logger.info("✅ Data refresh completed")
            except Exception as e:
                logger.error(f"❌ Data refresh failed: {e}")
        
        background_tasks.add_task(fetch_fresh_data)
        
        return {
            "status": "refresh_initiated",
            "message": "Data refresh started in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initiating refresh: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model
    """
    try:
        feature_names = pipeline.model_loader.get_feature_names()
        
        return {
            "model_loaded": pipeline is not None,
            "model_type": str(type(pipeline.model_loader.get_model()).__name__),
            "features_count": len(feature_names) if feature_names else None,
            "feature_names": feature_names,
            "horizons": config.HORIZONS,
            "ticker": config.TICKER_SYMBOL
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )