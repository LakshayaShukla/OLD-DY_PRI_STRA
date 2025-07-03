from fastapi import FastAPI, HTTPException, Query, Body, Depends
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, List, ClassVar
from datetime import datetime
import pandas as pd
import numpy as np
import uvicorn
from model import PricingModel
import logging
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Dynamic Pricing API",
    description="API for getting optimal dynamic pricing recommendations",
    version="1.0.0"
)

# Initialize model
try:
    model = PricingModel()
    model.load_model()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load model: {e}")
    model = None

# Input models
class PricingRequest(BaseModel):
    demand_score: float = Field(..., ge=0.0, le=1.0, description="Demand score (0.0-1.0)")
    inventory_level: float = Field(..., ge=0.0, le=1.0, description="Inventory level as percentage of max")
    competitor_price: float = Field(..., gt=0.0, description="Competitor's price")
    user_type: str = Field(..., description="User type (new, loyal, or premium)")
    timestamp: Optional[str] = Field(None, description="Timestamp (ISO format)")
    season: Optional[str] = Field(None, description="Season (winter, spring, summer, fall)")
    base_price: Optional[float] = Field(None, gt=0.0, description="Base price of the product")
    product_id: Optional[str] = Field(None, description="Product ID")
    
    @field_validator('user_type')
    def validate_user_type(cls, v):
        if v.lower() not in ['new', 'loyal', 'premium']:
            raise ValueError('user_type must be one of: new, loyal, premium')
        return v.lower()
    
    @field_validator('season')
    def validate_season(cls, v):
        if v is not None and v.lower() not in ['winter', 'spring', 'summer', 'fall']:
            raise ValueError('season must be one of: winter, spring, summer, fall')
        return v.lower() if v is not None else v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "demand_score": 0.75,
                "inventory_level": 0.5,
                "competitor_price": 199.99,
                "user_type": "loyal",
                "timestamp": "2023-07-15T12:00:00",
                "season": "summer",
                "base_price": 189.99,
                "product_id": "P001"
            }
        }}

class PricingResponse(BaseModel):
    recommended_price: float
    explanation: Dict[str, Any]
    timestamp: str
    request_id: str

# Helper functions
def get_features_from_request(request: PricingRequest) -> dict:
    """Convert API request to feature dictionary for model"""
    
    # Parse timestamp or use current time
    if request.timestamp:
        try:
            ts = pd.to_datetime(request.timestamp)
        except:
            ts = pd.Timestamp.now()
    else:
        ts = pd.Timestamp.now()
    
    # Create features dictionary
    features = {
        'demand_score': request.demand_score,
        'inventory_level': request.inventory_level,
        'competitor_price': request.competitor_price,
        'hour': ts.hour,
        'day': ts.day,
        'month': ts.month,
        'day_of_week_num': ts.dayofweek,
        'user_type': request.user_type,
        'base_price': request.base_price if request.base_price is not None else request.competitor_price
    }
    
    # Add season if provided, otherwise infer from month
    if request.season:
        features['season'] = request.season
    else:
        month = ts.month
        if month in [12, 1, 2]:
            features['season'] = 'winter'
        elif month in [3, 4, 5]:
            features['season'] = 'spring'
        elif month in [6, 7, 8]:
            features['season'] = 'summer'
        else:
            features['season'] = 'fall'
    
    return features

def generate_price_explanation(features: dict, price: float) -> dict:
    """Generate explanation for the recommended price"""
    
    # Calculate competitor price comparison
    comp_price = features['competitor_price']
    price_diff = price - comp_price
    price_diff_pct = (price_diff / comp_price) * 100
    
    # Create explanation
    explanation = {
        "price_comparison": {
            "competitor_price": comp_price,
            "price_difference": price_diff,
            "price_difference_percent": price_diff_pct,
            "is_higher": price > comp_price
        },
        "factors": {
            "demand": features['demand_score'],
            "inventory": features['inventory_level'],
            "user_type": features['user_type'],
            "season": features['season'],
            "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][features['day_of_week_num']]
        }
    }
    
    # Add factors explanations
    factor_explanations = []
    
    if features['demand_score'] > 0.7:
        factor_explanations.append("High demand suggests higher pricing")
    elif features['demand_score'] < 0.3:
        factor_explanations.append("Low demand suggests lower pricing")
    
    if features['inventory_level'] < 0.3:
        factor_explanations.append("Low inventory suggests higher pricing")
    elif features['inventory_level'] > 0.8:
        factor_explanations.append("High inventory suggests lower pricing")
    
    if features['user_type'] == 'premium':
        factor_explanations.append("Premium users tend to be less price sensitive")
    elif features['user_type'] == 'loyal':
        factor_explanations.append("Loyal users may expect some price benefits")
    
    if features['season'] == 'summer':
        factor_explanations.append("Summer season typically has higher demand")
    elif features['season'] == 'winter':
        factor_explanations.append("Winter season may have lower demand")
    
    explanation["factor_explanations"] = factor_explanations
    
    return explanation

def fallback_pricing_model(features: dict) -> float:
    """Simple fallback pricing model if ML model is unavailable"""
    
    # Start with base price or competitor price
    base_price = features.get('base_price', features['competitor_price'])
    
    # Apply demand factor (higher demand = higher price)
    demand_factor = 1.0 + (features['demand_score'] - 0.5) * 0.3
    
    # Apply inventory factor (lower inventory = higher price)
    inventory_factor = 1.0 + (0.5 - features['inventory_level']) * 0.2
    
    # Apply user type factor
    user_factors = {
        'new': 0.95,      # Discount for new users
        'loyal': 1.0,     # Standard price for loyal users
        'premium': 1.05   # Premium for premium users
    }
    user_factor = user_factors.get(features['user_type'], 1.0)
    
    # Apply seasonal factor
    season_factors = {
        'winter': 0.9,
        'spring': 1.0,
        'summer': 1.1,
        'fall': 1.0
    }
    season_factor = season_factors.get(features['season'], 1.0)
    
    # Calculate final price
    price = base_price * demand_factor * inventory_factor * user_factor * season_factor
    
    # Ensure price is within reasonable bounds of competitor price
    min_price = features['competitor_price'] * 0.8
    max_price = features['competitor_price'] * 1.2
    price = max(min_price, min(price, max_price))
    
    return price

# Routes
@app.get("/")
def read_root():
    return {"message": "Dynamic Pricing API", "docs": "/docs"}

@app.get("/status")
def get_status():
    """Get API and model status"""
    return {
        "status": "online",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/recommend_price", response_model=PricingResponse)
def recommend_price(request: PricingRequest):
    """Get price recommendation based on provided features"""
    
    try:
        # Convert request to features
        features = get_features_from_request(request)
        
        # Get prediction from model or fallback
        if model is not None:
            try:
                # Use ML model for prediction
                price = model.predict(pd.DataFrame([features]))
            except Exception as e:
                # If ML model fails, use fallback
                logger.warning(f"Model prediction failed: {e}. Using fallback.")
                price = fallback_pricing_model(features)
        else:
            # Use fallback if no model loaded
            logger.info("No model loaded. Using fallback pricing model.")
            price = fallback_pricing_model(features)
        
        # Generate explanation
        explanation = generate_price_explanation(features, price)
        
        # Create response
        response = PricingResponse(
            recommended_price=float(price),
            explanation=explanation,
            timestamp=datetime.now().isoformat(),
            request_id=f"req-{pd.Timestamp.now().timestamp():.0f}"
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_recommend")
def batch_recommend_prices(requests: List[PricingRequest]):
    """Get price recommendations for multiple products"""
    
    results = []
    
    for request in requests:
        try:
            # Get recommendation for single request
            result = recommend_price(request)
            results.append({
                "product_id": request.product_id,
                "recommendation": result
            })
        except Exception as e:
            # If one request fails, include error but continue processing
            results.append({
                "product_id": request.product_id,
                "error": str(e)
            })
    
    return {"results": results, "count": len(results)}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 