from pydantic import BaseModel
from typing import List, Optional

class ForecastRequest(BaseModel):
    horizon: int
    model:   str = "ARIMA"  # or "XGB"

class ForecastResponse(BaseModel):
    horizon: int
    mean:  List[float]
    lower: List[float]
    upper: List[float]
    xgb:    Optional[List[float]] = None  # recent XGB predictions
    mape:   Optional[float]       = None