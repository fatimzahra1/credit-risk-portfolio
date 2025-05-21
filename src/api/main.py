from fastapi import FastAPI
from src.api.schemas import ForecastRequest, ForecastResponse
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape
import os
from src.models.utils import load_model
from sklearn.metrics import mean_absolute_percentage_error


app = FastAPI(title="Credit Risk Forecast API")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/forecast", response_model=ForecastResponse)
async def forecast(req: ForecastRequest):
    model = load_model('arima')
    data_path = os.getenv('PORTFOLIO_CSV', 'data/processed/portfolio_df.csv')
    
    df = pd.read_csv(data_path, parse_dates=['timestamp'], index_col='timestamp')


    if req.model.upper() == "XGB":
        # only XGB
        xgb = load_model('xgb')
        from src.features.make_features import generate_features
        hist = df.reset_index().rename(columns={'timestamp':'timestamp','default_rate':'default_rate'})
        features_df = generate_features(
            hist,
            macro_df=pd.read_csv(
                'data/processed/macro_df.csv', parse_dates=['date'], index_col='date'
            )
        )
        X_all = features_df.drop(columns=['default_rate'])
        preds = xgb.predict(X_all)[-req.horizon:].tolist()
        y_true = features_df["default_rate"][-req.horizon:]
        error  = mape(y_true, preds)
        return ForecastResponse(horizon=req.horizon, mean=[], lower=[], upper=[], xgb=preds, mape=error)
    else:
        # default to ARIMA only
        arima = load_model('arima')
        pred  = arima.get_forecast(steps=req.horizon)
        mean  = pred.predicted_mean.tolist()
        ci    = pred.conf_int()
         # compute MAPE over the *last* req.horizon fitted points
        y_true   = df["default_rate"].iloc[-req.horizon:]
        y_fitted = arima.fittedvalues.iloc[-req.horizon:]
        error    = mean_absolute_percentage_error(y_true, y_fitted)
        return ForecastResponse(
            horizon=req.horizon,
            mean=mean,
            lower=ci.iloc[:,0].tolist(),
            upper=ci.iloc[:,1].tolist(),
            xgb=None,
            mape=error
        )

@app.post("/update-model")
async def update_model():
    from src.models.train import main as train_main
    train_main()
    return {"status": "model updated"}