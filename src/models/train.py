import pandas as pd
import mlflow
import mlflow.sklearn
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from src.models.utils import save_model
from src.features.make_features import generate_features

def train_arima(portfolio_df: pd.DataFrame):
    model = SARIMAX(portfolio_df['default_rate'], order=(1,1,1), seasonal_order=(1,1,1,12))
    result = model.fit(disp=False)
    save_model(result, 'arima')
    return result

def train_xgb(features_df: pd.DataFrame):
    from xgboost import XGBRegressor
    X = features_df.drop(['default_rate'], axis=1)
    y = features_df['default_rate']
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)
    save_model(model, 'xgb')
    return model, X, y

def main():
    portfolio_df = pd.read_csv('data/processed/portfolio_df.csv', parse_dates=['timestamp'])
    macro_df     = pd.read_csv('data/processed/macro_df.csv', parse_dates=['date'], index_col='date')
    features_df  = generate_features(portfolio_df, macro_df)

    mlflow.set_experiment('credit-risk')
    with mlflow.start_run():
        arima_model = train_arima(portfolio_df)
        y_true_arima   = portfolio_df['default_rate']
        y_fitted_arima = arima_model.fittedvalues
        arima_mape_value = mean_absolute_percentage_error(y_true_arima, y_fitted_arima)
        mlflow.log_metric('arima_mape', arima_mape_value)
        mlflow.sklearn.log_model(arima_model, 'arima-model')

        xgb_model, X, y = train_xgb(features_df)
        y_pred_xgb = xgb_model.predict(X)
        xgb_mape_value = mean_absolute_percentage_error(y, y_pred_xgb)
        mlflow.log_metric('xgb_mape', xgb_mape_value)
        mlflow.sklearn.log_model(xgb_model, 'xgb-model')

        print(f"ARIMA MAPE: {arima_mape_value:.4f}")
        print(f"XGB MAPE:   {xgb_mape_value:.4f}")

if __name__ == "__main__":
    main()