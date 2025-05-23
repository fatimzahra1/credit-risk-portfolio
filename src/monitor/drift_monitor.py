import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from src.models.utils import load_model
from src.features.make_features import generate_features


def monitor_drift(horizon: int = 6):
    # Load processed series
    df = pd.read_csv('data/processed/portfolio_df.csv', parse_dates=['timestamp'], index_col='timestamp')

    # ARIMA drift
    arima = load_model('arima')
    hist = df['default_rate']
    y_true_arima = hist[-horizon:]
    y_pred_arima = arima.fittedvalues[-horizon:]
    mape_arima = mean_absolute_percentage_error(y_true_arima, y_pred_arima)

    # XGB drift
    xgb = load_model('xgb')
    macro_df = pd.read_csv('data/processed/macro_df.csv', parse_dates=['date'], index_col='date')
    features = generate_features(df.reset_index(), macro_df)
    X_all = features.drop(columns=['default_rate'])
    y_true_xgb = features['default_rate'][-horizon:]
    y_pred_xgb = xgb.predict(X_all)[-horizon:]
    mape_xgb = mean_absolute_percentage_error(y_true_xgb, y_pred_xgb)

    report = {'arima_mape': mape_arima, 'xgb_mape': mape_xgb}
    print(report)
    # TODO: add alert if drift exceeds threshold

if __name__ == '__main__':
    monitor_drift()