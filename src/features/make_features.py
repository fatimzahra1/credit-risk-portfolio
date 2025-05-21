
import pandas as pd

def create_lag_features(df: pd.DataFrame, target_col: str, max_lag: int) -> pd.DataFrame:
    """
    For each lag from 1 to max_lag, add a column:
      {target_col}_lag_{lag} = df[target_col].shift(lag)
    """
    for lag in range(1, max_lag + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    """
    For each window in windows, add:
      {target_col}_roll_mean_{window}
      {target_col}_roll_std_{window}
    computed over a rolling window of that many periods.
    """
    for window in windows:
        df[f'{target_col}_roll_mean_{window}'] = df[target_col].rolling(window).mean()
        df[f'{target_col}_roll_std_{window}']  = df[target_col].rolling(window).std()
    return df


def generate_features(
    portfolio_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    max_lag: int = 12,
    rolling_windows: list[int] = [3, 6, 12]
) -> pd.DataFrame:
    """
    - Joins portfolio-level default_rate (with a 'timestamp' column) to macro_df (indexed by date).
    - Fills forward/backward any missing macro values.
    - Generates lag features up to `max_lag`.
    - Generates rolling-mean and rolling-std features over the specified windows.
    - Drops the first few rows that lack complete feature history.
    """
    df = portfolio_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.set_index('timestamp', inplace=True)

    # Join and impute macro data
    df = df.join(macro_df, how='left')
    df.ffill(inplace=True)  # forward-fill
    df.bfill(inplace=True)  # back-fill
    df.interpolate(method='linear', inplace=True)
    df.reset_index(inplace=True)

    # Lag and rolling features
    df = create_lag_features(df, 'default_rate', max_lag)
    df = create_rolling_features(df, 'default_rate', rolling_windows)

    # Determine how many rows to drop: only the longest lookback window
    drop_n = max(max_lag, max(rolling_windows))
    return df.iloc[drop_n:].reset_index(drop=True).drop(columns=['timestamp'])

