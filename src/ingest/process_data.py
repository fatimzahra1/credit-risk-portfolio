import os
import pandas as pd
from src.features.make_features import generate_features

def main():
    os.makedirs('data/processed', exist_ok=True)

    # 1) Portfolio-level default rate
    clients = pd.read_csv('data/raw/credit_default_clients.csv')
    # Simulate a monthly timestamp:
    clients['month_idx'] = (clients.index // 1250) + 1
    portfolio = (
        clients.groupby('month_idx')
               .default_flag.mean()
               .reset_index()
               .rename(columns={'month_idx':'timestamp','default_flag':'default_rate'})
    )
    portfolio.to_csv('data/processed/portfolio_df.csv', index=False)
    print(f"Wrote {len(portfolio)} rows to data/processed/portfolio_df.csv")

    # 2) Macro series
    macro = pd.read_csv('data/raw/fred_dgs10.csv', parse_dates=['date'], index_col='date')
    macro.to_csv('data/processed/macro_df.csv')
    print(f"Wrote {len(macro)} rows to data/processed/macro_df.csv")

if __name__ == "__main__":
    main()
