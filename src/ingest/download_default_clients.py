import os
import pandas as pd

def download_default_clients(
    url: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    download_path: str = "./data/raw/credit_default_clients.csv"
):
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    # read Excel (skip the first header row)
    df = pd.read_excel(url, header=1)
    # optional: rename TARGET to default_flag
    df = df.rename(columns={"default payment next month": "default_flag"})
    df.to_csv(download_path, index=False)
    print(f"Wrote {len(df)} rows to {download_path}")

if __name__ == "__main__":
    download_default_clients()
