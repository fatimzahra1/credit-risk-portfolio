import os
import pandas as pd

def fetch_macro_data(
    url: str = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
    download_path: str = "data/raw/fred_dgs10.csv"
):
    """
    Downloads the 10-year Treasury yield as a CSV directly from FRED’s graph endpoint,
    which gives you two columns: DATE and DGS10. We rename them to lowercase and save.
    """
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    # This CSV has header: DATE,DGS10
    df = pd.read_csv(url, header=0)
    # Normalize column names
    df.rename(columns={df.columns[0]: "date", df.columns[1]: "value"}, inplace=True)
    df.to_csv(download_path, index=False)
    print(f"Wrote {len(df)} rows to {download_path}")

if __name__ == "__main__":
    fetch_macro_data()
