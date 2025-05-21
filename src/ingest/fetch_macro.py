from dotenv import load_dotenv
import os
import requests
import pandas as pd

load_dotenv()

def fetch_fred_series(series_id: str = 'DGS10', start_date: str = '2000-01-01', end_date: str = None):
    api_key = os.getenv('FRED_API_KEY')
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': start_date
    }
    if end_date:
        params['observation_end'] = end_date
    r = requests.get(url, params=params)
    data = r.json().get('observations', [])
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df.set_index('date')

if __name__ == "__main__":
    df = fetch_fred_series()
    df.to_csv('./data/raw/fred_dgs10.csv')