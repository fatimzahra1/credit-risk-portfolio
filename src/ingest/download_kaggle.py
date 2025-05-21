import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_home_credit_data(dataset: str = 'home-credit-default-risk', download_path: str = '../data/raw'):
    api = KaggleApi()
    api.authenticate()
    os.makedirs(download_path, exist_ok=True)
    files = [
        'application_train.csv', 'application_test.csv', 'bureau.csv',
        'bureau_balance.csv', 'installments_payments.csv', 'credit_card_balance.csv',
        'previous_application.csv', 'POS_CASH_balance.csv'
    ]
    for fname in files:
        # 1) download the .zip around the single file
        api.dataset_download_file(dataset, file_name=fname, path=download_path, force=False, quiet=False)

        # 2) unzip it (Kaggle wraps each file in its own zip)
        zip_path = os.path.join(download_path, fname + '.zip')
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(download_path)
            os.remove(zip_path)

if __name__ == "__main__":
    download_home_credit_data()