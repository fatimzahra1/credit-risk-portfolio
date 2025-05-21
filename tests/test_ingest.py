import pytest
import os
from src.ingest.download_kaggle import download_home_credit_data

def test_download_creates_files(tmp_path, monkeypatch):
    class DummyApi:
        def authenticate(self): pass
        def dataset_download_file(self, dataset, file_name, path, unzip):
            open(os.path.join(path, file_name), 'w').close()

    monkeypatch.setattr('kaggle.api.kaggle_api_extended.KaggleApi', lambda: DummyApi())
    download_home_credit_data(download_path=str(tmp_path))
    assert (tmp_path / 'application_train.csv').exists()