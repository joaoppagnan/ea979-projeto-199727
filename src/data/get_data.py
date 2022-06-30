import os
from src.utils import download_url, unzip_file

def get_data(path:str, fname:str, url:str):
    os.makedirs(path, exist_ok=True)
    download_url(path, fname=fname)
    unzip_file(filename=fname, destination=path)