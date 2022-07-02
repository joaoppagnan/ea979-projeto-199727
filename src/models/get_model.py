import os
from src.utils.download_url import download_url
from src.utils.unzip_file import unzip_file

def get_model(path:str, fname:str, url:str):
    os.makedirs(path, exist_ok=True)
    download_url(url, fname=fname)
    unzip_file(filename=fname, destination=path)
    os.unlink(fname)