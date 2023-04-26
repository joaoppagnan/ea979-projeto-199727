import zipfile
from tqdm import tqdm

def unzip_file(fname: str, destination: str):
    with zipfile.ZipFile(fname, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            zf.extract(member, destination)
    pass