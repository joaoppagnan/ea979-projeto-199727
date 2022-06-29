import zipfile
import tqdm

def unzip_file(filename: str, destination: str):
    with zipfile.ZipFile(filename, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='Extracting '):
            zf.extract(member, destination)