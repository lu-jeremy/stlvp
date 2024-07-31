import os
import requests
import sys
import os
from bs4 import BeautifulSoup
import tqdm
import shutil

# shutil.rmtree('./process_data')
# shutil.rmtree('./vint_train')
#
# sys.exit()

base_url = 'https://rail.eecs.berkeley.edu/datasets/huron/'
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all directories
directories = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('/')]

print(directories)

def download_file(file_url, save_path):
    response = requests.get(file_url, stream=True)
    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def download_files_from_directory(directory_url, save_path):
    response = requests.get(directory_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].endswith('.bag')]

    for file in files:
        file_url = directory_url + file
        file_path = os.path.join(save_path, file)
        print(f'Downloading {file_url} to {file_path}')
        download_file(file_url, file_path)

# Create a folder to save the dataset
os.makedirs('sacson', exist_ok=True)

for directory in tqdm.tqdm(directories):
    dir_url = base_url + directory
    dir_path = os.path.join('./sacson', directory)
    os.makedirs(dir_path, exist_ok=True)
    download_files_from_directory(dir_url, dir_path)
