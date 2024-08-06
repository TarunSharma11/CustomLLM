import os
import requests

hellaswag_data_url = 'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl'

def download(url, file):
    req = requests.get(url, stream = True)
    with open(file, 'wb') as f:
        for chunk in req.iter_content(chunk_size = 16):
            f.write(chunk)
            
def get_hellaswag_val_data(hellaswag_data_location):
    if not os.path.isfile(hellaswag_data_location):
        download(hellaswag_data_url, hellaswag_data_location)

    with open(hellaswag_data_location, 'r') as f:
        hellaswag_val_data = f.readlines()
    return hellaswag_val_data