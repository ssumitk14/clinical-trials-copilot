import os
import json

from app.extraction_service.load_data_config import DATA_PATH, FILE_NAME, EXTRACTED_DATA_PATH
from app.extraction_service.data_extractor import extract_trial_info

class Utility:
    def load_raw_data():
        data = []
        with open(os.path.join(DATA_PATH, FILE_NAME)) as f:
            data = json.load(f)
        
        return data

    def extract_data(raw_data):
        extracted_data = []
        for data in raw_data:
            extracted_data.append(extract_trial_info(data))
        
        return extracted_data

    def save_json(data, path=EXTRACTED_DATA_PATH):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    
    def get_all_nct_ids(data):
        nct_ids = [trial['nctId'] for trial in data if 'nctId' in trial]
        return nct_ids
    

    