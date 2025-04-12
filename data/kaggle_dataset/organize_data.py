import re
import os
import shutil
from tqdm import tqdm

def get_raga_from_filename(filename):
    base_name = os.path.splitext(filename)[0]
    match = re.match(r'([a-zA-Z]+)(\d+)$', base_name)
    if match:
        return match.group(1).lower()
    return None

def organize_files_by_raga(data_path):
    for file in tqdm(os.listdir(data_path)):
        if not file.endswith('.wav'):
            continue
            
        raga = get_raga_from_filename(file)
        if not raga:
            continue
            
        raga_dir = os.path.join(data_path, raga)
        os.makedirs(raga_dir, exist_ok=True)
        
        # Extract the ID from the filename
        base_name = os.path.splitext(file)[0]
        match = re.match(r'([a-zA-Z]+)(\d+)$', base_name)
        if match:
            id_num = int(match.group(2))
            new_filename = f"{raga}_{id_num:04d}.wav"
            
            # Copy and rename the file
            src_path = os.path.join(data_path, file)
            dst_path = os.path.join(raga_dir, new_filename)
            shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    data_version = "2"
    data_path = f"{data_version}"
    
    # Organize files into raga folders
    organize_files_by_raga(data_path)
    
    