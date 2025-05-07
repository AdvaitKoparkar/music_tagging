import os
import pandas as pd
from tqdm import tqdm
from audio_separator.separator import Separator

def get_data_to_convert(directory_path, save_path):
    audio_sep_data = {
        'original': [],
        'vocals': [],
        'instrumental': [],
        'status': [],
    }
    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.mp3'):
                file_path = os.path.join(root, filename)
                file_save_path = os.path.dirname(file_path).replace(directory_path, save_path)
                os.makedirs(file_save_path, exist_ok=True)

                audio_sep_data['original'].append(file_path)
                audio_sep_data['vocals'].append(file_save_path+f'/{filename.replace(".mp3", "_vocals.mp3")}')
                audio_sep_data['instrumental'].append(file_save_path+f'/{filename.replace(".mp3", "_instrumental.mp3")}')
                audio_sep_data['status'].append(False)
    return audio_sep_data

if __name__ == '__main__':
    # data = pd.DataFrame(get_data_to_convert('./1', './1_sep_mel_band_roformer_sep'))
    # data.to_csv('./1_sep_mel_band_roformer_sep/checkpoint.csv')

    separator = Separator()
    separator.load_model()
    data = pd.read_csv('./1_sep_mel_band_roformer_sep/checkpoint.csv')
    for idx in tqdm(range(data.shape[0]), total=data.shape[0]):
        if data.iloc[idx]['status']:
            continue
            
        separator.separate(data.iloc[idx]['original'], {'Vocals': data.iloc[idx]['vocals'], 'Instrumental': data.iloc[idx]['instrumental'],})
        data['status'].iloc[idx] = True
        data.to_csv('./1_sep_mel_band_roformer_sep/checkpoint.csv')
