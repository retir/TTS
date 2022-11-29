import gdown
import argparse
import os

FILE_IDS = {
    "train.txt": "1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx",
    "waveglow_256channels_ljs_v2.pt": "1WsibBTsuRg_SF2Z6L6NFRTT-NjEy1oTx",
    "mel.tar.gz": "1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j",
    "alignments.zip": "1E32jmwdPenQIUcxhI4awBzbvrUD98Ksa",
    "energies.zip": "1SJt6T0weTb_yx5N3KYFtnmkOBpJajkVu",
    "LJSpeech.zip": "1qRnpmBWsY1x7V993O8BWBCmAadQTco1W",
    "pitches.zip": "1UiG1zfLLTmLkqWmb5Fo16vjGnljgnObO"
}

PRETRAINED_IDS = {
    "mfa_model.pth": "1vDI-9lck4FzsSfnT9a_bdsrkVH8GpsVE"
}


if __name__ == "__main__":
    models_dir = './pretrained_models/'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    
    for file_name, id in FILE_IDS.items():
        print(f'Loading {file_name}...')
        url = 'https://drive.google.com/uc?id=' + id
        gdown.download(url, file_name, quiet=True)
        
    for file_name, id in PRETRAINED_IDS.items():
        print(f'Loading {file_name}...')
        url = 'https://drive.google.com/uc?id=' + id
        gdown.download(url, models_dir + file_name, quiet=True)
    
    print('Done')