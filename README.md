# TTS project

## Installation guide

First of all you need to download the repository: 

```shell
git clone https://github.com/retir/TTS
cd TTS
git checout master
```
Then install necessary dependencies:

```shell
pip install -r requirements.txt
pip install zip
bash load.sh
```

To synthez voice read sentances in `test.txt` line by line and run

```shell
python3 test.py -c tts/configs/fastspeech1.json -pth pretrained_models/mfa_model.pth --texts_pth test.txt
```

where `-pth` means model checkpoint location, `-c` config of usable model and `--texts_pth` is a path to inference sentences. It is also available to add flags as `-d`, `-p` and `-e` with means duration, pitch and energy control respectively. For example `-d 2.0` means that generated audio will be twice facter then it should be.

Pretrained model from `downloader.py` was learnd with config `fastspeech1.json` (`tts/configs/fastspeech1.json`) with MFA. To train model use follow:

```shell
python3 train.py -c path/to/config.json
```
The main part of work (such as models, losses, datasets and etc.) is in `tts\` directory.
## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
