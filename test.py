import os
import argparse
import collections
import warnings
import utils

import numpy as np
import torch
import waveglow

import tts.model as module_arch
import tts.loss as module_loss
import tts.logger as module_loggers
from tqdm import tqdm
from tts.trainer import Trainer
from tts.utils.object_loading import get_dataloader
from tts.utils import prepare_device
from tts.utils.parse_config import ConfigParser
from tts.utils import get_data, preprocess_english

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def synthesis(model, text, device, dur_alpha=1.0, pitch_alpha=1.0, energy_alpha=1.0):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        batch = {"text": sequence, "src_pos": src_pos}
        mel = model.forward(batch, dur_alpha=dur_alpha, pitch_alpha=pitch_alpha, energy_alpha=energy_alpha)
    return mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)
    

def main(config, args):
    # prepare data
    raw_texts = []
    with open(args.texts_pth, 'r') as f:
        raw_texts = f.readlines()
    raw_texts = [text[:-2] for text in raw_texts]
        
    if config["data"]["use_mfa"]:
        val_text = preprocess_english(raw_texts, config["data"]["lexicon_path"], config['data']['text_cleaners'])
    else:      
        val_text = get_data(raw_texts, config['data']['text_cleaners'])

        
    # prepare model
    model = config.init_obj(config["arch"], module_arch)
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        
    checkpoint = torch.load(args.checkpoint_pth, device)
    assert checkpoint["config"]["arch"] == config["arch"]
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.eval()
    
    WaveGlow = utils.get_WaveGlow().to(device)
    
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)
    
    for i, phn in tqdm(enumerate(val_text)):
        mel, mel_cuda = synthesis(model, phn, device, args.duration, args.pitch, args.energy)

        waveglow.inference.inference(
            mel_cuda, WaveGlow,
            f"{save_dir}/text_{i}_d={args.duration}_p={args.pitch}_e={args.energy}.wav"
        )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        type=str,
        help="path to model config",
    )
    args.add_argument(
        "-pth",
        "--checkpoint_pth",
        type=str,
        help="path to model checkpoint",
    )
    args.add_argument(
        "-t",
        "--texts_pth",
        type=str,
        help="path to inference texts",
    )
    args.add_argument(
        "-o",
        "--results_dir",
        default="./results",
        type=str,
        help="path to results dir",
    )
    args.add_argument(
        "-d",
        "--duration",
        default=1.0,
        type=float,
        help="duration of generated video",
    )
    args.add_argument(
        "-e",
        "--energy",
        default=1.0,
        type=float,
        help="energy of generated video",
    )
    args.add_argument(
        "-p",
        "--pitch",
        default=1.0,
        type=float,
        help="pitch of generated video",
    )
    args.add_argument(
        "-device",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-z",
        "--resume",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    options = []
    config = ConfigParser.from_args(args, options)
    args = args.parse_args()
    main(config, args)