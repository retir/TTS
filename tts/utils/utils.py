import os
import re
import tgt
import time
import json
import torch
import text
import numpy as np
import torch.nn.functional as F

from text import text_to_sequence
from pathlib import Path
from collections import OrderedDict
from scipy.interpolate import interp1d
from string import punctuation
from g2p_en import G2p
from tqdm import tqdm

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def get_data(tests, text_cleaners):
    data_list = list(text.text_to_sequence(test, text_cleaners) for test in tests)
    
    return data_list

def pad_1D(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = np.pad(x, (0, length - x.shape[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_1D_tensor(inputs, PAD=0):

    def pad_data(x, length, PAD):
        x_padded = F.pad(x, (0, length - x.shape[0]))
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = torch.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):

    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(x, (0, max_len - np.shape(x)[0]),
                          mode='constant',
                          constant_values=PAD)
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_2D_tensor(inputs, maxlen=None):

    def pad(x, max_len):
        if x.size(0) > max_len:
            raise ValueError("not max_len")

        s = x.size(1)
        x_padded = F.pad(x, (0, 0, 0, max_len-x.size(0)))
        return x_padded[:, :s]

    if maxlen:
        output = torch.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(x.size(0) for x in inputs)
        output = torch.stack([pad(x, max_len) for x in inputs])

    return output


def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def get_data_to_buffer(train_config):
    buffer = list()
    text = process_text(train_config["data_path"])
    if train_config['use_mfa']:
        mfa_pathes = [train_config["mfa_path"] + f'/{path}' for path in sorted(os.listdir(train_config["mfa_path"]))]

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
    #for i in tqdm(range(3000)):

        # get mel spec
        mel_gt_name = os.path.join(
            train_config["mel_ground_truth"], "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        
        
        # get text and duration
        if train_config['use_mfa']:
            textgrid = tgt.io.read_textgrid(mfa_pathes[i])
            phone, duration, start, end = get_alignment(
                textgrid.get_tier_by_name("phones"), train_config['sample_rate'], train_config['hope_length']
            )
            phones = "{" + " ".join(phone) + "}"
            character = np.array(text_to_sequence(phones, train_config['text_cleaners']))
            duration = np.array(duration)
            ds = sum(duration)
            assert len(duration) == len(character)
            mel_gt_target = mel_gt_target[:ds]
            assert mel_gt_target.shape[0] == ds
        else:
            character = text[i][0:len(text[i])-1]
            character = np.array(
                text_to_sequence(character, train_config["text_cleaners"]))
            
            duration = np.load(os.path.join(
            train_config["alignment_path"], str(i)+".npy"))
        
        # get pitch
        pitch = np.load(os.path.join(
            train_config["pitch_path"], "pitch_" + str(i)+ ".npy"))
        pitch = pitch[:sum(duration)]
        
        # get energy
        energy = np.load(os.path.join(
            train_config["energy_path"], "energy_" + str(i)+ ".npy"))
        energy = energy[:sum(duration)]

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        mel_gt_target = torch.from_numpy(mel_gt_target)
        
#         #START PITCH
#         pitch = torch.from_numpy(pitch)
#         pitch_mask = np.where(pitch != 0)[0]
#         pitch_interpolate = interp1d(
#             pitch_mask,
#             pitch[pitch_mask],
#             fill_value=(pitch[pitch_mask[0]], pitch[pitch_mask[-1]]),
#             bounds_error=False)
#         pitch = pitch_interpolate(np.arange(len(pitch)))
#         window_start = 0
#         avarage_pitches = []
#         for phonem_len in duration:
#             if phonem_len > 0:
#                 avarage_pitches.append(np.mean(pitch[window_start:window_start + phonem_len]))
#             else:
#                 avarage_pitches.append(0)
#             window_start += phonem_len
#         pitch = avarage_pitches[:len(duration)]
        
#         #START ENERGY
        
#         window_start = 0
#         avarage_energies = []
#         for phonem_len in duration:
#             if phonem_len > 0:
#                 avarage_energies.append(np.mean(energy[window_start:window_start + phonem_len]))
#             else:
#                 avarage_energies.append(0)
#             window_start += phonem_len
#         energy = avarage_energies[:len(duration)]

        pitch = torch.tensor(pitch)
        energy = torch.tensor(energy)
        
        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target, "pitch": pitch,
                       "energy": energy})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer


def reprocess_tensor(batch, cut_list):
    texts = [batch[ind]["text"] for ind in cut_list]
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    energy = [batch[ind]["energy"] for ind in cut_list]

    length_text = np.array([])
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))

    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    texts = pad_1D_tensor(texts)
    durations = pad_1D_tensor(durations)
    mel_targets = pad_2D_tensor(mel_targets)
    pitches = pad_1D_tensor(pitches)
    energy = pad_1D_tensor(energy)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration": durations,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len,
           "pitches": pitches,
           "energy": energy}

    return out

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
        
        
def get_alignment(tier, sr, hop_length):
    sil_phones = ["sil", "sp", "spn"]

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        if phones == []:
            start_time = s
        phones.append(p)
        end_time = e

        durations.append(
            int(
                np.round(e * sr / hop_length)
                - np.round(s * sr / hop_length)
            )
        )

    return phones, durations, start_time, end_time



def preprocess_english(texts, lexicon_pth, text_cleaners):
    texts_preprocess = []
    for text in texts:
        text = text.rstrip(punctuation)
        lexicon = read_lexicon(lexicon_pth)

        g2p = G2p()
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in lexicon:
                phones += lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", g2p(w)))
        phones = "{" + "}{".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")

        sequence = np.array(
            text_to_sequence(
                phones, text_cleaners
            )
        )
        texts_preprocess.append(np.array(sequence))

    return texts_preprocess

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon