import torch
import numpy as np

from torch.utils.data import DataLoader
from tts.utils.parse_config import ConfigParser
from tts.utils import get_data_to_buffer
from tts.utils import reprocess_tensor
from tts.datasets.buffer_dataset import BufferDataset


def collate_fn_builder(config):
    def collate_fn_tensor(batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // config['batch_expand_size']

        cut_list = list()
        for i in range(config['batch_expand_size']):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(config['batch_expand_size']):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
    return collate_fn_tensor


def get_dataloader(config: ConfigParser,):
    buffer = get_data_to_buffer(config)

    dataset = BufferDataset(buffer)

    training_loader = DataLoader(
        dataset,
        batch_size=config["batch_expand_size"] * config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn_builder(config),
        drop_last=True,
        num_workers=0
    )
    return training_loader