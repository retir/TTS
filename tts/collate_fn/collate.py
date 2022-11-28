import numpy as np
from tts.utils import reprocess_tensor


def collate_fn_builder(config):
    def collate_fn_tensor(batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // config.batch_expand_size

        cut_list = list()
        for i in range(config.batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(config.batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
    return collate_fn_tensor