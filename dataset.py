import torch
import numpy as np

class DataLoaderLite:
    def __init__(self, B, T, rank, world_size, num_shards, split = 'train'):
        self.B = B
        self.T = T
        self.rank = rank
        self.world_size = world_size
        self.shard_idx = 1
        self.num_shards = num_shards
        self.file = f"./data/fineweb_10BT/shard_{split}"
        self.enc_text = self.load_shard(self.shard_idx)
        self.curr_start = self.calculate_initial_start()

    def calculate_initial_start(self):
        return self.B * self.T * self.rank

    def load_shard(self, shard_idx):
        return np.load(self.file + str(shard_idx) + ".npy").astype(np.uint16)
    
    def next_batch(self):
        end_index = self.curr_start + self.B * self.T
        data = torch.tensor(self.enc_text[self.curr_start: end_index], dtype = torch.long).view(self.B, self.T)
        labels = torch.tensor(self.enc_text[self.curr_start + 1: end_index + 1], dtype = torch.long).view(self.B,self.T)
        self.curr_start = self.curr_start + self.B * self.T * self.world_size
        if ((self.curr_start + self.B * self.T * self.world_size + 1) > len(self.enc_text)):
            self.shard_idx = (self.shard_idx + 1) % self.num_shards
            self.enc_text = self.load_shard(self.shard_idx)
            self.curr_start = (self.B * self.T) * self.rank
        return data, labels

    def reset(self):
        self.curr_start = self.calculate_initial_start()
                