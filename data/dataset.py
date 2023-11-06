import os
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TrainDataset(Dataset):

    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.folders = os.listdir(root_dir)
        self.tokenizer = BertTokenizer.from_pretrained(config.BERT_PATH)
        self.labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=torch.long)
        self.labelMap = {'01': 0, '02': 1, '03': 2, '04': 3, '05': 4, '06': 5, '07': 6, '08': 7, '09': 8, '10': 9,'11': 10, '12': 11}

    def __getitem__(self, index):
        fileName = self.folders[index]
        label = self.labels[self.labelMap[fileName[0:2]]]
        path = os.path.join(self.root_dir, fileName)
        with open(path, 'r') as f:
            # 该数据一行就是一句
            content = [line.rstrip('\n') for line in f]
            dum_idx = ["".join(content)]


        return dum_idx, label

    def __len__(self):
        return len(os.listdir(self.root_dir))
