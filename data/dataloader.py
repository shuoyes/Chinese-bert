import os
# import jieba
import torch
from torch.utils.data import DataLoader
from config import DefaultConfig

config = DefaultConfig()


class MyDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, num_workers, config, tokenizer_class):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.collate,
            shuffle=shuffle,
            num_workers=num_workers,
        )
        self.tokenizer = tokenizer_class.from_pretrained(config.BERT_PATH)
        self.max_length = config.max_sentence_length

    def collate(self, batch_data):
        tokens, label_ids = map(list, zip(*batch_data))
        text_ids = self.tokenizer(tokens,
                                  padding=True,
                                  truncation=True,
                                  max_length=self.max_length,
                                  is_split_into_words=True,
                                  add_special_tokens=True,
                                  return_tensors='pt')
        positions = torch.zeros_like(text_ids['input_ids'])
        positions[:, :] = torch.arange(0, text_ids['input_ids'].size(1))
        text_ids['position_ids'] = positions

        return text_ids, label_ids
