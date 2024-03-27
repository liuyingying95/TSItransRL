#!/usr/bin/env python
# coding=utf-8
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import pandas as pd


def condition_convert(x):
    if x["drd2"] >= 0.5 and x["qed"] >= 0.6 and x["sa"] <= 4.0:
        return "positive"
    else:
        return "negative"


def condition_convert_jnk_gsk3(x):
    if x["jnk3"] >= 0.5 and x["gsk3"] >= 0.5 and x["qed"] >= 0.6 and x["sa"] <= 4.0:
        return "positive"
    else:
        return "negative"


class myDataset(Dataset):

    def __init__(self, data, tokenizer):
        title, seqs = [], []
        for k, v in data.items():
            title.append(v[1])
            seqs.append(v[0])

        self.tokenizer = tokenizer
        self.title = title
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        inputs = self.tokenizer.bos_token + self.title[i] + \
                self.tokenizer.sep_token + self.seqs[i] + \
                self.tokenizer.eos_token
        encoding_dict = self.tokenizer(inputs,
                                       truncation=True,
                                       max_length=256,
                                       padding="max_length")
        input_ids = encoding_dict["input_ids"]

        return torch.tensor(input_ids)

    @classmethod
    def collate_fn(cls, arr):
        max_length = max([seq.size(0) for seq in arr])
        collated_arr = torch.zeros(len(arr), max_length)
        for i, seq in enumerate(arr):
            collated_arr[i, :seq.size(0)] = seq
        return collated_arr


def get_dict(csv_path):
    df = pd.read_csv(csv_path)
    #df.loc[:, "title"] = df.apply(condition_convert, axis=1)
    df.loc[:, "title"] = df.apply(condition_convert_jnk_gsk3, axis=1)
    data = dict()
    id = 1
    for row in df.itertuples():
        smiles = getattr(row, "SMILES")
        title = getattr(row, "title")
        data[id] = [smiles, title]
        id += 1
    return data


class Distilldata(Dataset):

    def __init__(self, data, tokenizer):
        seqs = []
        for k, v in data.items():
            seqs.append(v[0])
            self.tokenizer = tokenizer 
            self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, i):
        input = self.tokenizer.bos_token + \
                self.seqs[i] + self.tokenizer.eos_token
        encodings_dict = self.tokenizer(input, truncation=True, max_length=256, 
                                        padding="max_length")
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("./liuyingv3", max_length=256)
    df = pd.read_csv("./drd2data/drd_train_new2.csv")
    #df.loc[:, "title"] = df.apply(condition_convert, axis=1)
    df.loc[:, "title"] = df.apply(condition_convert_jnk_gsk3, axis=1)
    data = dict()
    id = 1
    for row in df.itertuples():
        smiles = getattr(row, "SMILES")
        title = getattr(row, "title")
        data[id] = [smiles, title]

        id += 1

    dataset = myDataset(data, tokenizer)
    import ipdb;ipdb.set_trace()
