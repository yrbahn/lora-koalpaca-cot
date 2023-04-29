import json
import copy
import itertools
import torch
import transformers
from tqdm import tqdm

from torch.utils.data import Dataset
from dataclasses import dataclass
from prompt_template import PROMPT_DICT

IGNORE_INDEX = -100

def load_dataset(data_path):
    with open(data_path, 'r') as json_file:
        dataset = [json.loads(line) for line in json_file]
    return dataset

class SupervisedDataset(Dataset):
    def __init__(self, data, tokenizer, seq_length):
        self.dataset = data
        self.tokenizer = tokenizer
        self.seq_length = seq_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        example = self.dataset[i]
        if example.get("input", "") != "":
            prompt = PROMPT_DICT["prompt_input"].format_map(example)
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format_map(example)

        len_prompt_tokens = len(self.tokenizer(prompt, truncation=True, 
            max_length=self.seq_length+1)["input_ids"]) -1
        full_tokens = self.tokenizer(
            prompt + example['output'],
            truncation=True,
            max_length = self.seq_length + 1,
            padding="max_length"
        )["input_ids"][:-1]

        return {
            "input_ids": full_tokens,
            "labels" :[-100] * len_prompt_tokens + full_tokens[len_prompt_tokens:],
            "attention_mask": [1] * (len(full_tokens)),
        }

def create_comparison_dataset(path, split="train"):
    dataset = load_dataset(path)
    pairs = []
    for sample in tqdm(dataset):
        pair = {}
        prompt = sample["prompt"]
        chosen_summary = sample["chosen"]
        rejected_summary = sample["rejected"]
        if chosen_summary == rejected_summary:
            continue
        if len(chosen_summary.split()) < 5 or len(rejected_summary.split()) < 5:
            continue
        pair["chosen"] = prompt + chosen_summary
        pair["rejected"] = prompt + rejected_summary
        pairs.append(pair)
    return pairs


class PairwiseDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_length):
        self.chosen_input_ids = []
        self.chosen_attn_masks = []
        self.rejected_input_ids = []
        self.rejected_attn_masks = []
        for pair in tqdm(pairs):
            chosen, rejected = pair["chosen"], pair["rejected"]
            chosen_encodings_dict = tokenizer(
                "<|startoftext|>" + chosen + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            rejected_encodings_dict = tokenizer(
                "<|startoftext|>" + rejected + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )
            self.chosen_input_ids.append(chosen_encodings_dict["input_ids"])
            self.chosen_attn_masks.append(chosen_encodings_dict["attention_mask"])
            self.rejected_input_ids.append(rejected_encodings_dict["input_ids"])
            self.rejected_attn_masks.append(rejected_encodings_dict["attention_mask"])

    def __len__(self):
        return len(self.chosen_input_ids)

    def __getitem__(self, idx):
        return (
            self.chosen_input_ids[idx],
            self.chosen_attn_masks[idx],
            self.rejected_input_ids[idx],
            self.rejected_attn_masks[idx],
        )

