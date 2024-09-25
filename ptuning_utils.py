import os
import json
from typing import List, Dict
import statistics
import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding


"""
The file trains ptunings for all project listed in the data_registry.json.
"""


def create_project_indexes(data_registry_json_path: str):
    """
    读取"./data_registry.json"，该文件记录一组project文件夹，形如:

    ```
    ["dir1", "dir2", "dir3", ...]
    ```

    每个project文件夹，包含一系列jsonl文件（比如每个文件为一集动画片的字幕）。
    
    每个jsonl文件形如（每一行为一个完整json）:

    ```
    {"jp": "日文", "cn": "中文翻译"}
    {"jp": "日文", "cn": "中文翻译"}
    {"jp": "日文", "cn": "中文翻译"}
    ```

    返回：[[project_files, ...], other porjects...]
    """
    projects: List[List[str]] = []

    with open(data_registry_json_path, encoding="utf8") as file:
        train_dirs = json.load(file)

    for dir in train_dirs:
        project_files = []
        for dir, _, files in os.walk(dir):
            for file in files:
                project_files.append(os.path.join(dir, file))
        projects.append(sorted(project_files))

    return projects


def create_cont_seq(input_ids_list, split_seq):
    vstack_list = []
    for seq in input_ids_list:
        vstack_list.append(seq)
        vstack_list.append(split_seq)
    cont_seq = torch.cat(vstack_list, dim=-1)
    return cont_seq


def create_continuous_seqs(input_ids_list, split_pattern, continuous_seq_length_limit: int, offset:int):
    collections: List[List] = []
    current_token_count = 0
    current_collection = []
    for input_ids in input_ids_list[offset:]:
        current_token_count += len(input_ids)
        current_collection.append(input_ids)

        if current_token_count > continuous_seq_length_limit:
            collections.append(current_collection)
            current_collection = []
            current_token_count = 0
    
    if current_collection and current_token_count > continuous_seq_length_limit / 3:
        # give up when remaining is too short
        collections.append(current_collection)

    seq_nums = list(map(len, collections))
    cont_seqs: List[torch.Tensor] = []
    for seqs in collections:
        cont_seqs.append(create_cont_seq(seqs, split_pattern))
    if seq_nums:
        return cont_seqs, statistics.mean(seq_nums)
    else:
        return cont_seqs, 0.0


def create_ptuning_batches(dataset_input_ids, batch_size, tokenizer: AutoTokenizer):
    raw_batches = [dataset_input_ids[i:i+batch_size] for i in range(0, len(dataset_input_ids), batch_size)]
    batches: List[BatchEncoding] = []
    for batch in raw_batches:
        max_len = max(list(map(len, batch)))
        input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        input_ids += tokenizer.pad_token_id
        for i in range(len(batch)):
            seq = batch[i]
            input_ids[i][max_len - len(seq):] = seq
        attention_mask = torch.where(input_ids != tokenizer.pad_token_id, 1, 0)
        #position_ids = torch.cumsum(attention_mask, dim=-1)
        batch = BatchEncoding(data={
            'input_ids': input_ids, 'attention_mask': attention_mask, # 'position_ids': position_ids
        })
        batches.append(batch)
    
    return batches



if __name__ == "__main__":
    data_registry_path = os.path.join(os.path.split(__file__)[0], "./data_registry.json")
    