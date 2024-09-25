from transformers import AutoTokenizer
import context_managers
from typing import List, Dict, Tuple
from transformers.tokenization_utils_base import BatchEncoding
import torch
import random


def encode_to_list(tokenizer: AutoTokenizer, text: str) -> List[int]:
    return tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]


def make_prompt(tokenizer: AutoTokenizer, input_text: str) -> List[int]:
    """
    get prompt: [id, id, id, xxx]
    """
    prompt_text = "将日文译为中文"
    # bos_token_id = tokenizer.get_command("<bos>")
    # system_token_id = tokenizer.get_command("<|system|>")
    # user_token_id = tokenizer.get_command("<|user|>")
    # assistant_token_id = tokenizer.get_command("<|assistant|>")
    bos_token_id = tokenizer.pad_token_id
    system_token_id = tokenizer.encode("<|file_sep|>", add_special_tokens=False)[0]
    user_token_id = tokenizer.encode("<|fim_middle|>", add_special_tokens=False)[0]
    assistant_token_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]

    prompt_text_encoded = encode_to_list(tokenizer, prompt_text)
    input_text_encoded = encode_to_list(tokenizer, input_text)

    return [bos_token_id, system_token_id] + prompt_text_encoded + [user_token_id] + input_text_encoded + [assistant_token_id]


def get_context_index(length: int, radius: int) -> List[Tuple[int, int]]:
    results = []
    for i in range(length):
        start = max(i - radius, 0)
        end = min(i + radius, length)
        results.append((start, end))
    return results


def get_context_with_input(contents: Dict[str, str], context_radius: int):
    jps: List[str] = list(map(lambda x: x['jp'], contents))
    contexts_list: List[List[str]] = [jps[start: end] for start, end in get_context_index(len(jps), context_radius)]
    return list(zip(contexts_list, jps))


def get_context_with_input_and_answer(contents: Dict[str, str], context_radius: int):
    jps: List[str] = list(map(lambda x: x['jp'], contents))
    cns: List[str] = list(map(lambda x: x['cn'], contents))
    contexts_list: List[List[str]] = [jps[start: end] for start, end in get_context_index(len(jps), context_radius)]
    return list(zip(contexts_list, jps, cns))


def make_prompt_with_context(tokenizer: AutoTokenizer, context_texts: List[str], input_text: str) -> List[int]:
    bos_token_id = tokenizer.pad_token_id
    system_token_id = tokenizer.encode("<|file_sep|>", add_special_tokens=False)[0]
    user_token_id = tokenizer.encode("<|fim_middle|>", add_special_tokens=False)[0]
    assistant_token_id = tokenizer.encode("<|im_start|>", add_special_tokens=False)[0]

    context_texts = " |".join(context_texts)
    context_encoded = encode_to_list(tokenizer, context_texts)

    prompt_text = "将日文译为中文"
    prompt_text_encoded = encode_to_list(tokenizer, prompt_text)
    input_text_encoded = encode_to_list(tokenizer, input_text)

    return [bos_token_id] + context_encoded \
        + [system_token_id] + prompt_text_encoded \
            + [user_token_id] + input_text_encoded \
                + [assistant_token_id]

def make_answer(tokenizer, answer_text) -> List[int]:
    eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    answer_encoded = encode_to_list(tokenizer, answer_text)
    return answer_encoded + [eos_token_id]

def make_prompt_with_context_and_answer(
        tokenizer: AutoTokenizer, context_texts: List[str],
        input_text: str, answer_text) -> List[int]:
    prompt = make_prompt_with_context(tokenizer, context_texts, input_text)
    eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    answer_encoded = encode_to_list(tokenizer, answer_text)
    return prompt + answer_encoded + [eos_token_id]

def make_contextual_prompt_answer_and_mask_length(tokenizer: AutoTokenizer, context_texts: str, jp: str, cn: str):
    prompt = make_prompt_with_context(tokenizer, context_texts, jp)
    answer = make_answer(tokenizer, cn)
    full_seq = prompt + answer
    return full_seq, len(answer)

def make_batch_and_prompt_mask(tokenizer: AutoTokenizer, prompts: List[List[int]], mask_lengths: List[int]):
    pad_id = tokenizer.pad_token_id
    max_len = max(list(map(len, prompts)))
    input_ids = torch.zeros((len(prompts), max_len), dtype=torch.int64)
    input_ids += pad_id
    for i, prompt in enumerate(prompts):
        length = len(prompt)
        input_tensor = torch.tensor(prompt, dtype=torch.int64)
        input_ids[i][max_len-length:] = input_tensor
    attention_mask = torch.where(input_ids != pad_id, 1, 0)
    
    batch = BatchEncoding(data={
        'input_ids': input_ids, 'attention_mask': attention_mask, 
    })

    prompt_mask = torch.zeros((len(prompts), max_len), dtype=torch.int64)
    for i, length in enumerate(mask_lengths):
        prompt_mask[i][max_len-length:] = 1

    return batch, prompt_mask

# -------------------------------------------------------------

def make_prompt_with_answer(tokenizer: AutoTokenizer, input_text: str, answer_text: str) -> List[int]:
    prompt = make_prompt(tokenizer, input_text)
    eos_token_id = tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]
    answer_encoded = encode_to_list(tokenizer, answer_text)
    return prompt + answer_encoded + [eos_token_id]

def make_proper_batch(tokenizer: AutoTokenizer, prompts: List[List[int]], create_labels: bool = False):
    pad_id = tokenizer.pad_token_id
    max_len = max(list(map(len, prompts)))
    input_ids = torch.zeros((len(prompts), max_len), dtype=torch.int64)
    input_ids += pad_id
    for i, prompt in enumerate(prompts):
        length = len(prompt)
        input_tensor = torch.tensor(prompt, dtype=torch.int64)
        input_ids[i][max_len-length:] = input_tensor
    attention_mask = torch.where(input_ids != pad_id, 1, 0)
    #position_ids = torch.cumsum(attention_mask, dim=-1)
    
    batch = BatchEncoding(data={
        'input_ids': input_ids, 'attention_mask': attention_mask, #'position_ids': position_ids
    })
    if create_labels:
        batch.data["labels"] = input_ids

    return batch


def get_batches_for_lora_training(tokenizer: AutoTokenizer, context: context_managers.Context, batch_size: int):
    contents = context.get_contents()
    random.shuffle(contents)
    batches = []
    for i in range(0, len(contents), batch_size):
        batch_pairs = contents[i: i+batch_size]
        prompts = [make_prompt_with_answer(tokenizer, pair["jp"], pair["cn"]) for pair in batch_pairs]
        batch = make_proper_batch(tokenizer, prompts)
        batches.append(batch)
    return batches


if __name__ == "__main__":
    model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side='left')
    out = make_prompt(tokenizer, "jp test")
    print(out)
    print(tokenizer.decode(out, skip_special_tokens=False))

    out = make_prompt_with_answer(tokenizer, "日文", "汉语")
    print(out)
    print(tokenizer.decode(out, skip_special_tokens=False))

    print(tokenizer.pad_token_id)

    prompts = [
        make_prompt_with_answer(tokenizer, "日文1", "汉语1"),
        make_prompt_with_answer(tokenizer, "日文12", "汉语12"),
        make_prompt_with_answer(tokenizer, "日文123", "汉语123")
    ]
    batch = make_proper_batch(tokenizer, prompts)
    print(batch["input_ids"])
    print(batch["attention_mask"])
    print(batch["position_ids"])

    context = context_managers.Context(["/home/leo/NLP/datasets/translate-jp-cn/datasets/subtitles/双语字幕/虫师 续章[全20集] Mushishi Zoku Shou (2014) 简繁日双语字幕 - 诸神字幕组/output/[Kamigami] Mushishi Zoku Shou - 01 [1920x1080 x264 AAC Sub(Chs,Cht,Jap)].简日.jsonl"])
    batches = get_batches_for_lora_training(tokenizer, context, 4)
    print(len(batches))
    input_ids = batches[3]['input_ids']
    for i in range(len(input_ids)):
        print(tokenizer.decode(input_ids[i]))