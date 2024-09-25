import translation_utils
import context_managers
from ptuning_trainer import PtuningTrainer, PtuningConfigs
from transformers.generation import GenerationConfig
import torch
import json
import tqdm
import model_utils

from wrappers.qwen2 import Qwen2Wrapper

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file_to_translate",
                        help='json文件，为一个列表，每个元素至少包含一个"jp"键（可以设置），结果会写在"pred"键中，保存时将保存其它原始内容，方便你写脚本。',
                        type=str)
    parser.add_argument("--model_name", help="基础模型名称，可以指定你本地的Qwen/Qwen2.5-3B目录", type=str,
                        default="Qwen/Qwen2.5-3B")
    parser.add_argument("--lora_path", help="LoRA adaptor路径，跟目录下自带了一个我微调出来的", type=str,
                        default="./lora_with_ptuning_epoch_7")
    parser.add_argument("--out_file_name", help="输出文件", type=str,
                        default="./translate_out.json")
    parser.add_argument("--context_radius", help="上下文大小，比如为2则前面截2条后面截2条", type=int,
                        default=2)
    parser.add_argument("--input_key", help="目标内容的键名，默认为jp，你可以自己设置", type=str,
                        default="jp")
    parser.add_argument("--longest_seq", help="上下文大小，比如为2则前面截2条后面截2条", type=int,
                        default=40)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    ptuning_configs = PtuningConfigs()
    ptuning_trainer = PtuningTrainer()
    torch.cuda.set_per_process_memory_fraction(0.9, device=0)

    model_path = args.model_name

    target_path = args.json_file_to_translate
    with open(target_path, "r", encoding="utf8") as file:
        content = json.load(file)
    content = [{"jp": data[args.input_key][:args.longest_seq]} for data in content]
    context = context_managers.Context([], content)

    wrapped_model, tokenizer = Qwen2Wrapper.from_model_dir(
        model_path, ptuning_configs.pre_seq_len, True, ptuning_configs.device
    )
    original_model = wrapped_model.wrapper_original_model

    # Train embedding
    wrapped_model.create_and_set_new_embedding()
    model_utils.set_requires_grad(wrapped_model.wrapper_original_model, False)
    ptuning_trainer.train_context(wrapped_model, tokenizer, ptuning_configs, context)
    model_utils.set_requires_grad(wrapped_model.wrapper_original_model, True)

    original_model.load_adapter(args.lora_path)
    wrapped_model.set_ptuning_status(True)
    wrapped_model.current_model = original_model
    model_utils.set_requires_grad(wrapped_model.prefix_encoder, False)
    model_utils.set_requires_grad(wrapped_model, False)

    gen_config = GenerationConfig(
        max_length=200,
        do_sample=True,
        top_p=0.95,
        no_repeat_ngram_size=2,
        num_beams=8,
        use_cache=True
    )

    # Load Target File
    with open(args.json_file_to_translate, 'r', encoding="utf8") as file:
        file_contents = json.load(file)
    
    target_key = args.input_key

    for i, data in tqdm.tqdm(list(enumerate(file_contents))):
        jp = data[target_key][:args.longest_seq]
        start = max(0, i - args.context_radius)
        end = min(len(file_contents), i + args.context_radius)
        context = list(map(lambda x: x[target_key][:args.longest_seq], file_contents[start:end]))
        prompt = translation_utils.make_prompt_with_context(tokenizer, context, jp)
        model_inputs = translation_utils.make_proper_batch(tokenizer, [prompt], create_labels=False).to('cuda')
        input_tokens_count = len(model_inputs["input_ids"][0])

        model_out = wrapped_model.generate(
            **model_inputs, generation_config=gen_config,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
            pad_token_id=tokenizer.pad_token_id
            )
        
        prediction_tokens = model_out[0][input_tokens_count:-1]
        prediction_string = tokenizer.decode(prediction_tokens)
        data["pred"] = prediction_string
        
    print("done!")
    with open(args.out_file_name, "w", encoding="utf8") as file:
        json.dump(file_contents, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()