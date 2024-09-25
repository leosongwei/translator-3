import translation_utils
import context_managers
from ptuning_trainer import PtuningTrainer, PtuningConfigs
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from transformers.generation import GenerationConfig
import torch
import statistics
import json
import tqdm
import model_utils

from wrappers.qwen2 import Qwen2Wrapper


PTUNING = True


ptuning_configs = PtuningConfigs()
#ptuning_configs.loss_limit = 3.5
#ptuning_configs.context_max_epoch = 50
ptuning_trainer = PtuningTrainer()
torch.cuda.set_per_process_memory_fraction(0.9, device=0)

model_path = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-3B"
wrapped_model, tokenizer = Qwen2Wrapper.from_model_dir(
    model_path, ptuning_configs.pre_seq_len, PTUNING, ptuning_configs.device
)

target_path = "/home/leo/NLP/datasets/translate-jp-cn/datasets/subtitles/双语字幕/宫崎骏_诸神字幕组/字幕/output/天空之城 Laputa：Castle in the Sky (1986) BluRay 简日双语@诸神字幕组.jsonl"
context = context_managers.Context([target_path])
contents = context.get_contents() # [-200:]


original_model = wrapped_model.wrapper_original_model
if PTUNING:
    # train embedding
    wrapped_model.create_and_set_new_embedding()
    context_with_embedding = ptuning_trainer.train_context(wrapped_model, tokenizer, ptuning_configs, context)
    # load adapter
    original_model.load_adapter("./lora_result/lora_with_ptuning_epoch_7")
    wrapped_model.set_ptuning_status(True)
    model_utils.set_requires_grad(wrapped_model.prefix_encoder.prefix_embedding, False)
else:
    original_model.load_adapter("./lora_out_without_ptuning")
    wrapped_model.set_ptuning_status(False)

wrapped_model.current_model = original_model
#print(wrapped_model.current_model)


def compute_sentence_bleu(ground_truth: str, prediction: str):
    # actual = list(jieba.cut(ground_truth))
    # pred = list(jieba.cut(prediction))

    actual = list(ground_truth)
    pred = list(prediction)

    result = sentence_bleu(
                    [actual],
                    pred,
                    smoothing_function=SmoothingFunction().method3,
                )
    return result


gen_config = GenerationConfig(
    max_length=200,
    do_sample=True,
    top_p=0.95,
    no_repeat_ngram_size=2,
    # repetition_penalty=1.2,
    # length_penalty=1.5,
    num_beams=8,
    use_cache=True
)


context_jp_cn_list = translation_utils.get_context_with_input_and_answer(contents, 2)


results = []

for i in range(1):
    translate_results = []

    total_score = 0.0
    for context, jp, cn in tqdm.tqdm(context_jp_cn_list):
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
        
        score = compute_sentence_bleu(cn, prediction_string)
        print({"jp": jp, "cn": cn}, "pred:", prediction_string, "score:", score)
        total_score += score

        translate_results.append({"jp": jp, "cn": cn, "pred": prediction_string})

    with open("translate_result.jsonl", "w", encoding="utf8") as file:
        for line in translate_results:
            json.dump(line, file, ensure_ascii=False)
            print("", file=file)
        
    result = total_score / len(contents)
    results.append(result)
    print(f"avg score: {result:.3f}")
print(f"test complete! scores: {results}, mean: {statistics.mean(results)}")
