import translation_utils
import context_managers
from ptuning_trainer import PtuningTrainer, PtuningConfigs
import ptuning_model
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
        #prompt = translation_utils.make_prompt(tokenizer, pair['jp'])
        prompt = translation_utils.make_prompt_with_context(tokenizer, context, jp)
        model_inputs = translation_utils.make_proper_batch(tokenizer, [prompt], create_labels=False).to('cuda')
        #model_inputs = wrapped_model.prepare_inputs_for_generation(**model_inputs)
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

# GLM3
# NO PTUNING ~ 0.209
# 0.208
# 0.207
# 0.214
# 0.213
# 0.206

# WITH PTUNING ~ 0.218
# 0.219
# 0.226
# 0.215
# 0.216
# 0.217


# Qwen2

# No Ptuning 0.227
# scores: [0.22955582577081107, 0.22905574217120678, 0.2319524777928255, 0.22200349184881993, 0.2241340125953616]
# , mean: 0.22734031003580496

# Ptuning:
# test complete! scores: [0.2154786973102854, 0.2105084849234588, 0.22284781627240555, 0.2293982529580425, 0.21524629049165916],
# mean: 0.21869590839117029

# Context
# --------------------------------------
#avg score: 0.238
# without ptuning
#test complete! scores: [0.2455124181679597, 0.25666824675898764, 0.24010735971118727, 0.25084386565313727, 0.23845139383396086],
# mean: 0.24631665682504655


# ptuning-loss-3.0
# test complete! scores: [0.20830857240376083, 0.2083954551115074, 0.20874045248834194, 0.1956899183191136, 0.21098421549277765],
# mean: 0.20642372276310028

# ptuning-loss-2.5
# test complete! scores: [0.21894797985751135, 0.21774796504543434, 0.2189628479177296, 0.21158281752753683, 0.2146601374662604]
# , mean: 0.2163803495628945

# ptuning-loss-2.0
# test complete! scores: [0.19707280026539883, 0.19521469654756796, 0.1967848166690209, 0.20286797057553627, 0.213098905752612],
# mean: 0.2010078379620272
