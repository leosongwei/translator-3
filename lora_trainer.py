from peft import LoraConfig, get_peft_model
import context_managers
import translation_utils
import json
import safetensors.torch
import torch
from typing import List, Tuple
import random
import bitsandbytes
import statistics
import tqdm
import os
from torch.nn import CrossEntropyLoss
from transformers.generation import GenerationConfig

from wrappers.qwen2 import Qwen2Wrapper
import model_utils


class LoraTrainingConfig:
    enable_ptuning = True
    lr = 1e-4
    r = 4
    lora_dropout = 0.1
    device = 'cuda'
    pre_seq_len = 10  # if you train ptuning embeddings, this must align with your ptuning settings.
    epochs = 8
    batch_size = 2
    clip_grad_max_norm = 2.0
    grad_accum_step = 8
    context_radius = 2
    log_per_steps = 40
    generate_test_per_steps = 200


def get_lora_model(config: LoraTrainingConfig, model, targets):
    lora_conf = LoraConfig(
        r=config.r,
        target_modules=targets,
        lora_dropout=config.lora_dropout,
        bias='none',
    )
    lora_model = get_peft_model(model, lora_conf)
    return lora_model


def get_all_context_with_embeddings_from_record():
    records_path = "./ptuning_results/ptuning_training_record.jsonl"
    with open(records_path, "r", encoding="utf8") as file:
        training_records = list(map(json.loads, file.readlines()))
    results: List[context_managers.ContextWithEmbedding] = []
    for record in training_records:
        embedding_path = os.path.join("./ptuning_results/embeddings", record["embedding_path"])
        embedding_weight = safetensors.torch.load_file(embedding_path)['weight']
        num_embeddings, embedding_size = embedding_weight.shape
        embedding = torch.nn.Embedding(
            num_embeddings, embedding_size, dtype=embedding_weight.dtype, _weight=embedding_weight
        )
        embedding.weight.requires_grad_(False)
        context = context_managers.Context(record["files"])
        context_with_embedding = context_managers.ContextWithEmbedding(
            context, embedding
        )
        results.append(context_with_embedding)
    return results


def clone_embedding(embedding: torch.nn.Embedding) -> torch.nn.Embedding:
    n_vocab = embedding.num_embeddings
    dim = embedding.embedding_dim
    return torch.nn.Embedding(n_vocab, dim, dtype=embedding.weight.dtype, _weight=embedding.weight.clone())


def test_generate(tokenizer, wrapper_model):
    gen_config = GenerationConfig(
        max_new_tokens=50,
        do_sample=True,
        top_p=0.97,
        no_repeat_ngram_size=2,
        num_beams=2,
        use_cache=True
    )

    def generate_and_print(context, jp, cn):
        prompt = translation_utils.make_prompt_with_context(tokenizer, context, jp)
        model_inputs = translation_utils.make_proper_batch(tokenizer, [prompt], create_labels=False).to('cuda')
        model_out = wrapper_model.generate(
            **model_inputs, generation_config=gen_config,
            eos_token_id=tokenizer.encode("<|im_end|>", add_special_tokens=False)[0],
            pad_token_id=tokenizer.pad_token_id
        )
        input_tokens_count = len(model_inputs["input_ids"][0])
        prediction_tokens = model_out[0][input_tokens_count:-1]
        prediction_string = tokenizer.decode(prediction_tokens)
        print("test gen:")
        print({"jp": jp, "cn": cn, "pred": prediction_string})

    context = "「そうやってマジな反応されると、さすがに恥ずかしいな。そこのバカみたいにチンポおっ勃ててくれた方が嬉しい」 | 「お前が変態じゃなかったら、勃起していたかもな」 |      男は刀の柄に手を添えて殺気を放ってくる。 |「絶対殺すなよ。こんな美人で乳もいい感じに大きい女って、 滅多にいないんだから。お前はすぐ人を殺すから……」"
    jp = "男は刀の柄に手を添えて殺気を放ってくる。"
    cn = "男人将手伸向刀柄，释放出强烈的杀气。"
    generate_and_print(context, jp, cn)

    context = "2010年、かずさとの別れから3度目の冬が訪れた | 峰城大学3年生となった春希は雪菜と疎遠になり、恋人とも友人とも言いきれない曖昧な関係を続けてきた | かずさは遠く欧州の地でピアニストへの道を歩み | 春希はギターをやめて自分を追いつめるように学業とアルバイトに没頭し、雪菜は歌を忘れた。"
    jp = "かずさは遠く欧州の地でピアニストへの道を歩み"
    cn = "和纱在遥远的欧洲追求成为钢琴家的道路"
    generate_and_print(context, jp, cn)
    print("")


class LoraTrainer:
    def __init__(self) -> None:
        pass

    def train(self, model_dir: str, config: LoraTrainingConfig):
        #original_model, tokenizer = ptuning_model.get_model_and_tokenizer(model_dir)
        wrapper_model, tokenizer = Qwen2Wrapper.from_model_dir(model_dir, config.pre_seq_len, True, device=config.device)

        wrapper_model.set_ptuning_status(config.enable_ptuning)
        model_utils.set_requires_grad(wrapper_model.wrapper_original_model, True)
        model_utils.set_requires_grad(wrapper_model.prefix_encoder.prefix_embedding, False)
        torch.cuda.set_per_process_memory_fraction(0.83, device=0)

        lora_model = get_lora_model(config, wrapper_model.wrapper_original_model, wrapper_model.get_lora_targets())
        wrapper_model.wrapper_lora_model = lora_model
        wrapper_model.current_model = lora_model
        #print(wrapper_model.current_model)
        # for name, p in wrapper_model.wrapper_original_model.named_parameters():
        #     print(p.requires_grad, name)
        parameters_to_train = filter(lambda p: p.requires_grad, lora_model.parameters())
        optimizer = bitsandbytes.optim.AdamW8bit(params=parameters_to_train, lr=config.lr, optim_bits=8)

        all_context_with_embeddings = get_all_context_with_embeddings_from_record()
        print(f"{len(all_context_with_embeddings)} context loaded.")

        random.seed(2333333333333)

        loss_fn = CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        for epoch in range(config.epochs):
            batches = []
            for context in all_context_with_embeddings:
                embedding = context.embedding
                contents = context.context.get_contents()
                context_jp_cn_list = translation_utils.get_context_with_input_and_answer(contents, config.context_radius)
                #contextual_entries = [context_jp_cn for context_jp_cn in context_jp_cn_list]
                # since we have embedding with training, the data of a batch must from the same document
                random.shuffle(context_jp_cn_list)
                context_jp_cn_list = context_jp_cn_list[:20]
                for i in range(0, len(context_jp_cn_list), config.batch_size):
                    batch = (embedding, context_jp_cn_list[i:i+ config.batch_size])
                    batches.append(batch)

            random.shuffle(batches)

            step = 0
            loss_window = []

            def optimize():
                torch.nn.utils.clip_grad_norm_(lora_model.parameters(), config.clip_grad_max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for batch in tqdm.tqdm(batches):
                embedding, context_jp_cn = batch
                context_jp_cn: List[Tuple[List[str], str, str]] = context_jp_cn

                if config.enable_ptuning:
                    new_embedding = clone_embedding(embedding).to(device=config.device)
                    new_embedding.weight.requires_grad_(False)
                    wrapper_model.set_new_embedding(new_embedding)

                prompts_and_mask = [
                    translation_utils.make_contextual_prompt_answer_and_mask_length(tokenizer, context_texts, jp, cn)
                    for context_texts, jp, cn in context_jp_cn
                ]
                prompts, mask_lengths = list(zip(*prompts_and_mask))

                model_inputs, prompt_mask = translation_utils.make_batch_and_prompt_mask(
                    tokenizer, prompts, mask_lengths
                    )
                model_inputs = model_inputs.to(config.device)
                prompt_mask = prompt_mask.to(config.device)

                model_out = wrapper_model(**model_inputs)

                logits = model_out.logits
                # only train the answer part
                labels = torch.where(prompt_mask == 0, tokenizer.pad_token_id, model_inputs["input_ids"])
                labels_flat = labels[..., 1:].contiguous().view(-1)
                logits_flat = logits[..., :-1, :].contiguous().view(-1, logits.shape[-1])
                loss = loss_fn(logits_flat, labels_flat)               
                loss.backward()
                
                step += 1
                loss_window.append(float(loss))
                loss_window = loss_window[len(loss_window) - config.log_per_steps * 4:]
                if step % config.log_per_steps == 0:
                    print(f"loss, avg: {statistics.mean(loss_window):.3f}, max: {max(loss_window):.3f}, min: {min(loss_window):.3f}")
                    print("sample:", tokenizer.decode(model_inputs["input_ids"][0], skip_special_tokens=True))
                    length = prompt_mask[0].cumsum(-1)[-1]
                    print("answer_sample:", tokenizer.decode(model_inputs["input_ids"][0][-length:]))
                    print()

                if step % config.generate_test_per_steps == 0:
                    test_generate(tokenizer, wrapper_model)

                if step % config.grad_accum_step == 0:
                    optimize()
                
            lora_path = "./lora_result"
            if not os.path.exists(lora_path):
                os.mkdir(lora_path)
            current_lora_name = f"lora_{'with' if config.enable_ptuning else 'without'}_ptuning_epoch_{epoch}"
            lora_model.save_pretrained(os.path.join(lora_path, current_lora_name), save_embedding_layers=False)
            print("saving complete!")
                    
            # if step % config.grad_accum_step != 0:
            #     optimize()
            print(f"finish epoch {epoch}!")
            print()
            
        print(f"training complete!")
        print("saving complete!")


if __name__ == "__main__":
    
    model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-3B"
    trainer = LoraTrainer()
    trainer.train(model_dir, LoraTrainingConfig())

    # original_model, tokenizer = ptuning_model.get_model_and_tokenizer(model_dir)
    # lora_model = get_lora_model(LoraTrainingConfig(), original_model)
    # for p in lora_model.parameters():
    #     print(p.shape, p.requires_grad)