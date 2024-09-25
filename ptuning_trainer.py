import random
import json
import os

import torch
import bitsandbytes

import context_managers
import tqdm
from wrappers.qwen2 import Qwen2Wrapper

import model_utils


class PtuningConfigs:
    pre_seq_len = 10 # virtual tokens num
    lr = 6e-3 # learning rate
    loss_limit = 2.5 # train ptuning until loss less than loss_limit
    qlen_demanded = 200 # p-tuning training sequence length
    batch_size = 4 # p-tuning training batch size
    clip_grad_max_norm = 2.0 # gradient clipping
    device = 'cuda'
    context_max_epoch = 40


class PtuningTrainer:
    def __init__(self) -> None:
        self.context_manager = context_managers.ContextManager()

    def train_context(self, model: Qwen2Wrapper, tokenizer, config, context: context_managers.Context):
        assert isinstance(model, Qwen2Wrapper)
        new_embedding = model.create_and_set_new_embedding()
        params_to_train = new_embedding.parameters()
        optimizer = bitsandbytes.optim.AdamW8bit(params=params_to_train, lr=config.lr)
        
        batches = context.get_batches_for_ptuning(
            "jp", config.batch_size, tokenizer, config.qlen_demanded
            )

        if len(batches) == 0:
            return None, 0.0

        random.seed(23333)
        loss_avg = 999.0
        epoch = 1
        while loss_avg > config.loss_limit and epoch < config.context_max_epoch:
            loss_accum = 0
            random.shuffle(batches)
            for batch in batches:
                batch = batch.to(config.device)
                model_inputs = model.prepare_inputs_for_generation(**batch)
                model_inputs['labels'] = batch["input_ids"]
                model_out = model(**model_inputs)
                loss = model_out.loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_to_train, config.clip_grad_max_norm)
                optimizer.step()
                optimizer.zero_grad()
                loss_accum += float(loss)
            epoch += 1
            loss_avg = loss_accum / len(batches)
            print(f"loss_avg: {loss_avg:.3f}")
        print("sample:", tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True))
        context_with_embedding = context_managers.ContextWithEmbedding(
            context, model.get_current_embedding()
        )
        
        return context_with_embedding, loss_avg

    def train_all(self, model_dir: str, config: PtuningConfigs):
        wrapper_model, tokenizer = Qwen2Wrapper.from_model_dir(model_dir, config.pre_seq_len, True, config.device)
        model_utils.set_requires_grad(wrapper_model.wrapper_original_model, False)
        
        ptuning_result_dir = "ptuning_results"
        if not os.path.exists(ptuning_result_dir):
            os.mkdir(ptuning_result_dir)
        records_path = os.path.join(ptuning_result_dir, "ptuning_training_record.jsonl")

        training_records = []
        embedding_files = set()
        if os.path.exists(records_path):
            with open(records_path, "r", encoding="utf8") as file:
                training_records = list(map(json.loads, file.readlines()))
            embedding_files = set(map(lambda x: x["embedding_path"], training_records))

        for context in tqdm.tqdm(self.context_manager.get_context_list(), desc="ptuning on context", smoothing=0.9):
            embedding_filename = context_managers.ContextWithEmbedding.get_embedding_filename(context.get_filenames())
            if embedding_filename in embedding_files:
                print('existing embedding file detected, skip!')
                continue

            context_with_embedding, loss_avg = self.train_context(wrapper_model, tokenizer, config, context)
            if context_with_embedding is None:
                print("no batch formed, give up...")
            elif loss_avg < config.loss_limit + 0.5:
                print('saving...')
                context_with_embedding.save_embedding(ptuning_result_dir)
                
                record = {
                    "embedding_path": embedding_filename,
                    "files": context.get_filenames()
                }
                with open(records_path, "a", encoding="utf8") as file:
                    json.dump(record, file, ensure_ascii=False)
                    print("", file=file)
                training_records.append(record)
                embedding_files.add(embedding_filename)
            else:
                print("loss too high, give up...")

        print('ptuning training complete!')


if __name__ == "__main__":
    model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-3B"
    trainer = PtuningTrainer()
    trainer.train_all(model_dir, PtuningConfigs())