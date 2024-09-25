import os
import json
import torch
from typing import Tuple, List, Dict
from transformers import AutoTokenizer
import hashlib
import safetensors.torch
import ptuning_utils


class Context:
    def __init__(self, filenames: List[str]) -> None:
        """
        I expect your RAM should hold everything, since we are doing fine tuning.
        For ptuning training. 
        expecting jsonl files. each line with dictionary {jp: xxx, cn: xxx}
        """

        self._filenames = filenames
        self._content = []
        for filename in self._filenames:
            with open(filename, 'r', encoding='utf8') as file:
                pairs = list(map(json.loads, file.readlines()))
                self._content += pairs
        
    def _get_cont_seqs(self, key, tokenizer: AutoTokenizer, seq_len_limit: int) -> Dict:
        input_ids_list = [
            tokenizer(pair[key], return_tensors='pt', add_special_tokens=False)['input_ids'][0]
            for pair in self._content
        ]
        pad_seq = tokenizer(" |", return_tensors='pt', add_special_tokens=False)['input_ids'][0]

        cont_seqs, avg_seq_count = ptuning_utils.create_continuous_seqs(input_ids_list, pad_seq, seq_len_limit, 0)
        offset = int(avg_seq_count // 2)
        if offset != 0:
            cont_seqs += ptuning_utils.create_continuous_seqs(input_ids_list, pad_seq, seq_len_limit, offset)[0]
        return cont_seqs
    
    def get_filenames(self):
        return sorted(self._filenames)

    def get_contents(self):
        return self._content[:]

    def get_batches_for_ptuning(self, key: str, batch_size:int, tokenizer: AutoTokenizer, seq_len_limit: int):
        cont_seqs = self._get_cont_seqs(key, tokenizer, seq_len_limit)
        batches = ptuning_utils.create_ptuning_batches(cont_seqs, batch_size, tokenizer)
        return batches


class ContextManager:
    def __init__(self) -> None:
        """
        Training Context is the essence of the project.

        * The minimum granularity is a single file.
        * Potentially, multiple file can form a context.
          * In the first implementation, a context is a file.
        """
        self.data_registry_path = os.path.join(os.path.split(__file__)[0], "./data_registry.json")
        self.project_indexes = ptuning_utils.create_project_indexes(self.data_registry_path)
        # make contexts
        self.context_list = []
        for project in self.project_indexes:
            print(f"project has {len(project)} files.")
            for i in range(len(project)):
                context = Context([project[i]])
                self.context_list.append(context)
                # enhance
                #enhance_len = 2
                #start = i if i < len(project) - enhance_len else len(project) - enhance_len
                #end = start + enhance_len
                #context = Context(project[start:end])
                #self.context_list.append(context)

    def get_context_list(self) -> List[Context]:
        return self.context_list


class ContextWithEmbedding:
    """for lora training and actual inference"""
    def __init__(self, context: Context, embedding: torch.nn.Embedding) -> None:
        self.context = context
        self.embedding = embedding

    @staticmethod
    def _get_filenames_hash(filenames: List[str]) -> str:
        hash_label = "".join(filenames)
        return hashlib.md5(hash_label.encode('utf8')).hexdigest()
    
    @classmethod
    def get_embedding_filename(cls, filenames: List[str]):
        return cls._get_filenames_hash(filenames) + ".safetensors"
    
    def get_embedding(self):
        return self.embedding
    
    def get_pair_batches_with_embedding(self, batch_size: int) -> List[Tuple[torch.nn.Embedding, List[str]]]:
        batches = []
        content = self.context.get_contents()
        for i in range(0, len(content), batch_size):
            batch = content[i: i+ batch_size]
            batches.append((self.embedding, batch))
        return batches

    def save_embedding(self, ptuning_result_dir: str) -> str:
        embeddings_dir = "embeddings"
        embeddings_dir_full = os.path.join(ptuning_result_dir, embeddings_dir)

        embedding_filename = os.path.join(embeddings_dir_full, self.get_embedding_filename(self.context.get_filenames()))

        if not os.path.exists(embeddings_dir_full):
            os.mkdir(embeddings_dir_full)
        
        safetensors.torch.save_model(self.embedding, embedding_filename)
        return embedding_filename

    @classmethod
    def load_from_files(cls, embedding: torch.nn.Embedding, filenames: List[str]):
        filenames = sorted(filenames)
        embedding_filename = cls.get_embedding_filename(filenames)
        safetensors.torch.load_model(embedding, embedding_filename)
        return cls(embedding, filenames)


class ContextManagerWithEmbedding:
    def __init__(self) -> None:
        """
        This one is for LoRA training.
        """
        self._context_list: List[ContextWithEmbedding] = []

    def add_new_context(self, context: ContextWithEmbedding):
        assert isinstance(context, ContextWithEmbedding)
        self._context_list.append(context)

    # def save_all_contexts(self) -> None:
    #     if not os.path.exists("./embeddings"):
    #         os.mkdir("./embeddings")
    #     for context in self._context_list:
    #         context.save_embedding()

    def from_data_registry_and_saved_embeddings(self) -> None:
        pass

    def create_training_batches(self, batch_size):
        out_batches = []
        for context in self._context_list:
            batches = context.get_pair_batches_with_embedding(batch_size)
            out_batches += batches
        return out_batches

if __name__ == "__main__":
    data_registry_path = os.path.join(os.path.split(__file__)[0], "./data_registry.json")
    # context = Context([
    #     "/home/leo/NLP/datasets/translate-jp-cn/datasets/subtitles/双语字幕/刀语/[诸神&异域][刀语][Katana_Gatari][01-12全][1920x1080][中日双语字幕][MKV][BDRip]/output/[诸神&异域][刀语][Katana_Gatari][09][1920x1080][中日双语字幕][BDRip].jsonl"
    # ])
    # model_dir = "/home/leo/NLP/models/chatGLM/chatglm3-6b-base"
    # tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side='left')
    # # seqs = list(map(len, context._get_cont_seqs('jp', tokenizer, 300)))
    # # print(seqs)
    # batch = context.get_batches_for_ptuning("jp", 4, tokenizer, 200)[0]
    # print(batch)
    context_manager = ContextManager()
    print(len(context_manager.get_context_list()))