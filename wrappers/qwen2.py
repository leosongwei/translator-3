from transformers import DynamicCache, Cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Tuple, List, Dict, Any
import torch
from peft import LoraConfig, get_peft_model, LoraModel

from wrappers.prefix_encoder import PrefixEncoder


class Qwen2Wrapper(Qwen2ForCausalLM):
    def __init__(self, model: Qwen2ForCausalLM, num_virtual_tokens: int, enable_ptuning = True) -> None:
        Qwen2PreTrainedModel.__init__(self, model.config)
        # Pretend to be Qwen2ForCausalLM
        self.model = model.model
        self.vocab_size = model.config.vocab_size
        self.lm_head = model.lm_head

        self.wrapper_original_model: Qwen2ForCausalLM = model
        self.wrapper_lora_model = None

        self._ptuning_is_enabled = enable_ptuning
        self.current_model = self.wrapper_original_model
        
        self.num_virtual_tokens = num_virtual_tokens
        self.prefix_encoder = PrefixEncoder(
            num_virtual_tokens, self._get_embedding_dim(model),
            dtype=model.dtype, device=model.device
            )

    @staticmethod
    def get_lora_targets():
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            #"gate_proj", 
            #"up_proj",
            #"down_proj"
            "lm_head"
        ]

    @classmethod
    def from_model_dir(cls, model_dir: str, num_virtual_tokens: int, enable_ptuning: bool, device=None) -> Tuple["Qwen2Wrapper", AutoTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, use_safetensors=True).to(device=device).eval()
        return cls(model, num_virtual_tokens, enable_ptuning), tokenizer

    def create_and_set_new_embedding(self) -> torch.nn.Embedding:
        self.prefix_encoder = PrefixEncoder(
            self.num_virtual_tokens, self._get_embedding_dim(self.wrapper_original_model),
            dtype=self.wrapper_original_model.dtype, device=self.wrapper_original_model.device
        )
        return self.prefix_encoder.prefix_embedding

    def set_new_embedding(self, embedding: torch.nn.Embedding):
        self.prefix_encoder.prefix_embedding = embedding

    def get_current_embedding(self):
        return self.prefix_encoder.prefix_embedding

    @staticmethod
    def _get_embedding_dim(model):
        layers = model.config.num_hidden_layers
        num_key_value_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        embedding_dim = 2 * layers * num_key_value_heads * head_dim
        return embedding_dim
    
    def _get_prompt(self, batch_size):
        model = self.wrapper_original_model
        layers = model.config.num_hidden_layers
        num_key_value_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads

        prefix_tokens = torch.arange(self.num_virtual_tokens, dtype=torch.long)\
            .unsqueeze(0)\
                .expand(batch_size, -1).to(device=model.device)
        past_key_values = self.prefix_encoder(prefix_tokens).to(model.dtype)
        # key: bz, num_key_value_heads, q_len, head_dim
        #   example: torch.Size([3, 2, 6, 128])
        past_key_values = past_key_values.view(
            batch_size, self.num_virtual_tokens,
            layers * 2,
            num_key_value_heads, head_dim
        )
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4])
        past_key_values = past_key_values.split(2)
        return DynamicCache.from_legacy_cache(past_key_values)
       
    def set_ptuning_status(self, should_enable: bool):
        self._ptuning_is_enabled = should_enable

    def _update_model_kwargs_for_generation(
        self,
        outputs,
        model_kwargs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        batch_size, attention_mask_seq_len = model_kwargs["attention_mask"].shape

        past_key_values = outputs.past_key_values
        if not isinstance(past_key_values, Cache):
            #past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
            kv_cache_seq_len = past_key_values[0][0].shape[2]
        else:
            kv_cache_seq_len = past_key_values.get_seq_length()

        if kv_cache_seq_len != attention_mask_seq_len:
            attention_mask = model_kwargs["attention_mask"]
            new_attention_mask = torch.ones(
                (batch_size, kv_cache_seq_len),
                dtype=attention_mask.dtype, device=attention_mask.device
            )
            new_attention_mask[:, -attention_mask_seq_len:] = attention_mask
            model_kwargs["attention_mask"] = new_attention_mask
        return super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, **kwargs
        )

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None
            ):
        
        if self._ptuning_is_enabled and past_key_values is None:
            batch_size, _ = input_ids.shape
            past_key_values = self._get_prompt(batch_size)
            attention_mask = torch.cat([
                attention_mask.new_ones((batch_size, self.num_virtual_tokens)),
                attention_mask
            ], dim=-1)

        out: CausalLMOutputWithPast = self.current_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position
        )
        return out


if __name__=="__main__":
    from transformers.generation import GenerationConfig
    model_dir = "/home/leo/NLP/models/Qwen-VL/Qwen2.5-3B"
    wrapped_model, tokenizer = Qwen2Wrapper.from_model_dir(model_dir, 20, True, "cuda")
    
    gen_config = GenerationConfig(
        max_length=50,
        do_sample=True,
        top_p=0.95,
        no_repeat_ngram_size=2,
        num_beams=3,
        use_cache=True
    )

    input_texts = ["李鸿章", "1 2 3 4"]
    model_inputs = tokenizer(input_texts, padding=True, return_tensors="pt").to('cuda')
    #model_inputs = model.prepare_inputs_for_generation(**model_inputs)
    #wrapped_model.set_ptuning_status(False)
    #wrapped_model.wrapper_original_model.load_adapter("./lora_out_without_ptuning")
    wrapped_model.set_ptuning_status(True)
    wrapped_model.current_model = wrapped_model.wrapper_original_model
    out = wrapped_model.generate(**model_inputs, generation_config=gen_config)

    print(tokenizer.decode(out[0]))