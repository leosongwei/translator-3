import torch


class PrefixEncoder(torch.nn.Module):
    def __init__(self, num_virtual_tokens: int, embedding_dim: int, dtype = None, device = None) -> None:
        super().__init__()
        self.prefix_embedding = torch.nn.Embedding(
            num_virtual_tokens, embedding_dim, dtype=dtype, device=device
        )
    
    def forward(self, prefix_token_ids: torch.LongTensor):
        return self.prefix_embedding(prefix_token_ids)
    