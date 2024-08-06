from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 1024
    n_head: int = 12
    n_layer: int = 12
    n_embd: int = 768
