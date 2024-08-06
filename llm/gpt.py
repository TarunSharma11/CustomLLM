import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from ..modules.block import Block


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02*(2*(self.config.n_layer**-0.5)))
            else:
                torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
    
    def forward(self, idx, targets = None):
        #idx is of shape B,T
        T = idx.shape[1] if idx.shape[1] <= self.config.block_size else self.config.block_size
        word_embd = self.transformer.wte(idx)     
        pos_embd = self.transformer.wpe(torch.arange(T, dtype = torch.long, device = idx.device))
        x = word_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, model, learning_rate, weight_decay, device):
        param_dict = {name: param for name, param in model.named_parameters()}
        trainable_param_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}

        decay_params = [param for name, param in trainable_param_dict.items() if param.dim() >= 2]
        non_decay_params = [param for name, param in trainable_param_dict.items() if param.dim()   < 2]
        optim_groups = [
            {
                "params": decay_params,
                "weight_decay": weight_decay
            },
            {
                "params": non_decay_params,
                "weight_decay": 0.0
            }
        ]
        total_decayed_params = sum([p.numel() for p in decay_params])
        total_non_decayed_params = sum([p.numel() for p in non_decay_params])
        print(f"Decayed parameters : {total_decayed_params} | Non decayed params: {total_non_decayed_params}")
        contains_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        is_cuda = device == 'cuda'
        fused = contains_fused and is_cuda
        optimizer = torch.optim.AdamW(optim_groups, lr = learning_rate, betas = (0.9, 0.95), eps = 1e-8, fused = fused)
        return optimizer