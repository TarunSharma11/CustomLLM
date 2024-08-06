import os
import math
import time
import json
import torch
import tiktoken

import torch.nn as nn
import torch.distributed as dist

from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .llm.gpt import GPT
from .config import GPTConfig
from .dataset import DataLoaderLite
from .utils.hellaswag import get_hellaswag_val_data


## Setup DDP
isDDP = int(os.environ.get("RANK", -1)) != -1
if isDDP:
    dist.init_process_group(backend = 'nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = 'cuda:' + str(local_rank)
    isMaster = rank == 0
else:
    rank = 0
    local_rank = 0
    world_size = 1
    isMaster = True

## Define constants 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_steps = 100
warmup_steps = 10
max_lr = 6e-4
min_lr = 0.1 * max_lr
epochs = 19074 #(10^10 / 2**19 i.e. dataset size/batch size)
weight_decay = 0.1
B = 64
T = 1024
total_batch_size = 2**19
acc_steps = total_batch_size // (B*T*world_size)
train_dataloader = DataLoaderLite(16, 1024, rank, world_size, 9, 'train')
val_dataloader = DataLoaderLite(16, 1024, rank, world_size, 9, 'val')
model_checkpoints_dir = 'checkpoints/'
hellaswag_val_data = get_hellaswag_val_data('./data/hellaswag/val_data.jsonl')

## Set seed for reproducability
torch.manual_seed(100)
if torch.cuda.is_available():
    torch.cuda.manual_seed(100)
    

def get_lr(epoch):
    if(epoch <= warmup_steps):
        return max_lr * epoch/warmup_steps
    if(epoch > max_steps):
        return min_lr
        
    curr_ratio = (epoch - warmup_steps)/(max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * curr_ratio))
    lr = min_lr + coeff * (max_lr - min_lr)
    return lr


def calculate_and_propagate_loss(x, y, acc_loss):
    with torch.autocast(device_type = device, dtype = torch.bfloat16):
        logits, loss = model(x,y)
        loss = loss / acc_steps
    acc_loss += loss.detach() 
    loss.backward()
    return logits, loss, acc_loss

def create_tensor_example_batch(example):
    ctx = example['ctx']
    label = example['label']
    tokenizer = tiktoken.get_encoding('cl100k_im')
    ctx_enc = tokenizer.encode(ctx)
    endings_enc = [tokenizer.encode(ending) for ending in  example['endings']]
    examples_enc = [ctx_enc + end_enc for end_enc in endings_enc]
    max_len = max(len(ex_enc) for ex_enc in examples_enc)
    padding_lens = [max_len - len(ex_enc) for ex_enc in examples_enc]
    padded_examples_enc = [ex_enc + [0]*pad_len for ex_enc, pad_len in zip(examples_enc, padding_lens)]
    examples_tensor = torch.tensor(padded_examples_enc, dtype = torch.long)
    
    padding_mask = torch.ones((4, max_len))
    for i, pad_len in enumerate(padding_lens):
        padding_mask[i, -pad_len:] = 0

    ending_mask = torch.zeros(4, max_len)
    for i, end_len in enumerate([len(end_enc) for end_enc in endings_enc]):
        ending_mask[i, len(ctx_enc): len(ctx_enc)+end_len] = 1

    return examples_tensor, padding_mask, ending_mask, label
    
    
## Initialize model
model = GPT(GPTConfig())
model.train()
model.to(device)
model = torch.compile(model)
if isDDP:
    model = DDP(model, device_ids = [local_rank])
    raw_model = model.module
else:
    raw_model = model
    
optimizer = raw_model.configure_optimizer(6e-4, weight_decay = 0.1)

torch.set_float32_matmul_precision('high')


encoder = tiktoken.get_encoding("cl100k_base")
generator_feed_tokens = torch.tensor(encoder.encode("Hello, I'm a language model,"), dtype = torch.long)
num_rand_generation = 5
generator_feed_tokens = generator_feed_tokens.unsqueeze(0).repeat(num_rand_generation,1).to(device)
generator_max_length = 50
checkpoint_dir = 'checkpoints/'
for ep in range(epochs):
    last_step = ep == epochs - 1
    if (ep > 0) and (last_step or (ep % 500) == 0):
        eval_steps = 10
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            for val_step in range(eval_steps):
                val_x, val_y = val_dataloader.next_batch()
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_loss_acc = 0.0
                with torch.autocast(device_type = device, dtype = torch.bfloat16):
                    _, val_loss = model(val_x, val_y)   
                    val_loss = val_loss / eval_steps
                val_loss_acc += val_loss.detach()
            if isDDP:
                torch.distributed.all_reduce(val_loss_acc, op = dist.ReduceOp.AVG)

        if ((not isDDP) or (isMaster)):
            print(f"Epoch : {ep + 1} | Val Loss: {val_loss_acc.item():.4f}")

    if (last_step or (ep % 500) == 0):
        hl_total_correct = 0
        hl_total_elem = 0
        for i, example in enumerate(hellaswag_val_data):
            if ((i - rank) % world_size == 0) :
                json_ex = json.loads(example)
                example_tensor, padding_mask, ending_mask, label = create_tensor_example_batch(json_ex)
                example_tensor, padding_mask, ending_mask = example_tensor.to('cuda'), padding_mask.to('cuda'), ending_mask.to('cuda')
                logits,_ = model(example_tensor)
                probs = F.softmax(logits, dim = 2)
                probs = probs[:, :-1, :].contiguous()
                y = example_tensor[:, 1:].contiguous()
                ending_mask = ending_mask[:, 1:].contiguous()
                probs = probs.view(-1, probs.shape[-1])
                y = y.view(-1)
                loss = F.cross_entropy(probs, y, reduction = 'none')
                loss = loss.view(4, -1)
                loss_endings = loss * ending_mask
                loss_sum = torch.sum(loss_endings, dim = 1)
                loss_avg = loss_sum / torch.sum(ending_mask, dim = 1)
                pred_label = torch.argmin(loss_avg).item()
                hl_total_correct += int(pred_label == label)
                hl_total_elem += 1
        if isDDP:
            torch.distributed.all_reduce(hl_total_correct, op = dist.ReduceOp.SUM)
            torch.distributed.all_reduce(hl_total_elem, op = dist.ReduceOp.SUM)
        
        if isMaster:
            print(f"Hellaswag accurancy: {(hl_total_correct / hl_total_elem)* 100}")

    if (last_step or (ep % 500) == 0):
        generated_tokens = torch.clone(generator_feed_tokens)
        for i in range(generator_max_length):
            with torch.no_grad():
                gen_logits, _ = model(generated_tokens)
                gen_logits = gen_logits[:, -1, :] # B, C
                gen_probs = F.softmax(gen_logits, dim = -1)
                topk_probs, topk_indices = torch.topk(gen_probs, 50, dim = -1) # B,50 B,50
                ix = torch.multinomial(topk_probs, 1) #B,1
                tokens_ix = torch.gather(topk_indices, 1, ix)
                generated_tokens = torch.cat((generated_tokens, tokens_ix), dim = -1)
                
        for i in range(num_rand_generation):
            print(f"Epoch : {ep + 1} | Rank: {rank} | Generated Text: {encoder.decode(generated_tokens.detach().cpu().numpy()[i])}")

    t0 = time.time()
    optimizer.zero_grad()
    lr = get_lr(ep+1)
    for param in optimizer.param_groups:
        param['lr'] = lr
    acc_loss = 0.0
    total_tokens_processed = 0
    for step in range(acc_steps):
        x,y = train_dataloader.next_batch()
        B,T = x.shape
        total_tokens_processed += (B*T)
        x,y = x.to(device), y.to(device)
        if isDDP and (step != (acc_steps - 1)):
            with model.no_sync():
                logits, loss, acc_loss = calculate_and_propagate_loss(x,y, acc_loss)
        else:
            logits, loss, acc_loss = calculate_and_propagate_loss(x,y, acc_loss)
    if isDDP:
        torch.distributed.all_reduce(acc_loss, op = dist.ReduceOp.AVG)
        torch.distributed.all_reduce(total_tokens_processed, op = dist.ReduceOp.SUM)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tk_sec = total_tokens_processed/(t1 - t0)
    if ((not isDDP) or (isMaster)):
        print(f"Epoch : {ep + 1} | Loss: {acc_loss.item():.4f} | Time taken: {dt:.2f} milli sec | Tokens per sec: {tk_sec:.2f}")

    if (ep > 0) and (last_step or (ep % 5000) == 0):
        checkpoint_dict = {
            'model_dict': raw_model.state_dict(),
            'config': raw_model.config,
            'val_loss': val_loss_acc.item(),
            'step': ep
        }
        checkpoint_file_name = checkpoint_dir + f"model_epoch_{ep:05d}.pt"
        torch.save(checkpoint_dict, checkpoint_file_name) 
    
if isDDP:
    destroy_process_group()
