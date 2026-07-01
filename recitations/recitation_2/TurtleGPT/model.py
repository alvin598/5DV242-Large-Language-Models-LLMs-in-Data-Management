# TurtleGPT is an ASAP-BNS (As Simple As Possible, But No Simpler) implementation of GPT.
# This code is derived from Andrej Karpathy's MinGPT implementation with additional
# influence from Umar Jamil's implementation of Attention is All you Need.

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import os

ATTN_PDROP = 0.1
RESID_PDROP = 0.1
EMBD_PDROP = 0.1

BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1

class TurtleGPT(nn.Module):

    def __init__(self, c_window_size=None, d_model = None, n_head = None, n_layer=None, vocab_size=None):
        super().__init__()
        assert d_model % n_head == 0

        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.c_window_size = c_window_size

        # ModuleDict registers all these submodules
        self.transformer = nn.ModuleDict({
            "input_embedding" :nn.Embedding(vocab_size, d_model),
            "positional_encoding" :nn.Embedding(c_window_size, d_model),
            "dropout" :nn.Dropout(EMBD_PDROP),
            "layers": nn.ModuleList(
                [TransformerBlock(c_window_size, d_model=d_model, n_head=n_head)
                 for _ in range(n_layer)]),
            "layer_norm": nn.LayerNorm(d_model)}
        )
        self.transformer.to(self.device)

        self.final_linear = nn.Linear(d_model, vocab_size, bias=False)
        self.final_linear.to(self.device)

        self.apply(self._init_weights) # initializes weights for all submodules

        n_params = 0
        for pn, p in self.named_parameters():
            n_params += p.numel()
            if pn.endswith('c_proj.weight'): # apply a special scaled init to the residual projections, per GPT-2 paper
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        print("number of parameters: %.2fM" % (n_params/1e6,))
        if os.path.exists("model.pt"):
            print("loading parameters from model.pt file")
            try:
                self.load_state_dict(torch.load("model.pt", map_location=self.device))
            except:
                print("Error: Incompatible model.pt file")
        else:
            print('Starting from scratch')



    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        else:
            pass


    def configure_optimizers(self, learning_rate):
        """
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        decay = set()
        no_decay = set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, torch.nn.Linear):
                    decay.add(fpn)
                elif (pn.endswith('weight') and
                      (isinstance(m, torch.nn.LayerNorm) or isinstance(m, torch.nn.Embedding))):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": WEIGHT_DECAY},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=BETAS)
        return optimizer

    def forward(self, input, targets=None):
        device = input.device
        b, t = input.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (1, t)
        pos = pos.unsqueeze(0) # pos is (shape 1,t) with vals [[0,1,2...,c_window_size]]

        input_emb = self.transformer.input_embedding(input) # shape (b, t, d_model)
        pos_emb = self.transformer.positional_encoding(pos) # shape (1, t, d_model)
        x = self.transformer.dropout(input_emb + pos_emb)
        for block in self.transformer.layers:
            x = block(x)
        x = self.transformer.layer_norm(x)
        logits = self.final_linear(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, running_input, num_new_tokens, temperature=0.001):
        for _ in range(num_new_tokens):
            if running_input.size(1) <= self.c_window_size:
                input = running_input
            else:
                input = running_input[:, -self.c_window_size:] # only use latest c_window_size tokens

            logits, _ = self(input)
            logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            running_input = torch.cat((running_input, next_token), dim=1)

        return running_input

class TransformerBlock(nn.Module):

    def __init__(self, c_window_size=None, d_model=None, n_head=None):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(c_window_size, d_model=d_model, n_head=n_head)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.ff = nn.ModuleDict({
            "c_fc": nn.Linear(d_model, 4 * d_model),
            "c_proj": nn.Linear(4 * d_model, d_model),
            "Gelu":  NewGELU(),
            "dropout" : nn.Dropout(RESID_PDROP),
        })
        m = self.ff
        self.feedforward = lambda x: m.dropout(m.c_proj(m.Gelu(m.c_fc(x))))

    def forward(self, x):
        x = x + self.attn(self.layer_norm_1(x))  # note residual connection
        x = x + self.feedforward(self.layer_norm_2(x)) # another residual
        return x

class NewGELU(nn.Module):
    def forward(self, x):
        return torch.nn.GELU(approximate='tanh')(x)
        #return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):

    def __init__(self, c_window_size=None, d_model=None, n_head=None):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head

        # Replace the combined attention projection with separate Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # output projection
        self.c_proj = nn.Linear(d_model, d_model)

        # regularization
        self.attn_dropout = nn.Dropout(ATTN_PDROP)
        self.resid_dropout = nn.Dropout(RESID_PDROP)

        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(c_window_size, c_window_size)
            ).view(1, 1, c_window_size, c_window_size))


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values separately
        q = self.q_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        k = self.k_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.v_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Attention calculation remains the same
        # (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y