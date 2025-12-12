import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from transformers import GPT2Tokenizer, GPT2LMHeadModel
 
# Hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
vocab_size = 50257 # Total Number of Tokens in the GPT2 Tokenizer
block_size = 1024 # This is how many tokens the model can handle at once, matches GPT2
n_embd = 768 # The dimensionality of the token, matches GPT2
n_head = 12 # Number of attention heads, matches GPT2
n_layer = 12 # Number of Layers in the Model, matches GPT2
dropout = 0.1 # Dropout which doesn't even matter since I am just porting the GPT2 weights
    
# Attention Implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd) # Query, Key, and Value projections combined for maximum efficiency
        self.c_proj = nn.Linear(n_embd, n_embd) # Projection Layer

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.n_heads = num_heads
        # Flash Attention mask (causal masking)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape

        # Calculate Query, Key, Value
        q, k, v = self.c_attn(x).split(n_embd, dim=2)

        # Reshape for Multi-Head Attention: (B, T, nh, hs) -> (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) 
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Scaled Dot-Product Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float('-inf')) 
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v 
        
        # Reassemble all heads side-by-side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        
        return y

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(approximate='tanh'), # GPT-2 uses tanh approximation
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
    
class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, input, targets=None):
        B, T = input.shape

        tok_emb = self.token_embedding_table(input)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, input, max_new_tokens):
        for _ in range(max_new_tokens):

            input_cond = input[:, -block_size:]

            logits, loss = self(input_cond)

            logits = logits[:, -1, :] 

            probs = F.softmax(logits, dim=-1)

            input_next = torch.multinomial(probs, num_samples=1)

            input = torch.cat((input, input_next), dim=1) 
        return input

def load_weights(model):
    print("Loading weights from Hugging Face GPT-2...")
    model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = model_hf.state_dict() 
    sd = model.state_dict()

    # Mapping layer
    sd['token_embedding_table.weight'].copy_(sd_hf['transformer.wte.weight'])
    sd['position_embedding_table.weight'].copy_(sd_hf['transformer.wpe.weight'])
    sd['ln_f.weight'].copy_(sd_hf['transformer.ln_f.weight'])
    sd['ln_f.bias'].copy_(sd_hf['transformer.ln_f.bias'])
    sd['lm_head.weight'].copy_(sd_hf['lm_head.weight'])

    with torch.no_grad():
        for i in range(12):

            prefix_hf = f'transformer.h.{i}.'
            
            block_mine = model.blocks[i] 
            
            block_mine.ln1.weight.copy_(sd_hf[prefix_hf + 'ln_1.weight'])
            block_mine.ln1.bias.copy_(sd_hf[prefix_hf + 'ln_1.bias'])
            
            # Transpose Conv1D weights for Linear layers
            block_mine.sa.c_attn.weight.copy_(sd_hf[prefix_hf + 'attn.c_attn.weight'].t())
            block_mine.sa.c_attn.bias.copy_(sd_hf[prefix_hf + 'attn.c_attn.bias'])

            block_mine.sa.c_proj.weight.copy_(sd_hf[prefix_hf + 'attn.c_proj.weight'].t())
            block_mine.sa.c_proj.bias.copy_(sd_hf[prefix_hf + 'attn.c_proj.bias'])

            block_mine.ln2.weight.copy_(sd_hf[prefix_hf + 'ln_2.weight'])
            block_mine.ln2.bias.copy_(sd_hf[prefix_hf + 'ln_2.bias'])

            block_mine.ffwd.net[0].weight.copy_(sd_hf[prefix_hf + 'mlp.c_fc.weight'].t())
            block_mine.ffwd.net[0].bias.copy_(sd_hf[prefix_hf + 'mlp.c_fc.bias'])

            block_mine.ffwd.net[2].weight.copy_(sd_hf[prefix_hf + 'mlp.c_proj.weight'].t())
            block_mine.ffwd.net[2].bias.copy_(sd_hf[prefix_hf + 'mlp.c_proj.bias'])
        
    print("Weights loaded successfully.")
    return model

if __name__ == "__main__":

    model = GPT2()
    model = load_weights(model)
    model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    print("\n" + "="*50)
    print("GPT-2 Inference Engine Ready")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("Prompt (Type 'exit' or 'quit' to exit.) > ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(input_ids, max_new_tokens=50)

            generated_text = tokenizer.decode(output_ids[0].tolist())
            print(f"\nResponse:\n{generated_text}\n")
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break