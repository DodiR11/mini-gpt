import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        #mask so tokens can't see future tokens
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        # calculate attention scores and scale them
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # # block future positions
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )
        #convert attention scores into probabilities
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        head_size = n_embd // n_head

        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size)
            for _ in range(n_head)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
    # concatenate the outputs from all attention heads
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        self.sa = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
         # residual connections around attention and FF layers
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size=64,
                 n_embd=128, n_head=4, n_layer=2):
        super().__init__()

        self.block_size = block_size
        # embeddings for character IDs and their positions
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        )

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        # predict scores for every character in the vocabulary
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # combine batch and sequence dimensions for cross entropy
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
             #only the last block_size characters get used as context
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # only the last position predicts the next character
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            # sample the next character and add it to the sequence
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx
