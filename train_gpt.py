"""
train_gpt.py - data.txt par chhota GPT scratch se train karo

Install:   pip install torch tokenizers
Train:     python train_gpt.py
Generate:  python train_gpt.py --prompt "Question: Who is the head of state in India?\nAnswer:" --temp 0.5
"""
import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================= CONFIG (yahan khelo) =======================
DATA_FILE = "data.txt"
TOKENIZER = "char"       # "char" (pehle ye) ya "bpe" (baad mein try karo)
BPE_VOCAB = 2000

BLOCK_SIZE = 128         # model ek baar mein kitne tokens peeche tak dekhta hai
BATCH_SIZE = 64
N_LAYER = 6
N_HEAD = 8
N_EMBD = 256
DROPOUT = 0.15           # overfit dikhe to badhao (0.3), underfit ho to ghatao

MAX_ITERS = 4000
EVAL_EVERY = 250
EVAL_ITERS = 40
LR = 6e-4
MIN_LR = 6e-5
WARMUP = 200
QA_REPEAT = 10           # Q&A samples kam hain, isliye training mein 10 baar repeat
SEED = 1337
CKPT = "ckpt.pt"
TOK_FILE = "tokenizer.json"
# ====================================================================


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------- Tokenizers -----------------------------
class CharTok:
    kind = "char"

    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


class BpeTok:
    kind = "bpe"

    def __init__(self, tok):
        self.tok = tok

    @property
    def vocab_size(self):
        return self.tok.get_vocab_size()

    def encode(self, s):
        return self.tok.encode(s).ids

    def decode(self, ids):
        return self.tok.decode(ids)


def build_bpe(text):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=BPE_VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    tok.train_from_iterator(text.split("\n"), trainer)
    tok.save(TOK_FILE)
    return BpeTok(tok)


# ------------------------------- Data ---------------------------------
def make_chunks(text, size=1500):
    """Text ko chhote tukdon mein todo (Q&A pair ek hi tukda rehta hai)."""
    chunks = []
    for block in text.split("\n\n"):
        if len(block) <= size:
            if block.strip():
                chunks.append(block)
            continue
        cur = ""
        for line in block.split("\n"):
            if cur and len(cur) + len(line) > size:
                chunks.append(cur)
                cur = ""
            cur += line + "\n"
        if cur.strip():
            chunks.append(cur)
    return chunks


def prepare(text):
    """Tukde shuffle karke 90/10 split. (Seedha last 10% lete to poora Q&A validation mein chala jata.)"""
    chunks = make_chunks(text)
    rng = random.Random(SEED)
    rng.shuffle(chunks)
    n = int(0.9 * len(chunks))
    train, val = chunks[:n], chunks[n:]

    train_chunks = []
    for c in train:
        reps = QA_REPEAT if c.startswith(("Question:", "Q:")) else 1
        train_chunks += [c] * reps
    rng.shuffle(train_chunks)
    return "\n\n".join(train_chunks), "\n\n".join(val)


def get_batch(data, device):
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(device), y.to(device)


# ------------------------------- Model --------------------------------
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD), nn.Dropout(DROPOUT),
        )
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v)]
        # Self-attention (causal: har token sirf apne se pehle wale tokens dekh sakta hai)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))   # residual connection
        x = x + self.mlp(self.ln2(x))     # feed-forward + residual
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new=300, temperature=0.8, top_k=40):
        for _ in range(max_new):
            logits, _ = self(idx[:, -BLOCK_SIZE:])
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def sample(model, tok, prompt, device, temperature=0.8, n=300):
    model.eval()
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new=n, temperature=temperature)[0].tolist()
    return tok.decode(out)


# ------------------------------ Training ------------------------------
@torch.no_grad()
def estimate_loss(model, splits, device):
    model.eval()
    res = {}
    for name, data in splits.items():
        losses = torch.zeros(EVAL_ITERS)
        for i in range(EVAL_ITERS):
            x, y = get_batch(data, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        res[name] = losses.mean().item()
    model.train()
    return res


def lr_at(it):
    if it < WARMUP:
        return LR * (it + 1) / WARMUP
    ratio = (it - WARMUP) / max(1, MAX_ITERS - WARMUP)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * ratio)) * (LR - MIN_LR)


def train():
    torch.manual_seed(SEED)
    device = get_device()
    print("Device:", device)

    text = open(DATA_FILE, encoding="utf-8").read()
    train_text, val_text = prepare(text)

    tok = build_bpe(train_text) if TOKENIZER == "bpe" else CharTok(sorted(set(text)))
    print(f"Tokenizer: {tok.kind} | vocab size: {tok.vocab_size}")

    train_data = torch.tensor(tok.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tok.encode(val_text), dtype=torch.long)
    print(f"Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")

    model = GPT(tok.vocab_size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    best_val, t0 = float("inf"), time.time()

    for it in range(MAX_ITERS + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % EVAL_EVERY == 0:
            l = estimate_loss(model, {"train": train_data, "val": val_data}, device)
            flag = ""
            if l["val"] < best_val:
                best_val = l["val"]
                flag = "  <- best, saved"
                torch.save({
                    "model": model.state_dict(), "kind": tok.kind,
                    "chars": getattr(tok, "chars", None),
                    "vocab_size": tok.vocab_size,
                }, CKPT)
            gap = l["val"] - l["train"]
            print(f"step {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | gap {gap:.3f} | {time.time()-t0:.0f}s{flag}")

        if it == MAX_ITERS:
            break
        x, y = get_batch(train_data, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\nTraining done. Best val loss: {best_val:.3f}. Checkpoint: {CKPT}")
    print("Best checkpoint se samples:\n")
    run_generate(
        "Question: Who is the head of state in India?\nAnswer:", [0.5, 0.8, 1.0]
    )


# ----------------------------- Generation -----------------------------
def load_for_generation():
    device = get_device()
    ck = torch.load(CKPT, map_location=device)
    if ck["kind"] == "char":
        tok = CharTok(ck["chars"])
    else:
        from tokenizers import Tokenizer
        tok = BpeTok(Tokenizer.from_file(TOK_FILE))
    model = GPT(ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    return model, tok, device


def run_generate(prompt, temps):
    model, tok, device = load_for_generation()
    for t in temps:
        print(f"----- temperature {t} -----")
        print(sample(model, tok, prompt, device, temperature=t))
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--temp", type=float, default=None)
    args = ap.parse_args()
    if args.prompt is not None:
        prompt = args.prompt.replace("\\n", "\n")
        run_generate(prompt, [args.temp] if args.temp else [0.5, 0.8, 1.0])
    else:
        train()