"""
sft.py - Step 3: Supervised Fine-Tuning (SFT)

Pehle train_gpt.py se pretraining ho chuki honi chahiye (CKPT file maujood ho).
Run:   python sft.py
Chat:  python sft.py --chat
Ask:   python sft.py --ask "Who appoints the Prime Minister of India?"
"""
import argparse
import copy
import math
import random

import torch
import torch.nn.functional as F

from train_gpt import GPT, CharTok, BLOCK_SIZE, DATA_FILE, load_for_generation, get_device
from sft_facts import FACTS, UNSEEN

# ======================= CONFIG =======================
EPOCHS = 25
BATCH = 16
LR = 2e-4            # pretraining (6e-4) se chhota: model ka purana gyaan na bigde
SEED = 1337
SFT_CKPT = "ckpt_sft.pt"
MAX_NEW = 80
# ======================================================


def pick_end_token(tok):
    """END marker: ek rare character jo vocab mein ho. Model ko seekhna hai 'yahan answer khatam'."""
    try:
        data = open(DATA_FILE, encoding="utf-8").read()
    except FileNotFoundError:
        data = ""
    cands = [c for c in "~^|@#`$*" if c in tok.stoi]
    if not cands:
        raise SystemExit("Vocab mein koi END character nahi mila. sft.py ka pick_end_token badlo.")
    return min(cands, key=lambda c: data.count(c))


def prompt_of(q):
    return f"Q: {q}\nA:"


def encode_example(tok, q, a, end):
    p = tok.encode(prompt_of(q))
    t = tok.encode(f" {a}{end}")
    ids = p + t
    labels = [-100] * len(p) + t      # loss sirf ANSWER par, sawal par nahi (asli SFT aise hi hota hai)
    return ids, labels


def make_batch(examples, tok, end, device):
    enc = [encode_example(tok, q, a, end) for q, a in examples]
    L = max(len(i) for i, _ in enc) - 1
    x = torch.zeros(len(enc), L, dtype=torch.long)
    y = torch.full((len(enc), L), -100, dtype=torch.long)
    for r, (ids, lab) in enumerate(enc):
        n = len(ids) - 1
        x[r, :n] = torch.tensor(ids[:-1])
        y[r, :n] = torch.tensor(lab[1:])
    return x.to(device), y.to(device)


def sft_loss(model, batch):
    x, y = batch
    logits, _ = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)


@torch.no_grad()
def eval_loss(model, examples, tok, end, device):
    model.eval()
    total = 0.0
    for i in range(0, len(examples), 32):
        total += sft_loss(model, make_batch(examples[i:i + 32], tok, end, device)).item() * len(examples[i:i + 32])
    model.train()
    return total / len(examples)


@torch.no_grad()
def answer(model, tok, q, end, device):
    """Greedy generation: END ya newline aate hi ruk jao."""
    model.eval()
    x = torch.tensor([tok.encode(prompt_of(q))], dtype=torch.long, device=device)
    out = []
    for _ in range(MAX_NEW):
        logits, _ = model(x[:, -BLOCK_SIZE:])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        c = tok.decode([nxt.item()])
        if c == end or c == "\n":
            break
        out.append(c)
        x = torch.cat([x, nxt], dim=1)
    return "".join(out).strip()


def score(model, tok, rows, end, device):
    results, ok = [], 0
    for q, a, key in rows:
        out = answer(model, tok, q, end, device)
        good = key in out.lower()
        ok += good
        results.append((q, a, out, good))
    return ok / len(rows), results


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    base, tok, device = load_for_generation()
    if tok.kind != "char":
        raise SystemExit("sft.py abhi sirf char tokenizer ke liye hai.")
    end = pick_end_token(tok)
    print(f"Device: {device} | END token: {end!r}")

    # ---- data ----
    train, held, skipped = [], [], 0
    for key, ans, qs in FACTS:
        for q in qs[:-1]:
            if len(prompt_of(q)) + len(ans) + 2 <= BLOCK_SIZE + 1:
                train.append((q, ans))
            else:
                skipped += 1
        held.append((qs[-1], ans, key))
    unseen = [(q, a, k) for k, a, q in UNSEEN]
    all_text = "".join(prompt_of(q) + " " + a for q, a in train)
    missing = set(all_text) - set(tok.stoi)
    print(f"Train examples: {len(train)} | held-out paraphrases: {len(held)} | unseen facts: {len(unseen)} | skipped (too long): {skipped}")
    if missing:
        print("WARNING: ye characters vocab mein nahi hain (drop honge):", missing)

    held_pairs = [(q, a) for q, a, _ in held]
    sft = copy.deepcopy(base)
    sft.train()

    print(f"\nBefore SFT -> held-out loss (answer tokens par): {eval_loss(sft, held_pairs, tok, end, device):.3f}")

    opt = torch.optim.AdamW(sft.parameters(), lr=LR, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / BATCH)
    total_steps, step = EPOCHS * steps_per_epoch, 0

    for ep in range(1, EPOCHS + 1):
        random.shuffle(train)
        run = 0.0
        for i in range(0, len(train), BATCH):
            lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total_steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            loss = sft_loss(sft, make_batch(train[i:i + BATCH], tok, end, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sft.parameters(), 1.0)
            opt.step()
            run += loss.item()
            step += 1
        if ep % 5 == 0 or ep == 1:
            hl = eval_loss(sft, held_pairs, tok, end, device)
            print(f"epoch {ep:2d} | train loss {run / steps_per_epoch:.3f} | held-out loss {hl:.3f}")

    torch.save({"model": sft.state_dict(), "kind": "char", "chars": tok.chars,
                "vocab_size": tok.vocab_size, "end": end}, SFT_CKPT)
    print(f"\nSaved: {SFT_CKPT}\n")

    # ---- Evaluation: base vs SFT ----
    b_acc, b_rows = score(base, tok, held, end, device)
    s_acc, s_rows = score(sft, tok, held, end, device)
    bu_acc, bu_rows = score(base, tok, unseen, end, device)
    su_acc, su_rows = score(sft, tok, unseen, end, device)

    print("=" * 64)
    print("RESULT (sahi = expected keyword answer mein aaya)")
    print(f"  Held-out paraphrases (naye shabd, fact seen) : base {b_acc*100:5.1f}%  ->  SFT {s_acc*100:5.1f}%")
    print(f"  Unseen facts (fact training mein tha hi nahi): base {bu_acc*100:5.1f}%  ->  SFT {su_acc*100:5.1f}%")
    print("=" * 64)

    print("\n--- Held-out paraphrases (pehle 12) ---")
    for (q, a, bo, bg), (_, _, so, sg) in list(zip(b_rows, s_rows))[:12]:
        print(f"Q: {q}\n   expected: {a}\n   base : {bo[:70]}  [{'OK' if bg else 'X'}]\n   SFT  : {so[:70]}  [{'OK' if sg else 'X'}]")

    print("\n--- Unseen facts (honest test) ---")
    for (q, a, bo, bg), (_, _, so, sg) in zip(bu_rows, su_rows):
        print(f"Q: {q}\n   expected: {a}\n   base : {bo[:70]}  [{'OK' if bg else 'X'}]\n   SFT  : {so[:70]}  [{'OK' if sg else 'X'}]")

    print("\nNote: 'Who' wale sawalon mein 'The President.' bahut baar aata hai, isliye kuch OK chance se bhi ho sakte hain.")


def load_sft():
    device = get_device()
    ck = torch.load(SFT_CKPT, map_location=device)
    tok = CharTok(ck["chars"])
    model = GPT(ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    return model, tok, ck["end"], device


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ask", type=str, default=None)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()
    if args.ask:
        m, t, e, d = load_sft()
        print(answer(m, t, args.ask, e, d))
    elif args.chat:
        m, t, e, d = load_sft()
        print("SFT model se poochho (band karne ke liye khaali enter):")
        while True:
            q = input("\nQ: ").strip()
            if not q:
                break
            print("A:", answer(m, t, q, e, d))
    else:
        main()