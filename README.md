# Small GPT on Indian Politics (built from scratch)

A ~4.8M-parameter, character-level GPT trained from scratch on a laptop, then fine-tuned (SFT) to answer short civics questions. The goal is **learning how an LLM is built end to end** (data, tokenizer, pretraining, SFT, evaluation), not producing a reliable question-answering system.

> **Disclaimer:** This model hallucinates. It answers confidently even when it is wrong. Do **not** use it as a source of facts about Indian politics or the Constitution.

---

## 1. Results at a glance

| Item | Value |
|---|---|
| Model | Decoder-only Transformer, 6 layers, 8 heads, 256-dim, **4,807,168 parameters** |
| Tokenizer | Character-level, vocab size 138 |
| Pretraining data | 157,358 words (English Wikipedia, 42 Indian-politics articles + 36 Q&A samples) |
| Pretraining time | ~25 min (1,484 s) on an Apple-silicon Mac (MPS), 4,000 steps |
| Best validation loss | **1.128** (cross-entropy per character, ~3.09 perplexity per character) |
| SFT data | 208 training Q&A pairs (52 facts x 4 phrasings) |
| **Accuracy on seen facts, new phrasing** | **44 / 52 = 84.6%** (base model before SFT: 7 / 52 = 13.5%) |
| **Accuracy on facts never trained on** | **0 / 8 = 0%** (base model: 0 / 8) |
| Wrong-but-confident answers (hallucination) | 8 / 52 = **15.4%** on seen facts, 8 / 8 = **100%** on unseen facts |

**In one line:** after SFT the model reliably answers questions about the 52 facts it was taught, even when phrased differently, but it has no real knowledge beyond them and never says "I don't know".

---

## 2. Project files

| File | Purpose |
|---|---|
| `collect_data.py` | Downloads Wikipedia articles, cleans them, builds `data.txt` (with retry, rate-limit handling and a `raw/` cache) |
| `train_gpt.py` | Tokenizer, GPT model, pretraining loop, text generation |
| `sft_facts.py` | The Q&A facts used for SFT (training / held-out / unseen) |
| `sft.py` | Supervised fine-tuning, before/after evaluation, chat mode |
| `ckpt.pt` | Pretrained base model checkpoint |
| `ckpt_sft.pt` | SFT model checkpoint |

---

## 3. Data

**Source:** English Wikipedia (via the MediaWiki API), 42 articles on Indian civics and politics, e.g. Constitution of India, Lok Sabha, Rajya Sabha, Supreme Court, Election Commission, Fundamental Rights, Panchayati Raj, Right to Information Act, GST, and others (full list in `collect_data.py`).

**Cleaning:**
- removed section headings and `[1]`-style citations
- normalised dashes and curly quotes to plain ASCII (an early version deleted them, turning `1951-52` into `1951 52`)
- dropped lines shorter than 6 words and exact duplicate lines

**Size:** 157,358 words. A first run only produced 72,281 words because Wikipedia returned HTTP 429 (rate limit) for 20 of 42 topics; this was fixed with retries, a 2 s delay between requests and local caching.

**Q&A part of the pretraining data:** 12 facts x 3 formats = 36 samples (about 4% of the data).

**Train/validation split:** the text is cut into ~1,500-character chunks, shuffled with a fixed seed, then split 90/10 (a plain "last 10%" split would have put all Q&A samples in validation). Q&A chunks in the training split are repeated 10 times.
Result: 926,272 training tokens, 98,791 validation tokens.

---

## 4. Pretraining

Task: predict the next character. Loss: cross-entropy. Optimizer: AdamW (betas 0.9/0.95, weight decay 0.1), gradient clipping at 1.0, linear warmup then cosine decay.

Two model sizes were trained for comparison (same data, same tokenizer):

| | v1 (small) | **v2 (final)** |
|---|---|---|
| Layers / heads / embedding | 4 / 4 / 128 | **6 / 8 / 256** |
| Parameters | 827,392 | **4,807,168** |
| Context length (block size) | 128 chars | 128 chars |
| Batch size | 64 | 64 |
| Dropout | 0.2 | 0.15 |
| Peak learning rate | 1e-3 | 6e-4 (min 6e-5, warmup 200) |
| Steps | 3,000 | 4,000 |
| Training time | ~5 min | ~25 min |
| Best validation loss | 1.405 | **1.128** |
| Final train loss | 1.332 | 0.909 |

**v2 loss curve (selected steps):**

| Step | Train loss | Val loss | Gap |
|---|---|---|---|
| 0 | 5.002 | 5.004 | 0.002 |
| 250 | 2.316 | 2.330 | 0.014 |
| 500 | 1.736 | 1.788 | 0.051 |
| 1000 | 1.313 | 1.405 | 0.093 |
| 2000 | 1.072 | 1.227 | 0.155 |
| 3000 | 0.955 | 1.162 | 0.207 |
| 4000 | 0.909 | 1.128 | 0.219 |

The train/validation gap widens slowly (0.22 at the end) while validation loss was still decreasing, so mild overfitting had started but the best checkpoint was the last one. The v1 model was underfitting (gap 0.07, loss still falling when training stopped).

Note: character-level losses are not comparable to token-level (BPE) losses of other models.

**What the base model does after pretraining:** it writes fluent, Wikipedia-style English with mostly correct spelling and relevant vocabulary. It memorised the 12 Q&A facts but, as free text, it invents details. Example hallucinations from the base model at temperature 0.8-1.0: *"Article 33 of the President of India..."*, *"allowed 29 seats in the Lok Sabha"*. At temperature 1.0 spelling also starts to break. It also does not know when to stop answering and keeps generating new Q&A pairs.

---

## 5. Supervised fine-tuning (SFT)

Starts from the pretrained v2 checkpoint.

- **Format:** `Q: <question>` newline `A: <answer>~` where `~` is an end-of-answer marker (the rarest ASCII character in the vocabulary), so the model learns to stop.
- **Facts:** 52 facts, each written in 5 different phrasings. 4 phrasings are used for training (208 examples), the 5th is held out for testing (52 examples).
- **Unseen set:** 8 additional facts that never appear in SFT data.
- **Loss** is computed on answer tokens only, not on the question.
- **Hyperparameters:** 25 epochs, batch size 16, learning rate 2e-4 with cosine decay to 10%, no weight decay.
- Answers are deliberately short, because the model can only see 128 characters at once.

| Epoch | Train loss | Held-out loss |
|---|---|---|
| before SFT | - | 2.342 |
| 1 | 0.898 | 0.357 |
| 5 | 0.138 | 0.109 |
| 10 | 0.044 | 0.055 |
| 15 | 0.021 | 0.035 |
| 20 | 0.014 | 0.032 |
| 25 | 0.009 | 0.030 |

A train loss of 0.009 means the training answers are close to memorised. The held-out loss is also low because held-out questions are re-phrasings of facts the model has seen, not new facts.

---

## 6. Evaluation: how many answers are correct, how many hallucinate

### Method
- Greedy decoding (always pick the most likely next character), stop at `~` or a newline, max 80 characters.
- An answer counts as **correct** if the expected key phrase (e.g. `president`, `26 january 1950`, `543`) appears in the generated answer (case-insensitive). This is an automatic, lenient check, not a human grade.
- **Hallucination rate** here means **wrong-but-fluent answers**. The model was never trained to abstain, so every wrong answer is a confident wrong answer.

### Results

| Question set | n | Base model correct | SFT model correct | SFT model wrong |
|---|---|---|---|---|
| Held-out phrasings of seen facts | 52 | 7 (13.5%) | **44 (84.6%)** | 8 (15.4%) |
| Unseen facts | 8 | 0 (0%) | **0 (0%)** | 8 (100%) |
| Both sets combined | 60 | 7 (11.7%) | 44 (73.3%) | 16 (26.7%) |

- The combined number mixes two very different situations and should not be read as a general error rate. On arbitrary politics questions the real error rate would be far higher, because the model only knows 52 facts.
- With n = 52, the 84.6% has an approximate 95% confidence interval of roughly 72%-92%. The unseen set (n = 8) is too small for a precise rate, but 0/8 for both models is a clear signal.
- Only the first 12 held-out cases were printed in the run log. The 8 held-out errors are therefore known in detail for only 1 case (see below); the remaining 7 were counted but not inspected.

### First 12 held-out questions (from the run log)

| Question | Expected | Base | SFT |
|---|---|---|---|
| Who is the head of the Indian state? | The President | OK (long sentence) | OK |
| Who is the head of the Indian government? | The Prime Minister | OK (long sentence) | OK |
| Who appoints the Prime Minister of India? | The President | wrong | **OK** |
| From which date is the Indian Constitution in force? | 26 January 1950 | wrong | **OK** |
| The Indian Constitution was adopted on which date? | 26 November 1949 | wrong | **OK** |
| How many constituencies elect members to the Lok Sabha? | 543 seats | wrong | **OK** |
| What is the normal duration of a Lok Sabha? | Five years | OK | **wrong** ("The Speaker.") |
| The Rajya Sabha can have how many members at maximum? | 250 members | OK | OK |
| What is the length of a Rajya Sabha member's term? | Six years | OK | OK |
| Which house of Parliament is always in existence? | The Rajya Sabha | wrong | **OK** |
| Which officer conducts the proceedings of the Lok Sabha? | The Speaker | wrong | **OK** |
| Which authority organises elections in India? | Election Commission of India | wrong | **OK** |
| **Total** | | **5 / 12** | **11 / 12** |

(The base column on the full set is 7/52. Several "OK" base answers are long memorised sentences that happen to contain the keyword.)

### Unseen facts: hallucination examples

| Question | Expected | SFT answer | What went wrong |
|---|---|---|---|
| How many members does the President nominate to the Rajya Sabha? | 12 members | "30 years." | Wrong type of answer (an age) |
| How many languages are listed in the Eighth Schedule? | 22 languages | "25 years." | Wrong type of answer |
| At what age do High Court judges retire? | 62 years | "65 years." | Mixed up with Supreme Court judges |
| Who presides over a joint sitting of Parliament? | The Speaker | "The Rajya Sabha." | Plausible-sounding but wrong |
| Who elects the members of the Rajya Sabha? | Elected members of state legislative assemblies | "Directly by the people." | Mixed up with the Lok Sabha |
| Who can dissolve the Lok Sabha? | The President | "Five years." | Answered a different question |
| Which article gives the right to constitutional remedies? | Article 32 | "Equality before the law." | Mixed up with Article 14 |
| What is the normal term of a Governor? | Five years | "The President." | Wrong |

### Informal chat test (`python sft.py --chat`)

| Prompt | Answer | Verdict |
|---|---|---|
| "Who appoints the Prime Minister of India?" (with quotes) | The President. | Correct |
| who is india prime minister (lowercase, different grammar) | The President. | Wrong (also asks for a name, which the data never contains) |

---

## 7. What the model did and did not learn

**Learned**
- Spelling, grammar and Wikipedia-like style of the domain
- The Q&A format, short answers, and when to stop (the `~` marker)
- Mapping of different phrasings of the same question to the correct memorised answer (84.6% on held-out phrasings, up from 13.5%)

**Did not learn**
- Any fact it was not trained on (0/8)
- To say "I don't know" - it always answers, even with the wrong type of answer
- Robustness to very different wording, casing or typos
- Reasoning: it matches surface patterns to memorised answers rather than understanding the question

**One regression:** the base model answered "normal duration of a Lok Sabha" correctly, the SFT model did not ("The Speaker."). Fine-tuning on a small dataset can overwrite some previously correct behaviour.

---

## 8. Limitations and caveats

- **Tiny scale:** ~1M characters and 4.8M parameters. Real LLMs use trillions of tokens and billions of parameters; knowledge and robustness come from that scale.
- **128-character context:** question plus answer must fit in 128 characters, so answers are very short.
- **Small evaluation:** 52 held-out and 8 unseen questions, one run, one seed (1337). Treat percentages as indicative.
- **Metric is lenient and noisy:** keyword matching can mark an answer correct by chance. Several facts share answers ("The President." appears for 4 facts; "Five years." for 3), so some "Who/How long" questions can be right by luck. The 12 original Q&A facts were also seen during pretraining, which slightly helps the base model.
- **Q&A facts were written by an AI assistant (Claude) and not independently verified by a domain expert.** Facts such as the number of states and union territories may change over time. Check `sft_facts.py` before reusing it.
- **Data bias:** Wikipedia text (including pages about political parties) can carry its own biases and may be out of date.
- **English only.**
- SFT training time was not measured (it took a few minutes).

---

## 9. How to reproduce

```bash
python -m pip install torch tokenizers requests

python collect_data.py          # builds data.txt (~157k words)
python train_gpt.py             # pretraining -> ckpt.pt (~25 min on an Apple M-series Mac)
python sft.py                   # SFT + evaluation -> ckpt_sft.pt

python sft.py --chat            # talk to the SFT model
python sft.py --ask "Who appoints the Prime Minister of India?"
python train_gpt.py --prompt "The Lok Sabha is" --temp 0.5   # base-model text generation
```

Edit the `CONFIG` block at the top of `train_gpt.py` and `sft.py` to change model size, steps, tokenizer (`"char"` or `"bpe"`) and learning rates.

---

## 10. Possible next steps

1. **SFT v2:** more phrasings (typos, lowercase, short forms), fewer epochs, and mixing some pretraining text to reduce forgetting.
2. **Teach abstention:** add out-of-scope questions with an "I don't know" answer.
3. **BPE tokenizer and longer context:** remove the 128-character limit and re-pretrain.
4. **Compare with a real pretrained model** (e.g. a ~0.5B open model fine-tuned with LoRA) on the same questions to see the effect of scale.
5. **Larger, human-verified Q&A set** with a proper held-out benchmark.

---

## 11. Licence and attribution

- Training text comes from English Wikipedia, which is licensed under CC BY-SA. If you share `data.txt` or derived text, keep that licence and attribution in mind.
- Code in this repository is for learning purposes; add your own licence file before publishing.
