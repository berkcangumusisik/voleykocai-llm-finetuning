# voleykocai-llm-finetuning

**VoleykoçAI**: a Turkish-speaking volleyball coaching assistant. This repo holds the four parts of an AI course assignment in one place: a dataset, a BPE tokenizer, a LoRA adapter, and a separate identity fine-tune.

I picked volleyball as the theme because it lets the first three assignments feed each other: the dataset's language becomes the tokenizer's corpus, and the terms the tokenizer learns are the language the model speaks.

---

## Submission links

| Assignment | Deliverable | Hugging Face | In this repo |
|---|---|---|---|
| 1 | Dataset | [`voleykoc-antrenorluk-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-antrenorluk-tr) | [`01-dataset/`](01-dataset/) |
| 2 | BPE tokenizer | [`voleykoc-bpe-tokenizer`](https://huggingface.co/berkcangumusisik/voleykoc-bpe-tokenizer) | [`02-tokenizer/`](02-tokenizer/) |
| 3 | LoRA adapter | [`voleykoc-qwen3-4b-lora`](https://huggingface.co/berkcangumusisik/voleykoc-qwen3-4b-lora) | [`03-finetune/`](03-finetune/) |
| Bonus | Identity fine-tune | [`voleykoc-identity-tr`](https://huggingface.co/datasets/berkcangumusisik/voleykoc-identity-tr) · [`voleykoc-identity-lora`](https://huggingface.co/berkcangumusisik/voleykoc-identity-lora) | [`04-identity/`](04-identity/) |

Full upload walkthrough: [HUGGINGFACE.md](HUGGINGFACE.md)

---

## How to run it yourself

```bash
# 0) Set up the environment (Python 3.11+)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1) Scrape the source material                 [needs internet, ~2 min]
python 01-dataset/scrape.py

# 2) Expand the 20 hand-written seeds           [model mode if ANTHROPIC_API_KEY is set]
python 01-dataset/augment.py

# 3) Merge and validate the dataset
python 01-dataset/build_dataset.py              # -> data/train.jsonl

# 4) Train the BPE tokenizer
python 02-tokenizer/train_tokenizer.py          # -> 02-tokenizer/voleykoc-bpe-tokenizer/

# 5) Build the identity dataset (bonus)
python 04-identity/build_identity.py            # -> data/identity/*.jsonl

# 6) Publish to Hugging Face                    [needs a token]
python 01-dataset/upload.py
python 02-tokenizer/upload.py
python 04-identity/upload.py

# 7) Fine-tune                                  [needs Colab + a T4 GPU]
#    Open 03-finetune/finetune_voleykoc.ipynb and
#    04-identity/finetune_identity.ipynb in Colab and run them
```

Steps 1–6 run on this machine. Step 7 does not: see "What doesn't run on this machine" below.

---

## Where the data came from

Three sources, tagged by the `source` field in `data/train.jsonl`:

| Source | Examples | How |
|---|---:|---|
| `wikipedia` | 96 | 17 volleyball pages on Turkish Wikipedia |
| `synthetic` | 60 | Expanded from 20 hand-written seeds |
| `tvf` | 10 | tvf.org.tr news pages |
| **Total** | **166** | after dedup |

**Wikipedia pages:** Voleybol, Plaj voleybolu, Oturarak voleybol, FIVB, CEV, TVF, the Turkish women's and men's national teams, Sultanlar Ligi, Efeler Ligi, Kadınlar 1. Ligi, 1. Lig, Fenerbahçe, Eczacıbaşı, Galatasaray, Beşiktaş (women), Halkbank, Arkas.

I verified every page title against the MediaWiki API first. Some titles I expected (Smaç, Manşet, Servis) aren't separate pages on Turkish Wikipedia: those technique terms live inside the main Voleybol article. Others, like "Pasör", redirect to Voleybol; `scrape.py` reads each page's canonical title so the same article isn't downloaded twice.

**TVF:** The federation publishes its rulebook only as a PDF, with no HTML page. So I crawl the news feed instead: collect `/icerik/...` links from the index pages, then visit each article.

**robots.txt:** I checked both sources by hand and again from inside the script: `tr.wikipedia.org` only disallows `/w/` and `/wiki/Special:`, and `tvf.org.tr` says `User-agent: * / Allow: /`. `scrape.py` re-checks on every run and skips a source if the rules have changed. Requests are spaced one second apart.

**Licensing:** Wikipedia content is **CC BY-SA 4.0**, and dataset rows derived from it inherit that licence. TVF content belongs to tvf.org.tr and is used here only in transformed form for educational purposes.

---

## Assignment 1: the dataset

**Schema.** The assignment asks for the format used by `alibayram/identity_finetune_magibu_q3`. That repo is gated, so I took the schema from the same team's open dataset (`magibu/turkish-multi-turn-dialog-dataset`):

```json
{"system": "...", "source": "wikipedia",
 "conversations": [{"role": "user", "content": "..."},
                   {"role": "assistant", "content": "..."}],
 "num_turns": 2}
```

**From scraped text to Q&A.** Wikipedia's HTML is flat: headings and paragraphs sit at the same level in document order. `scrape.py` walks that order, opening a new section on each heading and appending paragraphs to the section it's currently in. `build_dataset.py` then turns each section into a question, templated from the section heading, with the paragraphs as the answer.

Template choice isn't random; it's keyed on the character sum of the heading, so re-running the script produces the same output:

```python
def pick(templates: list[str], key: str) -> str:
    """Deterministic template choice, keyed on the heading (not random)."""
    return templates[sum(ord(c) for c in key) % len(templates)]
```

Rather than squeezing a long section into one answer and throwing away the rest, I split it into chunks: no data lost, more examples gained. The second and third chunk get `(devamı, 2. bölüm)` appended to the question so dedup doesn't mistake them for duplicates.

**The synthetic side.** `seeds.jsonl` holds 20 coaching Q&A pairs I wrote by hand: the 5-1 rotation, forearm-pass technique, spike approach rhythm, libero rules, a U14 training plan, jump training, injury prevention, match-day nutrition. All real coaching content, not filler.

`augment.py` expands them. It has two modes: with `ANTHROPIC_API_KEY` set it asks a model for genuinely new Q&A pairs; without one it rewrites the question from templates and copies the answer from the seed.

> **The `train.jsonl` in this repo was generated in offline mode.** That means the 60 synthetic examples reuse answers from the 20 seeds. `reports/dataset_stats.md` reports this as a unique-answer ratio (125 of 166, 75%). Re-running `augment.py` with an API key raises both that ratio and the total example count substantially.

**Validation.** Before writing anything, `build_dataset.py` checks every row: correct field set, roles in user→assistant order, `num_turns` matching the message count, no empty content. If it finds a problem it writes nothing and exits.

---

## Assignment 2: the BPE tokenizer

There are **two** BPE implementations in this repo, and there's a reason.

`bpe.py` is the from-scratch implementation I wrote for my previous assignment ([countries-bpe-tokenizer](https://github.com/berkcangumusisik/countries-bpe-tokenizer)): pure standard library, with `train`/`encode`/`decode`/`save`/`load`. I carried it over as evidence that I understand the algorithm. But it writes its own JSON format, which Hugging Face can't read.

`train_tokenizer.py` trains the published tokenizer with the `tokenizers` library instead: same algorithm (byte-level BPE), but files that reload through `AutoTokenizer.from_pretrained()`.

**Corpus.** All the dataset's text plus the raw scraped volleyball prose: 214,015 characters / 29,532 words. The system message is identical on every row, so it enters the corpus only once. Otherwise the tokenizer memorises that one sentence.

**Result.** Target vocab was 16,000; the actual figure is 7,532. The corpus is small and I set `min_frequency=2`, so merges ran out early. I left the real number rather than padding the vocabulary artificially.

**Domain efficiency.** Across the same 10 Turkish volleyball sentences:

| | Total tokens |
|---|---:|
| VoleykoçAI | **114** |
| Qwen3-4B-Instruct-2507 | 179 |

36% fewer in this domain. Because the vocabulary was learned entirely from Turkish volleyball text, `pasör`, `manşet` and `rotasyon` stay whole, while Qwen's multilingual vocabulary splits them apart. The fair reading: this measures efficiency **in this domain**, not general quality. Qwen's vocabulary covers hundreds of languages; mine covers one subject.

**Byte fallback.** With a 256-byte base vocabulary there is no `<unk>`: an unseen character (an emoji, another alphabet) is still represented by its raw bytes. All 10 test sentences round-tripped through encode→decode unchanged; one of them contains an emoji on purpose.

Full report: [`reports/tokenizer_report.md`](reports/tokenizer_report.md)

---

## Assignment 3: the fine-tune

`03-finetune/finetune_voleykoc.ipynb`, on a Colab T4:

| | |
|---|---|
| Base model | `unsloth/Qwen3-4B-Instruct-2507` |
| Method | 4-bit QLoRA |
| LoRA | r=16, alpha=16, dropout=0 |
| Target modules | q, k, v, o, gate, up, down |
| Training | 3 epochs, lr=2e-4, effective batch 8 |
| Seed | 1337 |

Instead of training all 4B parameters, I add small low-rank matrices to the attention and MLP layers; under 1% of parameters are trained. That's what makes it fit on Colab.

The notebook asks the same five questions **before** and **after** training and writes both sets of answers to `karsilastirma.json`. That's how I show the fine-tune actually changed something: a loss curve alone doesn't demonstrate it.

Only the LoRA adapter is published (~100 MB), not the merged model (~8 GB). The assignment asks for the adapter anyway.

### Why doesn't the fine-tune use my tokenizer?

This is the first thing anyone reading the repo will ask, so it gets its own heading.

The tokenizer from Assignment 2 is a **standalone deliverable**. The adapter from Assignment 3 doesn't use it: it was trained with the base model's own tokenizer.

The reason: changing a model's vocabulary requires resizing the embedding layer and retraining from scratch. Qwen3-4B's embeddings were learned for 151,000 tokens; moving them onto a new 7,532-token vocabulary is well outside what LoRA fine-tuning does. The assignment doesn't ask for it either: it asks for three separate deliverables.

So the tokenizer proves "I can train a domain-specific vocabulary," and the adapter proves "I can fine-tune a model on this domain." Two different questions.

---

## Bonus: identity fine-tuning

This is a separate process from the first three assignments. The goal isn't to teach the model volleyball; it's to teach it **who it is**.

`identity_seeds.jsonl` holds 26 identity Q&A pairs (15 Turkish, 11 English) across seven categories: name, creator, purpose, capabilities, **limits**, language, and how it was trained. `build_identity.py` expands them to 182 examples via question templates and splits them into `turkish` / `english`, mirroring the instructor's dataset.

Binding one answer to many phrasings is **deliberate** here: "what's your name", "who are you" and "introduce yourself" all resolve to the same fact, and the model learns its identity from that repetition. In the main dataset that repetition was a flaw; here it's the method.

I included the `sinir` (limits) category on purpose: an identity is defined as much by what a model *isn't* as by what it is. Those examples teach it not to diagnose injuries and to redirect out-of-domain questions.

Training is capped at `max_steps=60`. The identity data is small and repetitive; training longer degrades the model's volleyball knowledge and general fluency. At the end the notebook also asks one non-identity question (`5-1 rotasyon sistemi nasıl çalışır?`) to check the model can still do its job.

---

## What doesn't run on this machine

The assignment asks for training "via Unsloth". Unsloth depends on Triton, and Triton isn't available on macOS: this repo was developed on an Apple M5 Pro, so **training does not run here**.

That's why Assignment 3 and the bonus are Colab notebooks. Everything else: data, tokenizer, uploads: runs locally; only the two GPU-bound steps moved to Colab.

LoRA training on Apple Silicon is genuinely possible via MLX, but the assignment says Unsloth explicitly, so I didn't take that route.

---

## Repo layout

```
01-dataset/      scrape.py, seeds.jsonl, augment.py, build_dataset.py, upload.py
02-tokenizer/    bpe.py (from scratch), train_tokenizer.py (HF format), upload.py
03-finetune/     finetune_voleykoc.ipynb  [Colab]
04-identity/     identity_seeds.jsonl, build_identity.py, finetune_identity.ipynb, upload.py
data/            train.jsonl, identity/*.jsonl  (raw/ is gitignored)
reports/         dataset_stats.md, tokenizer_report.md, identity_stats.md
hf_upload.py     shared confirm + upload logic behind the three upload.py scripts
HUGGINGFACE.md   step-by-step upload guide
```

Each stage folder has a `NOTLAR.md`: which scripts run in what order, what they produce, what they need. So you can return to one stage without re-reading this README.

---

## Credits & references

- Turkish Wikipedia: volleyball pages (CC BY-SA 4.0)
- [Turkish Volleyball Federation](https://tvf.org.tr)
- [Unsloth](https://github.com/unslothai/unsloth): fast LoRA fine-tuning
- [`alibayram/identity_finetune_magibu_q3`](https://huggingface.co/datasets/alibayram/identity_finetune_magibu_q3): dataset schema reference
- [`magibu/turkish-multi-turn-dialog-dataset`](https://huggingface.co/datasets/magibu/turkish-multi-turn-dialog-dataset): the open example of that schema
- [Qwen3](https://huggingface.co/Qwen): base model
- My own previous assignment: [countries-bpe-tokenizer](https://github.com/berkcangumusisik/countries-bpe-tokenizer): where `bpe.py` came from

## License

[MIT](LICENSE). For the data sources' own licences, see "Where the data came from" above.

---

🇹🇷 [Türkçe sürüm](README.tr.md)
