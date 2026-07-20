"""Byte-level Byte-Pair Encoding (BPE) tokenizer, written from scratch.

The design follows the references from the lecture:
  - HuggingFace LLM course, chapter 6.5 (the training loop of counting the most
    frequent adjacent pair and merging it)
  - karpathy/minbpe (byte-level base vocabulary -> no <unk> token is needed,
    thanks to "byte fallback": every possible byte 0..255 is already a token)
  - Sebastian Raschka's "BPE from scratch" (storing vocab + merges, save/load)

Why byte level?  Because the base vocabulary is the 256 possible bytes, *any*
UTF-8 text can be encoded without an unknown token.  This is exactly the
"byte fallback" idea discussed in class: instead of emitting <unk> for a symbol
the tokenizer has never seen, it falls back to the raw bytes of that symbol.

The tokenizer is also *word aware*: text is first split into words with a
regex (pre-tokenization) and merges are never allowed to cross a word
boundary.  For a corpus of country names this lets BPE discover meaningful
sub-words such as "stan", "land", "ia" or " Islands" while never gluing two
separate country names together.
"""

from __future__ import annotations

import json
import re
from collections import Counter

# Pre-tokenization pattern (a small, GPT-2-style split).  It keeps a leading
# space attached to a word (" Islands"), and separates letters, digits and
# punctuation into their own chunks so merges stay inside a single "word".
# `[^\W\d_]` is the stdlib-`re` way of writing "a Unicode letter" (\p{L}),
# which keeps accented country names such as "Cote d'Ivoire" intact.
SPLIT_PATTERN = re.compile(
    r""" ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+""",
    re.UNICODE,
)


def get_stats(ids: list[int], counts: dict[tuple[int, int], int] | None = None) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair appears in a list of token ids.

    Optionally accumulate into an existing `counts` dict so statistics can be
    gathered across many words without allocating a new dict each time.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with the single `new_id`."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """A minimal byte-level BPE tokenizer with word-aware pre-tokenization."""

    def __init__(self) -> None:
        # merges: (int, int) -> int   (a pair of ids -> the id of the new token)
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: int -> bytes         (a token id -> the raw bytes it stands for)
        self.vocab: dict[int, bytes] = {idx: bytes([idx]) for idx in range(256)}

    # ---- training ---------------------------------------------------------

    def _pretokenize(self, text: str) -> list[str]:
        """Split raw text into word chunks before byte encoding."""
        return SPLIT_PATTERN.findall(text)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn `vocab_size - 256` merge rules from `text`.

        Steps (matching the HuggingFace description):
          1. pre-tokenize into words, encode each word to its UTF-8 bytes
          2. repeatedly find the most frequent adjacent pair across all words
          3. merge that pair into a new token id, record the rule
          4. stop once the vocabulary reaches `vocab_size`
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte base)"
        num_merges = vocab_size - 256

        # Each word becomes a list of byte ids.  We also keep a frequency for
        # each distinct word so a country that appears once is not overcounted.
        word_freqs = Counter(self._pretokenize(text))
        words: list[list[int]] = [list(w.encode("utf-8")) for w in word_freqs]
        freqs: list[int] = [word_freqs[w] for w in word_freqs]

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            # 1. count pair frequencies across every word (weighted by freq)
            stats: dict[tuple[int, int], int] = {}
            for ids, freq in zip(words, freqs):
                for pair in zip(ids, ids[1:]):
                    stats[pair] = stats.get(pair, 0) + freq
            if not stats:
                break  # nothing left to merge

            # 2. pick the most frequent pair (ties broken by pair order for
            #    reproducibility -- this is why re-runs are deterministic)
            best_pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
            if stats[best_pair] < 2:
                break  # no pair repeats; further merges would be meaningless

            # 3. mint a new token and rewrite every word
            new_id = 256 + i
            words = [merge(ids, best_pair, new_id) for ids in words]
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose:
                token = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1:>3}: {best_pair} -> {new_id} ({token!r}) "
                      f"had {stats[best_pair]} occurrences")

    # ---- encoding / decoding ---------------------------------------------

    def _encode_chunk(self, ids: list[int]) -> list[int]:
        """Apply learned merges to one word's byte ids, lowest-rank first."""
        while len(ids) >= 2:
            stats = get_stats(ids)
            # choose the pair whose merge was learned earliest (lowest new_id).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no remaining pair can be merged
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode a string into a list of token ids."""
        ids: list[int] = []
        for chunk in self._pretokenize(text):
            ids.extend(self._encode_chunk(list(chunk.encode("utf-8"))))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string."""
        data = b"".join(self.vocab[idx] for idx in ids)
        return data.decode("utf-8", errors="replace")

    def tokens(self, text: str) -> list[str]:
        """Human-readable tokens (for demos): decode each id on its own."""
        return [self.vocab[idx].decode("utf-8", errors="replace") for idx in self.encode(text)]

    # ---- persistence ------------------------------------------------------

    def save(self, path: str) -> None:
        """Save merges + vocab to a JSON file."""
        payload = {
            "vocab_size": len(self.vocab),
            # store merges as ["id1 id2", new_id] so JSON keys stay simple
            "merges": [[f"{a} {b}", idx] for (a, b), idx in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load a tokenizer previously written by `save`."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        tok = cls()
        # merges must be replayed in order so the vocab bytes rebuild correctly
        for pair_str, new_id in sorted(payload["merges"], key=lambda x: x[1]):
            a, b = (int(x) for x in pair_str.split())
            tok.merges[(a, b)] = new_id
            tok.vocab[new_id] = tok.vocab[a] + tok.vocab[b]
        return tok

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
