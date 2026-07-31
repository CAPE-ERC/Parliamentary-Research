"""Word-level vocabulary for the from-scratch LSTM embedding layer.

No pretrained embeddings - trained from scratch, which is appropriate (and
fast on CPU) for a small model over formulaic procedural language, unlike the
Topic layer's open-ended policy classification.
"""

import re
from collections import Counter

PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-zà-ÿ]+|\d+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], max_size: int = 20_000, min_freq: int = 2) -> dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common(max_size):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)[:max_len]]
    ids += [vocab[PAD]] * (max_len - len(ids))
    return ids
