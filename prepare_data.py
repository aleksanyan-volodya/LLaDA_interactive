"""
I'm gonna use TinyStories as it's a short dataset with now much difficult words. 
I don't want my model to be a complexe one, just a simple thing that can be easly tested online.
I'll train an 8000-token BPE tokenizer.

To Run once:        python prepare_data.py           (for a full ~2.2 GB download)
The quick test:     python prepare_data.py --smoke   (only the small validation file)
"""
import os
import sys

import numpy as np
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["[EOS]", "[MASK]"]  # ids 0 and 1

SMOKE = "--smoke" in sys.argv

# I am gonna use kaggle to train as I don't have GPU :(
if os.path.exists("/kaggle"):
    OUT_DIR = "/kaggle/working"
else:
    OUT_DIR = "data/smoke" if SMOKE else "data"

#################
# We replace non-ASCII punctuation so the small vocab is not wasted on it.
REPLACEMENTS = {"“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-", "…": "...",
                "\x92": "'", "\x93": '"', "\x94": '"'}


def load_(filename):

    path = hf_hub_download("roneneldan/TinyStories", filename, repo_type="dataset")

    with open(path, encoding="utf-8") as f:
        text = f.read()
    for a, b in REPLACEMENTS.items():
        text = text.replace(a, b)

    stories = [s.strip() for s in text.split("<|endoftext|>")]
    return [s for s in stories if s]

##############################
def train_tokenizer(stories):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(stories, trainer)
    return tok


def encode(tok, stories, chunk=10_000):
    eos = tok.token_to_id("[EOS]")
    parts = []
    for i in range(0, len(stories), chunk):
        for enc in tok.encode_batch(stories[i:i + chunk]):
            parts.append(np.array(enc.ids + [eos], dtype=np.uint16))
        print(f"  encoded {min(i + chunk, len(stories))}/{len(stories)} stories", end="\r")
    print()
    return np.concatenate(parts)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    val = load_("TinyStoriesV2-GPT4-valid.txt")
    if SMOKE:
        train, val = val[:4000], val[4000:4500]
    else:
        train = load_("TinyStoriesV2-GPT4-train.txt")
    print(f"{len(train)} train stories, {len(val)} val stories")

    tok = train_tokenizer(train)
    tok.save(os.path.join(OUT_DIR, "tokenizer.json"))
    print("vocab size:", tok.get_vocab_size(), "| [EOS] id:", tok.token_to_id("[EOS]"),
          "| [MASK] id:", tok.token_to_id("[MASK]"))

    for name, stories in [("train", train), ("val", val)]:
        ids = encode(tok, stories)
        np.save(os.path.join(OUT_DIR, f"{name}.npy"), ids)
        print(f"{name}: {len(ids):,} tokens, {len(ids) / len(stories):.0f} tokens per story")

    # Round-trip check: decoding must give back the original story.
    sample = val[0]
    assert tok.decode(tok.encode(sample).ids) == sample, "tokenizer round-trip failed"
    print("round-trip OK:", sample[:100].replace("\n", " "), "...")
