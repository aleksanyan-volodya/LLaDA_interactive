"""Generate text with a trained mask predictor (LLaDA reverse process).

python sample.py results_v1/model_step60000.pt --prompt "Once upon a time" --steps 64
python sample.py CKPT --suffix "and they lived happily ever after."   (infilling: the text goes at the end)
python sample.py CKPT --show-steps                                      (print the sequence at every step)

Works with both ckpt.pt and model_step*.pt (both contain "model" and "config").
"""
import argparse

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, MaskPredictor


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    model = MaskPredictor(Config(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model


@torch.no_grad()
def generate_steps(model, ids, steps, strategy="confidence", temperature=0.0, seed=0):
    """Fill every [MASK] in `ids` in `steps` steps. Yields the state after each step (first = input, last = result).

    At each step: predict all masked positions, keep k of them, the others stay masked.
    strategy="confidence": keep the k predictions with the highest probability.
    strategy="random":     keep k random predictions.
    temperature=0 takes the most likely token, >0 samples from softmax(logits / temperature).
    """
    mask_id = model.cfg.mask_id
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.tensor(ids, device=device)
    assert len(x) <= model.cfg.seq_len, f"prompt + length must be <= {model.cfg.seq_len} tokens"

    n_masked = int((x == mask_id).sum())
    steps = max(1, min(steps, n_masked))
    # How many tokens to reveal at each step, spread as evenly as possible (they sum to n_masked).
    counts = [n_masked // steps + (1 if i < n_masked % steps else 0) for i in range(steps)]

    yield x.tolist()
    for k in counts:
        masked = x == mask_id
        logits = model(x[None])[0].float()       # (T, vocab)
        logits[:, mask_id] = -float("inf")        # never predict [MASK] itself
        if temperature > 0:
            pred = torch.multinomial(F.softmax(logits / temperature, -1), 1, generator=gen).squeeze(-1)
        else:
            pred = logits.argmax(-1)

        if strategy == "confidence":
            score = F.softmax(logits, -1).gather(-1, pred[:, None]).squeeze(-1)
        elif strategy == "random":
            score = torch.rand(len(x), generator=gen, device=device)
        else:
            raise ValueError(f"unknown strategy {strategy}")
        score[~masked] = -float("inf")            # already revealed tokens are never changed
        keep = score.topk(k).indices
        x[keep] = pred[keep]
        yield x.tolist()


def generate(model, ids, steps, strategy="confidence", temperature=0.0, seed=0):
    """Same as generate_steps, but returns the list of all states at once."""
    return list(generate_steps(model, ids, steps, strategy, temperature, seed))


def make_input(tok, mask_id, length, prompt="", suffix=""):
    """prompt tokens, then `length` masks, then suffix tokens."""
    p = tok.encode(prompt).ids if prompt else []
    s = tok.encode(" " + suffix.lstrip()).ids if suffix else []  # leading space: suffix starts a new word
    return p + [mask_id] * length + s


def to_text(tok, ids, mask_id, eos_id):
    """Readable text: [MASK] shown as '_', [EOS] as a blank line."""
    words = []
    for i in ids:
        if i == mask_id:
            words.append(" _")
        elif i == eos_id:
            words.append("\n\n")
        else:
            words.append(tok.decode([i]))
    return "".join(words)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", default="data/tokenizer.json")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--strategy", default="confidence", choices=["confidence", "random"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1, help="number of samples (seeds seed, seed+1, ...)")
    ap.add_argument("--show-steps", action="store_true")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    model = load_model(args.checkpoint)
    mask_id, eos_id = model.cfg.mask_id, tok.token_to_id("[EOS]")
    assert tok.token_to_id("[MASK]") == mask_id, "tokenizer and checkpoint do not match"

    for i in range(args.n):
        ids = make_input(tok, mask_id, args.length, args.prompt, args.suffix)
        states = generate(model, ids, args.steps, args.strategy, args.temperature, args.seed + i)
        print(f"===== sample {i + 1} (seed {args.seed + i}) =====")
        if args.show_steps:
            for s, state in enumerate(states):
                print(f"--- step {s}:\n{to_text(tok, state, mask_id, eos_id)}")
        else:
            print(to_text(tok, states[-1], mask_id, eos_id))
