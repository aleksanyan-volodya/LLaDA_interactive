"""
train the mask xith LLaDA loss.

python train.py --smoke            tiny model, 50 steps on CPU, a few seconds
python train.py                    full config (on Kaggle)
python train.py --smoke --overfit  check 2: overfit one small batch (also works without --smoke)
"""
import glob
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Config, MaskPredictor

# ---------------- config ----------------
SMOKE = "--smoke" in sys.argv
OVERFIT = "--overfit" in sys.argv

if SMOKE:
    MODEL = Config(dim=64, n_layers=2, n_heads=4, seq_len=64)  # short blocks keep it fast on CPU
    BATCH_SIZE = 8
    MAX_STEPS = 50
    WARMUP_STEPS = 10
    LR = 1e-3
    LOG_EVERY, EVAL_EVERY = 10, 25
else:
    MODEL = Config()
    BATCH_SIZE = 64          # 64 x 256 = 16k tokens per step
    MAX_STEPS = 60_000       # ~1B tokens. At 37k tok/s on the T4: ~7.4h, fits in TIME_BUDGET_H.
    WARMUP_STEPS = 1000
    LR = 6e-4                # not tuned, see the review notes
    LOG_EVERY, EVAL_EVERY = 100, 1000

# normal training settings
DECAY_FRAC = 0.2             # WSD: linear decay to 0 over the last 20% of steps as in the paper
WEIGHT_DECAY = 0.1           
EPS = 1e-3                   # t is drawn uniformly in [EPS, 1]
T_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]  # fixed t values only for the validation loss
N_VAL_BATCHES = 4
TIME_BUDGET_H = 8.0
CKPT_MINUTES = 20
N_SNAPSHOTS = 10

if OVERFIT:
    MAX_STEPS, WARMUP_STEPS, LOG_EVERY, LR = 300, 10, 25, 3e-3

# ---------------- paths ----------------
if os.path.exists("/kaggle"):
    found_data = glob.glob("/kaggle/input/**/train.npy", recursive=True)
    assert found_data, "no train.npy under /kaggle/input: attach the dataset made by prepare_data.py (Add Input)"
    DATA_DIR = os.path.dirname(found_data[0])
    OUT_DIR = "/kaggle/working"
    CKPT_SEARCH = ["/kaggle/working/ckpt.pt"] + glob.glob("/kaggle/input/**/ckpt.pt", recursive=True)
else:
    DATA_DIR = "data/smoke" if SMOKE else "data"
    OUT_DIR = "out/smoke" if SMOKE else "out"
    CKPT_SEARCH = [os.path.join(OUT_DIR, "ckpt.pt")]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"   # fp16 + GradScaler on the T4, plain fp32 on CPU


# ---------------- loss ----------------
def diffusion_loss(model, x):
    """LLaDA loss. Returns (weighted loss to optimize, unweighted CE on masked tokens for logging)."""
    B, L = x.shape
    t = EPS + (1 - EPS) * torch.rand(B, 1, device=x.device)       # one t per sequence
    masked = torch.rand(B, L, device=x.device) < t                # mask each token with probability t
    xt = torch.where(masked, MODEL.mask_id, x)
    logits = model(xt)
    ce = F.cross_entropy(logits[masked].float(), x[masked], reduction="none")
    weighted = (ce / t.expand(B, L)[masked]).sum() / (B * L)      # 1/t weighting, normalized by tokens
    return weighted, ce.mean()


def lr_at(step):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    decay_start = int(MAX_STEPS * (1 - DECAY_FRAC))
    if step < decay_start:
        return LR
    return LR * (MAX_STEPS - step) / (MAX_STEPS - decay_start)


@torch.no_grad()
def evaluate(model, val_x):
    """Masked-token CE at each fixed t, with the same masks every time (fixed generator)."""
    model.eval()
    gen = torch.Generator().manual_seed(1234)
    result = {}
    for t in T_GRID:
        total, count = 0.0, 0
        for x in val_x.split(BATCH_SIZE):
            masked = torch.rand(x.shape, generator=gen) < t
            x, masked = x.to(DEVICE), masked.to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=USE_AMP):
                logits = model(torch.where(masked, MODEL.mask_id, x))
            total += F.cross_entropy(logits[masked].float(), x[masked], reduction="sum").item()
            count += masked.sum().item()
        result[t] = total / count
    model.train()
    return result


def check_bidirectional(model):
    """Check that changing the last token must change the prediction at position 0."""
    model.eval()
    x = torch.randint(2, MODEL.vocab_size, (1, MODEL.seq_len), device=DEVICE)
    y = x.clone()
    y[0, -1] = MODEL.mask_id
    with torch.no_grad():
        diff = (model(x)[0, 0] - model(y)[0, 0]).abs().max().item()
    model.train()
    assert diff > 0, "CHECK 3 FAILED: attention is not bidirectional"
    print(f"check 3 OK: bidirectional (max logit change at position 0: {diff:.2e})")


# ---------------- checkpoints ----------------
def save(path, state):
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)  # atomic: a cut session never leaves a half-written file


def save_checkpoint(model, opt, scaler, step, log):
    save(os.path.join(OUT_DIR, "ckpt.pt"), {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "config": asdict(MODEL), "log": log,
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None,
    })
    print(f"saved ckpt.pt at step {step}")


# ---------------- main ----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    tok = Tokenizer.from_file(os.path.join(DATA_DIR, "tokenizer.json"))
    assert tok.get_vocab_size() == MODEL.vocab_size and tok.token_to_id("[MASK]") == MODEL.mask_id
    L = MODEL.seq_len
    train_ids = np.load(os.path.join(DATA_DIR, "train.npy"))
    val_ids = np.load(os.path.join(DATA_DIR, "val.npy"))
    train_blocks = train_ids[: len(train_ids) // L * L].reshape(-1, L)
    val_x = torch.from_numpy(val_ids[: N_VAL_BATCHES * BATCH_SIZE * L].astype(np.int64)).view(-1, L)
    print(f"device {DEVICE} | {len(train_blocks):,} train blocks | {len(val_x)} val blocks")

    model = MaskPredictor(MODEL).to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | {MODEL}")
    check_bidirectional(model)

    params = list(model.parameters())
    opt = torch.optim.AdamW([
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": WEIGHT_DECAY},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},  # norms: no decay
    ], lr=LR, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler(DEVICE, enabled=USE_AMP)

    step, log = 0, []
    found = [p for p in CKPT_SEARCH if os.path.exists(p)]
    if found and not OVERFIT:
        ck = torch.load(found[0], map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, log = ck["step"], ck["log"]
        torch.set_rng_state(ck["rng_cpu"])
        if DEVICE == "cuda" and ck["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(ck["rng_cuda"])
        print(f"resumed from {found[0]} at step {step}")

    if OVERFIT:
        # One sequence repeated 8 times: each copy gets different masks, so every step has signal.
        fixed_batch = torch.from_numpy(train_blocks[[0] * 8].astype(np.int64)).to(DEVICE)
        print("check 2: overfitting one fixed sequence, the loss should go close to 0")

    start = last_ckpt = last_log = time.time()
    snapshot_every = max(1, MAX_STEPS // N_SNAPSHOTS)
    while step < MAX_STEPS:
        if step % EVAL_EVERY == 0 and not OVERFIT:
            val = evaluate(model, val_x)
            val_mean = sum(val.values()) / len(val)
            print(f"step {step} | val CE {val_mean:.3f} | " + " ".join(f"t={t}:{v:.2f}" for t, v in val.items()))
            if step == 0:
                status = "OK" if abs(val_mean - math.log(MODEL.vocab_size)) < 1.0 else "FAILED"
                print(f"check 1 {status}: initial CE {val_mean:.3f}, expected about log(V) = {math.log(MODEL.vocab_size):.3f}")
            log.append({"step": step, "val": val_mean, "val_per_t": val})
            with open(os.path.join(OUT_DIR, "log.json"), "w") as f:
                json.dump(log, f)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        if OVERFIT:
            x = fixed_batch
        else:
            idx = torch.randint(len(train_blocks), (BATCH_SIZE,))
            x = torch.from_numpy(train_blocks[idx.numpy()].astype(np.int64)).to(DEVICE)

        with torch.autocast(DEVICE, dtype=torch.float16, enabled=USE_AMP):
            loss, ce = diffusion_loss(model, x)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        if step % LOG_EVERY == 0:
            now = time.time()
            tok_per_s = LOG_EVERY * x.numel() / (now - last_log)
            last_log = now
            print(f"step {step} | loss {loss.item():.3f} | masked CE {ce.item():.3f} | lr {lr_at(step - 1):.2e} | {tok_per_s:,.0f} tok/s")
            log.append({"step": step, "loss": loss.item(), "ce": ce.item(), "lr": lr_at(step - 1)})
        if OVERFIT:
            continue

        if step % snapshot_every == 0:
            save(os.path.join(OUT_DIR, f"model_step{step}.pt"),
                 {"model": model.state_dict(), "config": asdict(MODEL), "step": step})
        if time.time() - last_ckpt > CKPT_MINUTES * 60:
            save_checkpoint(model, opt, scaler, step, log)
            last_ckpt = time.time()
        if time.time() - start > TIME_BUDGET_H * 3600:
            print(f"time budget of {TIME_BUDGET_H}h reached")
            break

    if not OVERFIT:
        val = evaluate(model, val_x)
        print(f"final val CE {sum(val.values()) / len(val):.3f}")
        save_checkpoint(model, opt, scaler, step, log)


if __name__ == "__main__":
    main()
