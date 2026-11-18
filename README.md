# Mini-LLaDA

A small re-implementation of [LLaDA](https://arxiv.org/abs/2502.09992) (*Large Language Diffusion Models*),
trained from scratch on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

Instead of writing text left to right, LLaDA starts from a fully masked sequence and **unmasks** it step by step:
at each step the model predicts every masked token at once and keeps only some of its predictions.

- **Try it in your browser:** https://aleksanyan-volodya.github.io/llada/
- **Trained model:** https://huggingface.co/volodyaDuckDuck/mini-llada

## How it works

**Model** ([model.py](model.py)): a bidirectional Transformer (no causal mask), 33M parameters:
8 layers, 8 heads, dimension 512, RoPE, RMSNorm, SwiGLU, context of 256 tokens, 8,000-token BPE vocabulary.

**Training** ([train.py](train.py)): for each sequence, draw a masking ratio `t` uniformly in [0, 1], mask each token
with probability `t`, and train the model to recover the masked tokens. The loss is the cross-entropy on the masked
tokens only, weighted by `1/t`, as in the paper.

**Generation** ([sample.py](sample.py)): start with `[MASK]` everywhere (after the prompt), then repeat for `steps` steps:
predict all masked tokens, keep the `k` most confident predictions (or `k` random ones), leave the rest masked.
Since the model sees both sides, it can also fill the gap between a given beginning and a given ending.

## Results

Trained for 60,000 steps (about 1B tokens, batch 64 × 256) on one Kaggle T4 GPU, within one Kaggle session (under 8 hours).
Final validation cross-entropy on masked tokens: 1.74. The model writes readable short stories.

Limitation: generations very often start with "Once upon a time", like most of TinyStories.
More varied data or a bigger model trained longer should help.

## Usage

The project uses [uv](https://docs.astral.sh/uv/) (Python 3.13). Locally, PyTorch is the CPU version.

```bash
uv sync
```

**1. Data** ([prepare_data.py](prepare_data.py)): downloads TinyStories, trains the tokenizer, writes token ids to `data/`.

```bash
uv run prepare_data.py --smoke   # small test (validation file only, writes data/smoke/)
uv run prepare_data.py           # full dataset (~2.2 GB download)
```

**2. Training** ([train.py](train.py)): writes `ckpt.pt` (to resume), `model_step*.pt` snapshots and `log.json`.

```bash
uv run train.py --smoke          # tiny model, 50 steps on CPU, a few seconds (writes out/smoke/)
uv run train.py                  # full model: needs a GPU (we used Kaggle, see below)
```

On Kaggle: upload the output of `prepare_data.py` as a dataset, attach it to a notebook with a GPU and run `train.py`.
The script finds the data under `/kaggle/input`, saves to `/kaggle/working` and resumes from `ckpt.pt` if a run is cut.
Download the results into a `results_v1/` folder.

**3. Generate text** ([sample.py](sample.py)):

```bash
uv run sample.py results_v1/model_step60000.pt --prompt "Once upon a time" --steps 64 --temperature 0.8
uv run sample.py results_v1/model_step60000.pt --prompt "Tom had a dog." --suffix "They were best friends." --length 40
uv run sample.py results_v1/model_step60000.pt --show-steps   # print every denoising step
```

**4. Local web demo** ([app.py](app.py) + [app.html](app.html)): watch the unmasking live, compare settings side by side.
It loads the checkpoints from `results*/`.

```bash
uv run app.py                    # then open http://localhost:8000
```

**5. Export for the browser** ([export_onnx.py](export_onnx.py)): converts a checkpoint to ONNX (fp32 and int8),
the format used by the online demo.

```bash
uv run --with onnx --with onnxruntime --with onnxscript export_onnx.py results_v1/model_step60000.pt web_model/
```

## Files

| file | content |
|---|---|
| `model.py` | the mask predictor (Transformer) |
| `prepare_data.py` | dataset download, tokenizer training, tokenization |
| `train.py` | training loop with the LLaDA loss |
| `sample.py` | generation (reverse process) |
| `app.py`, `app.html` | local web demo |
| `export_onnx.py` | export to ONNX for the online demo |
| `notes.md` | my notes on the paper |
