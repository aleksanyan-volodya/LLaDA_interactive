# Mini-LLaDA + Streamlit: todo

Goal: a small masked diffusion language model that generates readable text, and a Streamlit app that shows the unmasking step by step. No tests, no clean directory structure. Four files are enough: model.py, train.py, sample.py, app.py.

Rough schedule (presentation around November 5):
- Week 1: read the paper, steps 0 to 3
- Week 2: steps 4 to 5 (the main training run on Kaggle)
- Week 3: steps 6 to 8 (app, evaluation, deploying the weights)
- Week 4: step 9, prepare the presentation, no new development

Rule: if step 5 gives nothing by the end of week 2, we stop there, keep what we have, and present the paper with an honest analysis.

## Working rules (to give Claude at the start of every session)

- Claude runs no git commands (no add, commit, push, branch, reset, stash). The repository is managed only by me.
- I am the senior dev who reviews. Claude gives the code for one step only, stops, and waits for my validation before moving to the next one. We follow the order of this file.
- At the end of each step, Claude gives: what was written, the 3 to 5 specific points I should review, the command to test locally, and the expected result.
- Claude works locally. It tests only with the tiny configuration (smoke mode, see below). It never launches a long training run. Real runs happen on Kaggle, launched by me.
- If Claude is not sure about a detail (PyTorch API, hyperparameter, point from the paper), it says so instead of guessing.
- No refactoring, no new file, no new dependency without asking me first.
- Claude does not configure any cloud account. It provides the commands or the steps and I run them myself.

## Kaggle constraints (the same code must work locally and on Kaggle)

- One config at the top of train.py with two modes. Smoke mode: tiny model, 50 steps, a few seconds on CPU. Full mode: the real config. Same code, only the config changes.
- Environment detection: if the /kaggle folder exists, read data from /kaggle/input and write to /kaggle/working, otherwise use local paths.
- Use the torch and Python versions already installed on Kaggle. Before the code is written, I check these versions in a Kaggle notebook and give them to Claude. No pip install of heavy packages, no flash-attn (the T4 does not support it), no torch.compile until it has been tested on Kaggle.
- A single T4 GPU, fp16 mixed precision with GradScaler. The code must also run in fp32 on CPU locally.
- The script stops by itself after a time budget (for example 8h) and saves a full checkpoint: model, optimizer, scaler, step number, random generator state. It resumes automatically from a checkpoint if it finds one in the input data.
- Downloading TinyStories and training the tokenizer are done once. The result (tokens as a .npy or .bin file, tokenizer.json) is saved as a Kaggle dataset so it is not redone at every run.
- The code reaches Kaggle either through a dataset I upload, or by cloning my repository (internet enabled in the notebook).
- First pass on Kaggle: a run of 5 to 10 minutes to check compatibility and measure throughput in tokens per second, before launching the long one.
- Long runs are launched with Save Version (Save & Run All) so they run without a browser open.
- I retrieve the outputs (checkpoints, curves, samples) from Kaggle myself.

## 0. Decisions to make before coding

- [x] Training on Kaggle, a single T4. Note the session limit (about 9h) and the weekly GPU quota (about 30h), to be rechecked on the site.
- [ ] Data: TinyStories (Hugging Face, roneneldan/TinyStories). Simple text, small vocabulary, which gives the best chance of coherent text with little compute.
- [ ] Tokenizer: train an 8000-token BPE on TinyStories (tokenizers library). This avoids a huge 50k-token embedding that is useless for this corpus.
- [ ] Fixed sequence length: 256 tokens. Stories are concatenated with an EOS token between them and cut into blocks of 256.
- [ ] Add a special [MASK] token to the vocabulary.

## 1. Read the paper before any code

- [ ] Reread LLaDA and note precisely: the loss (masking with t uniform, 1/t weighting, loss only on masked positions), the absence of conditioning on t, the bidirectional attention, the generation algorithm and the remasking.
- [ ] Write these formulas in your own words in a notes.md file. They will also be used for the slides.

## 2. Model (model.py)

- [ ] Llama-style transformer but without a causal mask: RMSNorm, RoPE, SwiGLU, bidirectional attention (scaled_dot_product_attention with is_causal=False).
- [ ] Starting size: 8 layers, dimension 512, 8 heads, so about 30M parameters. Make these three values easy to change.
- [ ] No time input t. The model receives only the partially masked sequence.
- [ ] Output: logits over the whole vocabulary for each position.

## 3. Training loop (train.py) and basic checks

- [ ] For each batch: draw a different t per sequence (uniform in [eps, 1], with eps around 1e-3 to avoid dividing by nearly zero), mask each token with probability t, compute the cross-entropy on masked positions only, divide each term by t, then normalize by the number of tokens in the sequence.
- [ ] AdamW, short warmup then cosine decay, gradient clipping at 1.0, mixed precision (fp16 or bf16 depending on the GPU).
- [ ] Save a checkpoint every few minutes, with the ability to resume (Colab and Kaggle sessions get cut).
- [ ] Log the training loss and a validation loss. For validation, use a fixed grid of t values (for example 0.1, 0.3, 0.5, 0.7, 0.9) on a fixed validation set, otherwise the curve is too noisy to read.
- [ ] Check 1: at the very start, the unweighted loss should be close to log(8000), about 9. If not, there is a bug.
- [ ] Check 2: overfit a single small batch in a few hundred steps. If the loss does not drop close to zero, do not launch the real training run.
- [ ] Check 3: confirm that the attention mask is really bidirectional (changing a token on the right must change the prediction on the left).

## 4. Generation (sample.py)

- [ ] Generation function: start from a fully masked sequence (or a prompt left intact followed by masks), run N steps, at each step predict all masked positions, keep a certain number of them, remask the rest.
- [ ] Implement two remasking strategies: random, and confidence-based (keep the highest-probability predictions). A parameter to switch between them.
- [ ] Add a temperature (Gumbel noise or standard sampling). Without it, the confidence strategy tends to produce repetitive text, to be checked empirically.
- [ ] Make a function that also returns the intermediate states (the sequence at each step), for the animation in the app.
- [ ] Test with an early checkpoint (after a few thousand steps) to validate that the pipeline works, even if the text is bad.

## 5. Main training run

- [ ] Launch the long training run. Aim for a few GPU hours in total. I do not know in advance how many are needed for the text to become really readable with this model, this has to be found out empirically.
- [ ] Every 30 to 60 minutes, generate 5 samples from the latest checkpoint and read them. This is the real criterion, more than the loss curve.
- [ ] If the validation loss stagnates and the samples are noise, do not keep waiting. Try in this order: lower the learning rate, check the 1/t weighting, check the tokenizer, and only then increase the model size.
- [ ] Keep all intermediate checkpoints, spaced out (for example one every 10 percent of training). They will be useful to show how quality evolves in the presentation.
- [ ] Save the loss curves as an image file as early as possible.

## 6. Streamlit app (app.py)

- [ ] Load the model only once (st.cache_resource).
- [ ] Sidebar controls: prompt (optional), length to generate, number of steps, remasking strategy, temperature, random seed.
- [ ] Display area updated at each step: tokens still masked appear as greyed blocks, tokens already fixed as normal text, and those just unmasked are highlighted. This is the part that serves the demo.
- [ ] Button to compare two settings side by side with the same seed (for example 8 steps against 64 steps, or random against confidence).
- [ ] Option to choose an early or the final checkpoint, to show how quality evolves.
- [ ] Check that the app runs locally on your computer without a GPU (the model is small, generation should stay feasible on CPU, but this needs testing).

## 7. Light evaluation for the presentation

- [ ] Choose 10 fixed prompts and generate with different numbers of steps (for example 8, 32, 128) and both strategies. Keep the outputs in a file.
- [ ] Note qualitatively what changes (coherence, repetitions, length). No need for an official metric.
- [ ] Optional, only if everything else is done: train a small autoregressive model of the same size on the same data and compare the validation loss and the samples. Careful, comparing the losses of the two is tricky (the diffusion loss is a bound on the log-likelihood, not the same quantity), so only present it with that caveat.

## 8. Deploying the weights for a website

- [ ] Export the final weights from the checkpoint as safetensors (model only, no optimizer), with a config.json containing the model size, the sequence length, and the id of the [MASK] token. The tokenizer.json goes with it. Check that reloading gives exactly the same outputs as the checkpoint (same seed, same text).
- [ ] Test the weights in fp16 (about half the size, around 60 MB). Compare the samples with fp32 and keep fp16 only if quality does not visibly drop.
- [ ] Separate the inference code: a module that imports only model.py, the tokenizer and torch, with nothing tied to training or Kaggle. A generate(prompt, length, steps, strategy, temperature, seed) function that returns the final text and, optionally, the intermediate states.
- [ ] Measure the time of one generation on CPU (256 tokens, 32 steps). I do not know in advance whether it is one second or ten, and that decides whether a CPU server is enough for the site.
- [ ] Choose where to host. Google Cloud option: weights in a private Cloud Storage bucket, a small Cloud Run service that loads the weights at startup and exposes a POST /generate route. Simpler option: weights on the Hugging Face Hub and the app in a Space. I am not sure about the current free tiers of each, to be checked before choosing.
- [ ] The website calls the API and displays the result. Expect the first request after a cold start to be slow because of the weight loading.
- [ ] Do not put the weights in the git repository. They live in the bucket or the Hub and are downloaded when the service starts.
- [ ] Fallback if the cloud becomes too complicated: the Streamlit app runs locally with the weights downloaded from Kaggle. That is enough for the presentation.

## 9. Prepare the presentation (week 4, nothing new on the code side)

- [ ] Prepare a video or screenshots of the app as a backup. A live demo can crash.
- [ ] Proposed format for 15 minutes: about 8 minutes on the paper's method, 3 minutes on your implementation (size, data, curves), 2 minutes of the unmasking demo, 2 minutes of limitations and conclusion.
- [ ] Prepare an honest answer to the likely question: why is your model much smaller and worse than the one in the paper.

## Instruction to give Claude for the code

Simple code in four files (model.py, train.py, sample.py, app.py), with the configuration at the top of train.py and a checkpoint that can be reloaded. The three checks from step 3 must be coded before the first long training run. The working rules and the Kaggle constraints at the top of this file apply to every step.
