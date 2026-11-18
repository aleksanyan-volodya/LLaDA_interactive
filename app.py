"""Web demo without Streamlit: watch the mask predictor unmask a story step by step.

uv run app.py                         then open http://localhost:8000
uv run app.py --host 0.0.0.0          (reachable from other machines on the network)

Only the standard library is used: this file serves app.html and streams the generation.
Two modes in the page: "Presentation" plays pre-configured scenes in one click, "Playground" exposes every setting.

Routes:
  GET  /          the page (app.html)
  GET  /info      model size, checkpoints, scenes, mask/eos ids and the vocabulary (id -> text)
  POST /prepare   JSON settings -> one JSON line describing the runs (title, chips, input ids)
  POST /generate  same, then one JSON line per denoising step, sent while the model runs
"""
import argparse
import functools
import glob
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tokenizers import Tokenizer

from sample import generate_steps, load_model, make_input

TOKENIZER = "data/tokenizer.json"
CHECKPOINTS = "results*/model_step*.pt"
PAGE = "app.html"

# Each scene: shared story settings + one or two runs. ckpt is "final" or "early" (first snapshot).
# Seeds were picked by reading samples, so the live demo shows a clear example.
SCENES = [
    dict(title="Watch it write", tag="A story appears out of noise",
         note="Every grey pill is a masked token. Watch the order: punctuation and frequent words "
              "(\"there was a\") appear first, the details of the story come last.",
         prompt="Once upon a time", suffix="", length=96, temperature=0.8, seed=0, delay=0.06,
         runs=[dict(ckpt="final", steps=96, strategy="confidence")]),
    dict(title="Fill the gap", tag="Context on both sides",
         note="The beginning and the ending are given. The middle has to lead to an ending the model "
              "already knows: easy with bidirectional attention, impossible for a left-to-right model.",
         prompt="Once upon a time, there was a little dog named Max. Max was very sad because he had no friends.",
         suffix="From that day on, Max and the cat were best friends.",
         length=40, temperature=0.8, seed=1, delay=0.12,
         runs=[dict(ckpt="final", steps=40, strategy="confidence")]),
    dict(title="Fewer steps", tag="96 steps against 8",
         note="With 8 steps, 12 tokens are revealed at once and cannot see each other. "
              "Same model, same seed: only the number of steps changes.",
         prompt="Once upon a time", suffix="", length=96, temperature=0.8, seed=1, delay=0.0,
         runs=[dict(ckpt="final", steps=96, strategy="confidence"), dict(ckpt="final", steps=8, strategy="confidence")]),
    dict(title="Training", tag="Early checkpoint against final",
         note="The same sampler on two checkpoints of the same run. Early on, the model already writes "
              "sentences but loses the thread and repeats itself.",
         prompt="Once upon a time", suffix="", length=96, temperature=0.8, seed=4, delay=0.0,
         runs=[dict(ckpt="early", steps=96, strategy="confidence"), dict(ckpt="final", steps=96, strategy="confidence")]),
    dict(title="Remasking", tag="Confidence against random",
         note="Confidence fills the story roughly left to right, keeping the safest guesses first, and can "
              "get stuck on a safe phrase. Random reveals tokens everywhere at once, as in the theory of the paper.",
         prompt="Once upon a time", suffix="", length=96, temperature=0.8, seed=2, delay=0.0,
         runs=[dict(ckpt="final", steps=96, strategy="confidence"), dict(ckpt="final", steps=96, strategy="random")]),
]

get_model = functools.cache(load_model)  # each checkpoint is loaded once, on first use


def step_of(path):
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else 0


def build_runs(req):
    """req: the JSON settings sent by the page. Raises ValueError with a message for the page if they are wrong."""
    prompt, suffix = str(req.get("prompt", "")), str(req.get("suffix", ""))
    length, temperature = int(req["length"]), float(req["temperature"])
    if not 1 <= length or not 0 <= temperature <= 2:
        raise ValueError("Length must be at least 1 and temperature between 0 and 2.")
    specs = req["runs"][:2]
    runs = []
    for k, s in enumerate(specs):
        path = {"final": CKPTS[-1], "early": CKPTS[0]}.get(s["ckpt"], s["ckpt"])
        if path not in CKPTS:  # never torch.load a path that came from the network
            raise ValueError(f"Unknown checkpoint {s['ckpt']}.")
        if s["strategy"] not in ("confidence", "random"):
            raise ValueError(f"Unknown strategy {s['strategy']}.")
        model = get_model(path)
        ids = make_input(TOK, model.cfg.mask_id, length, prompt, suffix)
        if len(ids) > model.cfg.seq_len:
            raise ValueError(f"Beginning + ending + length = {len(ids)} tokens, "
                             f"but the model handles at most {model.cfg.seq_len}.")
        name = "Final model" if path == CKPTS[-1] else f"After {step_of(path):,} training steps"
        steps = max(1, min(int(s["steps"]), length))
        runs.append({
            "model": model, "ids": ids, "steps": steps, "strategy": s["strategy"],
            "title": f"{'AB'[k]} · {name}" if len(specs) > 1 else name,
            "chips": [f"{steps} steps", s["strategy"], f"T = {temperature:g}"],
        })
    return runs


def stream(runs, temperature, seed, delay):
    """Advance all runs one step at a time, so side-by-side runs move together. Yields one dict per run and step."""
    gens = [generate_steps(r["model"], r["ids"], r["steps"], r["strategy"], temperature, seed) for r in runs]
    seconds = [0.0] * len(runs)
    active, step = list(range(len(runs))), 0
    while active:
        for j in list(active):
            t0 = time.time()
            state = next(gens[j], None)
            seconds[j] += time.time() - t0
            if state is None:
                active.remove(j)
                yield {"run": j, "done": True, "seconds": seconds[j]}
            else:
                yield {"run": j, "step": step, "state": state, "seconds": seconds[j]}
        step += 1
        time.sleep(delay)


class Handler(BaseHTTPRequestHandler):
    def send(self, code, kind, body):
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def line(self, obj):
        self.wfile.write((json.dumps(obj) + "\n").encode())
        self.wfile.flush()

    def do_GET(self):
        if self.path == "/":
            with open(PAGE, "rb") as f:
                self.send(200, "text/html; charset=utf-8", f.read())
        elif self.path == "/info":
            self.send(200, "application/json", INFO)
        else:
            self.send(404, "text/plain", b"not found")

    def do_POST(self):
        if self.path not in ("/prepare", "/generate"):
            return self.send(404, "text/plain", b"not found")
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            runs = build_runs(req)
            temperature, seed = float(req["temperature"]), int(req["seed"])
            delay = min(1.0, max(0.0, float(req.get("delay", 0))))
        except (ValueError, KeyError, TypeError) as e:
            return self.send(400, "application/json", json.dumps({"error": str(e)}).encode())

        # No Content-Length: the answer is streamed line by line and ends when the connection closes.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            self.line({"runs": [{k: r[k] for k in ("title", "chips", "steps", "ids")} for r in runs]})
            if self.path == "/generate":
                for event in stream(runs, temperature, seed, delay):
                    self.line(event)
        except (BrokenPipeError, ConnectionResetError):  # the page stopped listening (other scene clicked)
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    CKPTS = sorted(glob.glob(CHECKPOINTS), key=step_of)
    if not CKPTS:
        raise SystemExit(f"No checkpoint found matching {CHECKPOINTS}")
    TOK = Tokenizer.from_file(TOKENIZER)
    final = get_model(CKPTS[-1])
    INFO = json.dumps({
        "checkpoints": [{"path": p, "step": step_of(p)} for p in CKPTS],
        "params_m": sum(p.numel() for p in final.parameters()) / 1e6,
        "layers": final.cfg.n_layers, "dim": final.cfg.dim,
        "mask_id": final.cfg.mask_id, "eos_id": TOK.token_to_id("[EOS]"),
        "vocab": [TOK.decode([i]) for i in range(TOK.get_vocab_size())],
        "scenes": SCENES,
    }).encode()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mini-LLaDA demo on http://localhost:{args.port}  ({len(CKPTS)} checkpoints, Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
