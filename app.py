"""Streamlit demo: watch the mask predictor unmask a story step by step.

uv run streamlit run app.py      (theme in .streamlit/config.toml)

Two modes: "Presentation" plays pre-configured scenes in one click, "Playground" exposes every setting.
"""
import glob
import html
import re
import time

import streamlit as st
from tokenizers import Tokenizer

from sample import generate_steps, load_model, make_input

TOKENIZER = "data/tokenizer.json"
CHECKPOINTS = "results*/model_step*.pt"

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

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Fraunces:opsz,wght@9..144,600&display=swap');
:root {
  --accent: #A78BFA; --accent2: #F472B6; --new: #FCD34D; --given: #7DD3FC;
  --text: #E7E9F3; --muted: #8A90AD; --card: rgba(19, 25, 50, 0.78); --line: rgba(167, 139, 250, 0.18);
}
.stApp { background:
  radial-gradient(60rem 30rem at 10% -10%, rgba(167, 139, 250, 0.16), transparent 60%),
  radial-gradient(50rem 30rem at 95% 0%, rgba(244, 114, 182, 0.10), transparent 60%),
  radial-gradient(60rem 40rem at 50% 120%, rgba(125, 211, 252, 0.07), transparent 60%), #0B0E1A; }
[data-testid="stMainBlockContainer"] { padding-top: 1.4rem; max-width: 1400px; }
[data-testid="stHeader"] { background: transparent; }

.eyebrow { color: var(--accent); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; }
.title { display: inline-block; font-family: 'Fraunces', Georgia, serif; font-size: 3.2rem; font-weight: 600; line-height: 1.1;
         margin: 0.25rem 0; background: linear-gradient(90deg, var(--accent), var(--accent2) 55%, var(--new));
         -webkit-background-clip: text; background-clip: text; color: transparent; }
.subtitle { color: var(--muted); font-size: 1.05rem; max-width: 46rem; line-height: 1.6; }
.chips { display: flex; flex-wrap: wrap; gap: 0.45rem; margin-top: 0.9rem; }
.chip { border: 1px solid var(--line); background: rgba(167, 139, 250, 0.08); color: var(--text);
        border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.8rem; white-space: nowrap; }

.st-key-scenes button { min-height: 5.6rem; align-items: flex-start; justify-content: flex-start; padding: 0.9rem 1rem;
  border-radius: 16px; border: 1px solid var(--line); background: var(--card); backdrop-filter: blur(8px);
  transition: transform 0.15s, border-color 0.15s, box-shadow 0.15s; }
.st-key-scenes button:hover { transform: translateY(-3px); border-color: var(--accent); box-shadow: 0 10px 30px rgba(167, 139, 250, 0.18); }
.st-key-scenes button[kind="primary"] { border-color: transparent; color: var(--text);
  background: linear-gradient(var(--card), var(--card)) padding-box,
              linear-gradient(120deg, var(--accent), var(--accent2), var(--new)) border-box;
  border: 1.5px solid transparent; box-shadow: 0 10px 34px rgba(167, 139, 250, 0.25); }
.st-key-scenes button > div, .st-key-scenes button > div > span, .st-key-scenes button [data-testid="stMarkdownContainer"] {
  width: 100%; justify-content: flex-start; text-align: left; }
.st-key-scenes button p { text-align: left; margin: 0; font-size: 0.85rem; color: var(--muted); line-height: 1.45; }
.st-key-scenes button p strong { display: block; color: var(--text); font-family: 'Fraunces', Georgia, serif;
  font-size: 1.12rem; font-weight: 600; margin: 0.15rem 0 0.1rem; }

.scene-head { display: flex; align-items: baseline; gap: 0.8rem; margin: 0.2rem 0 0.4rem; }
.scene-num { font-family: 'Fraunces', Georgia, serif; font-size: 2.6rem; line-height: 1; font-weight: 600;
  background: linear-gradient(120deg, var(--accent), var(--accent2)); -webkit-background-clip: text; background-clip: text; color: transparent; }
.scene-title { font-family: 'Fraunces', Georgia, serif; font-size: 1.9rem; color: var(--text); }
.scene-tag { color: var(--muted); font-size: 1rem; }
.note { border-left: 3px solid var(--accent); background: rgba(167, 139, 250, 0.07); border-radius: 0 12px 12px 0;
  padding: 0.7rem 1rem; color: var(--text); font-size: 0.98rem; line-height: 1.55; max-width: 60rem; }
.note b { color: var(--accent); font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; display: block; margin-bottom: 0.15rem; }
.legend { display: flex; flex-wrap: wrap; gap: 1.2rem; color: var(--muted); font-size: 0.85rem; align-items: center; min-height: 2.6rem; }
.legend span { display: inline-flex; align-items: center; gap: 0.4rem; }

.card { background: linear-gradient(180deg, rgba(167, 139, 250, 0.08), rgba(167, 139, 250, 0.01)), var(--card);
  backdrop-filter: blur(10px); border: 1px solid var(--line); border-radius: 20px; padding: 1.25rem 1.5rem 1rem;
  box-shadow: 0 20px 50px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.05); }
.card-head { display: flex; justify-content: space-between; align-items: center; gap: 0.8rem; flex-wrap: wrap; }
.card-title { font-family: 'Fraunces', Georgia, serif; font-size: 1.35rem; color: var(--text); }
.card .chips { margin-top: 0; }
.bar { height: 4px; border-radius: 4px; background: rgba(255, 255, 255, 0.06); margin: 0.9rem 0 1rem; overflow: hidden; }
.bar div { height: 100%; border-radius: 4px; background: linear-gradient(90deg, var(--accent), var(--accent2), var(--new)); }
.story { font-family: 'Literata', Georgia, serif; font-size: 1.22rem; line-height: 2.05; color: var(--text); min-height: 8rem; }
.card-foot { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.8rem; margin-top: 0.8rem;
  border-top: 1px solid var(--line); padding-top: 0.6rem; font-variant-numeric: tabular-nums; }

.m { display: inline-block; width: 1.5em; height: 0.95em; margin: 0 2px; vertical-align: -0.05em; border-radius: 6px;
  background: linear-gradient(90deg, rgba(167, 139, 250, 0.14), rgba(167, 139, 250, 0.4), rgba(167, 139, 250, 0.14));
  background-size: 300% 100%; background-attachment: fixed; animation: shimmer 2.2s linear infinite; }
@keyframes shimmer { from { background-position: 100% 0; } to { background-position: -200% 0; } }
.new { display: inline-block; background: var(--new); color: #1A1300; border-radius: 6px; padding: 0 2px;
  box-shadow: 0 0 18px rgba(252, 211, 77, 0.55); animation: pop 0.45s ease-out; }
@keyframes pop { from { transform: scale(1.35); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.given { color: var(--given); }
.eos { display: inline-block; color: var(--muted); font-family: 'Inter', sans-serif; font-size: 0.7rem;
  border: 1px solid var(--line); border-radius: 999px; padding: 0 0.5rem; margin: 0 0.3rem; vertical-align: middle; }

.how { display: grid; grid-template-columns: repeat(auto-fit, minmax(14rem, 1fr)); gap: 1rem; }
.how div { border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; background: rgba(167, 139, 250, 0.04); }
.how b { display: block; font-family: 'Fraunces', Georgia, serif; font-size: 1.05rem; margin-bottom: 0.3rem; color: var(--text); }
.how span { color: var(--muted); font-size: 0.92rem; }
.how .num { color: var(--accent); font-weight: 600; }
.credit { color: var(--muted); font-size: 0.8rem; text-align: center; margin-top: 2rem; opacity: 0.8; }
</style>"""

HOW_IT_WORKS = """<div class="how">
  <div><span class="num">1</span><b>Start from noise</b><span>The answer begins fully masked. Only the prompt is given.</span></div>
  <div><span class="num">2</span><b>Predict every mask at once</b><span>A bidirectional transformer guesses all masked tokens in parallel, using context on both sides.</span></div>
  <div><span class="num">3</span><b>Keep a few, remask the rest</b><span>The most confident (or random) guesses are kept. The others are masked again and predicted at the next step.</span></div>
  <div><span class="num">4</span><b>Repeat until nothing is masked</b><span>Fewer steps is faster, but tokens revealed together cannot see each other: quality drops.</span></div>
</div>"""

LEGEND = """<div class="legend">
  <span><span class="m"></span> masked</span>
  <span><span class="new">just revealed</span></span>
  <span><span class="given">given text</span></span>
</div>"""


@st.cache_resource
def get_model(path):
    return load_model(path)


@st.cache_resource
def get_tokenizer():
    return Tokenizer.from_file(TOKENIZER)


def step_of(path):
    m = re.search(r"step(\d+)", path)
    return int(m.group(1)) if m else 0


def chips(items):
    return '<div class="chips">' + "".join(f'<span class="chip">{html.escape(i)}</span>' for i in items) + "</div>"


# ---------------- rendering ----------------
def render_tokens(state, prev, given, mask_id):
    """Masks as shimmering pills, tokens revealed at this step glowing, given text in blue."""
    parts = []
    for i, t in enumerate(state):
        if t == mask_id:
            parts.append('<span class="m"></span>')
        elif t == EOS_ID:
            parts.append('<span class="eos">end</span><br>')
        else:
            text = html.escape(TOK.decode([t])).replace("\n", "<br>")
            space, word = (" ", text[1:]) if text.startswith(" ") else ("", text)  # keep the space outside the pill
            if i in given:
                parts.append(f'{space}<span class="given">{word}</span>')
            elif prev is not None and prev[i] == mask_id:
                parts.append(f'{space}<span class="new">{word}</span>')
            else:
                parts.append(text)
    return "".join(parts)


def card(run, state, prev, step, seconds):
    masks_left = state.count(run["mask_id"])
    done = 100 * (1 - masks_left / max(1, run["n_masks"]))
    foot = f"step {step} / {run['steps']}" + (" · done" if masks_left == 0 and step > 0 else "")
    return (
        f'<div class="card"><div class="card-head"><div class="card-title">{run["title"]}</div>{chips(run["chips"])}</div>'
        f'<div class="bar"><div style="width:{done:.1f}%"></div></div>'
        f'<div class="story">{render_tokens(state, prev, run["given"], run["mask_id"])}</div>'
        f'<div class="card-foot"><span>{foot}</span><span>{masks_left} masked</span><span>{seconds:.1f} s compute</span></div></div>'
    )


# ---------------- generation ----------------
def build_runs(specs, prompt, suffix, length, temperature):
    """specs: list of {ckpt, steps, strategy}. Returns the runs ready to render, or None if too long."""
    runs = []
    for k, s in enumerate(specs):
        model = get_model(s["ckpt"])
        ids = make_input(TOK, model.cfg.mask_id, length, prompt, suffix)
        if len(ids) > model.cfg.seq_len:
            st.error(f"Beginning + ending + length = {len(ids)} tokens, but the model handles at most {model.cfg.seq_len}.")
            return None
        name = "Final model" if s["ckpt"] == CKPTS[-1] else f"After {step_of(s['ckpt']):,} training steps"
        steps = min(s["steps"], length)
        runs.append({
            **s, "model": model, "ids": ids, "mask_id": model.cfg.mask_id, "steps": steps,
            "n_masks": ids.count(model.cfg.mask_id),
            "given": {i for i, t in enumerate(ids) if t != model.cfg.mask_id},
            "title": f"{'AB'[k]} · {name}" if len(specs) > 1 else name,
            "chips": [f"{steps} steps", s["strategy"], f"T = {temperature:g}"],
        })
    return runs


def show_runs(runs, temperature, seed, delay, play, key):
    """Draw the cards. If play, animate the generation; otherwise show the last result for these settings."""
    cols = st.columns([1, 6, 1])[1:2] if len(runs) == 1 else st.columns(len(runs))
    boxes = [c.empty() for c in cols]
    results = st.session_state.setdefault("results", {})

    if not play:
        for j, (run, box) in enumerate(zip(runs, boxes)):
            if key in results:
                state, seconds = results[key][j]
                box.html(card(run, state, state, run["steps"], seconds))
            else:
                box.html(card(run, run["ids"], None, 0, 0.0))
        return

    gens = [generate_steps(r["model"], r["ids"], r["steps"], r["strategy"], temperature, seed) for r in runs]
    prev, seconds = [None] * len(runs), [0.0] * len(runs)
    active, step = list(range(len(runs))), 0
    while active:
        for j in list(active):
            t0 = time.time()
            state = next(gens[j], None)
            seconds[j] += time.time() - t0
            if state is None:  # finished: show once more without highlights
                boxes[j].html(card(runs[j], prev[j], prev[j], runs[j]["steps"], seconds[j]))
                active.remove(j)
                continue
            boxes[j].html(card(runs[j], state, prev[j], step, seconds[j]))
            prev[j] = state
        step += 1
        time.sleep(delay)
    results[key] = list(zip(prev, seconds))


# ---------------- page ----------------
st.set_page_config(page_title="Mini-LLaDA", page_icon="✦", layout="wide")
st.html(CSS)

CKPTS = sorted(glob.glob(CHECKPOINTS), key=step_of)
if not CKPTS:
    st.error(f"No checkpoint found matching {CHECKPOINTS}")
    st.stop()
TOK = get_tokenizer()
EOS_ID = TOK.token_to_id("[EOS]")
final = get_model(CKPTS[-1])
n_params = sum(p.numel() for p in final.parameters()) / 1e6

hero, switch = st.columns([5, 1.3], vertical_alignment="top")
with hero:
    st.html(f"""
      <div class="eyebrow">Masked diffusion language model · LLaDA, in miniature</div>
      <div class="title">Mini-LLaDA</div>
      <div class="subtitle">Stories don't have to be written left to right. This model starts from a fully masked
      page and reveals the text a few tokens at a time, each step looking at both sides of every gap.</div>
      {chips([f"{n_params:.0f}M parameters", f"{final.cfg.n_layers} layers · {final.cfg.dim} dim",
              "bidirectional attention", "TinyStories · 8k BPE", f"{step_of(CKPTS[-1]):,} training steps on one T4"])}""")
with switch:
    mode = st.segmented_control("Mode", ["Presentation", "Playground"], default="Presentation",
                                label_visibility="collapsed", key="mode") or "Presentation"
st.write("")

if mode == "Presentation":
    def choose(i):  # runs before the page is redrawn, so the active button is highlighted right away
        st.session_state.scene, st.session_state.play = i, True

    st.session_state.setdefault("scene", 0)
    with st.container(key="scenes"):
        for i, (col, sc) in enumerate(zip(st.columns(len(SCENES)), SCENES)):
            col.button(f"{i + 1:02d}  \n**{sc['title']}**  \n{sc['tag']}", key=f"scene{i}", on_click=choose, args=(i,),
                       use_container_width=True, type="primary" if st.session_state.scene == i else "secondary")
    play = st.session_state.pop("play", False)
    sc = SCENES[st.session_state.scene]

    head, replay = st.columns([5, 1], vertical_alignment="bottom")
    with head:
        st.html(f"""<div class="scene-head"><span class="scene-num">{st.session_state.scene + 1:02d}</span>
          <span class="scene-title">{sc['title']}</span><span class="scene-tag">{sc['tag']}</span></div>
          <div class="note"><b>What to look at</b>{html.escape(sc['note'])}</div>""")
    with replay:
        play = st.button("▶  Play", type="primary", use_container_width=True) or play
    st.html(LEGEND)

    ckpt_of = {"final": CKPTS[-1], "early": CKPTS[0]}
    specs = [{**r, "ckpt": ckpt_of[r["ckpt"]]} for r in sc["runs"]]
    runs = build_runs(specs, sc["prompt"], sc["suffix"], sc["length"], sc["temperature"])
    if runs:
        show_runs(runs, sc["temperature"], sc["seed"], sc["delay"], play, key=("scene", st.session_state.scene))

else:
    with st.sidebar:
        st.markdown("### Story")
        prompt = st.text_input("Beginning", "Once upon a time")
        suffix = st.text_input("Ending (optional, for infilling)", "")
        length = st.slider("Tokens to generate", 8, 240, 128)
        st.markdown("### Sampling")
        temperature = st.slider("Temperature", 0.0, 1.5, 0.8, 0.05)
        seed = st.number_input("Seed", 0, 10_000, 0)
        delay = st.slider("Slow motion (s per step)", 0.0, 0.5, 0.0, 0.05)
        specs = []
        for label, default_steps in [("A", 128), ("B", 8)]:
            if label == "B" and not st.toggle("Compare with a second run"):
                break
            st.markdown(f"### Run {label}")
            specs.append({
                "ckpt": st.selectbox("Checkpoint", CKPTS, index=len(CKPTS) - 1, key=f"ckpt{label}",
                                     format_func=lambda p: f"training step {step_of(p):,}"),
                "steps": st.slider("Denoising steps", 1, 256, default_steps, key=f"steps{label}"),
                "strategy": st.segmented_control("Remasking", ["confidence", "random"], default="confidence",
                                                 key=f"strat{label}") or "confidence",
            })

    legend, button = st.columns([5, 1], vertical_alignment="center")
    legend.html(LEGEND)
    play = button.button("▶  Generate", type="primary", use_container_width=True)
    runs = build_runs(specs, prompt, suffix, length, temperature)
    if runs:
        key = ("playground", prompt, suffix, length, temperature, seed, tuple(tuple(s.values()) for s in specs))
        show_runs(runs, temperature, seed, delay, play, key)

st.write("")
with st.expander("How it works"):
    st.html(HOW_IT_WORKS)
st.html('<div class="credit">Re-implementation at small scale of <i>Large Language Diffusion Models</i> '
        '(Nie et al., 2025) · trained from scratch on TinyStories</div>')
