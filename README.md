# README — Installation & Run Guide

Steps to go from a fresh `git clone` of `twin-classifier-demo` to running the
twin-classification demo. For the technical design, see
[`TECHNICAL.md`](./TECHNICAL.md).

> **Rights & privacy**: This repository ships neither images of real people nor
> trained weights. The default validation runs on rights-cleared material
> (synthetic images). A path to switch to real-photo twins (e.g. The Tacchi)
> is provided (`TWIN_DATASET=the_touch`), but **those images must be prepared
> locally by each user and are never included in the repository** (only
> aggregate figures such as accuracy are reported). `data/` and `*.pt` etc. are
> already in `.gitignore`.

---

## 0. Prerequisites

| Item | Detail |
|---|---|
| OS | Ubuntu 24.04 |
| GPU | AMD Ryzen AI MAX+ 395 (gfx1151, 48 GB unified memory) |
| ROCm | 7.2.x (`/opt/rocm`) |
| Python | 3.12 |
| Required env var | `HSA_OVERRIDE_GFX_VERSION=11.5.1` (so HIP recognizes gfx1151) |

The ROCm build of PyTorch (`torch 2.9.x+rocm7.2.1`) must already be installed in
the **Python 3.12 user site (`~/.local`)**. Do not use the default torch from
PyPI, since it does not recognize the iGPU.

If you use the VLM route, separately build a **ROCm-enabled llama.cpp**
(`-DGGML_HIP=ON`, `gfx1151`) and have the `llama-server` binary and the Nemotron
GGUF ready (see the article for model download links).

---

## 1. Clone and virtual environment

```bash
git clone https://github.com/kotetsuy/twin-classifier-demo.git
cd twin-classifier-demo

# Create a venv that inherits the user-site ROCm torch (do not reinstall torch)
python3 -m venv --system-site-packages .venv

# Add only the project dependencies
.venv/bin/pip install -r requirements.txt
```

## 2. Verify ROCm connectivity

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
bash scripts/verify_rocm.sh        # PASSED if gfx1151 is visible and matmul runs
```

> Even on ROCm, PyTorch's device name is `"cuda"` (`torch.cuda.is_available()`
> returns True).

---

## 3. Prepare data (2 routes)

Images of real people are not bundled, so you prepare them yourself. Use the
route that matches your purpose.

### route A — synthetic twins (for training/evaluation / reliable)

Generate rights-cleared, labeled A/B data locally. CNN training and evaluation
use this.

```bash
# Generate into data/synthetic/{train,val}/{A,B} (reproducible from seed, gitignored)
.venv/bin/python scripts/make_synthetic_twins.py --n-train 300 --n-val 20 --diff 0.7
#   Lower --diff (e.g. 0.4) makes A/B more similar and harder
```

### route C — real-photo twins (for the VLM explanation demo / optional)

Fetch CC-licensed real twin photos from Openverse (a gallery for the VLM's
qualitative demo).

```bash
.venv/bin/python scripts/fetch_cc_faces.py -q "identical twins" \
    --source wikimedia --license "cc0,pdm,by,by-sa" -n 40
#   Saved to data/raw/. Sources recorded in attribution.csv (handles CC-BY attribution)
```

### Build an A/B training set from real-photo twins (optional / local only)

A path to build A/B data of real twins (e.g. The Tacchi) locally. **The images
stay private and are never committed.** Four stages — collect → extract both
faces → manual labeling → split — produce `data/the_touch/{train,val}/{A,B}`.

```bash
.venv/bin/python scripts/fetch_the_touch.py            # Collect (ddgs image search -> face filter -> source CSV)
.venv/bin/python scripts/extract_faces.py              # Aligned 224x224 crops of both faces from duo photos
.venv/bin/python scripts/label_faces.py                # Manually label A=Takuya / B=Kazuya / skip in a GUI
.venv/bin/python scripts/build_ab_split.py --clean     # Generate data/the_touch/{train,val}
```

For details and dataset switching (`TWIN_DATASET`), see
[`TECHNICAL.md`](./TECHNICAL.md).

---

## 4. Train the fast CNN (route A data)

```bash
.venv/bin/python src/train_cnn.py --epochs 12
#   Saves the best weights to results/cnn.pt (gitignored / not committed)
#   Single-image prediction: .venv/bin/python src/train_cnn.py --predict path/to/face.png
```

> The CNN overfits when training data is scarce. `--n-train` of 200–300/class or
> more is recommended.

> **Dataset switching**: Prefixing with `TWIN_DATASET=the_touch` switches the
> reference data and weights used by training, evaluation, classification, and
> real-time mode all at once to the real-photo set (`data/the_touch` /
> `results/cnn_thetouch.pt`) (default is synthetic `synthetic`). Example:
> `TWIN_DATASET=the_touch .venv/bin/python src/train_cnn.py --epochs 12`

---

## 5. Start the Nemotron VLM as a resident server (separate terminal)

```bash
bash scripts/serve_nemotron.sh        # Exposes an OpenAI-compatible API on :8080
# Model/binary paths can be overridden via env vars:
#   LLAMA_SERVER, NEMOTRON_MODEL, NEMOTRON_MMPROJ
curl -s localhost:8080/health         # Ready when this returns {"status":"ok"}
```

Example of calling the VLM's judge / explain directly:

```bash
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, "src")
import nemotron_client as nc
D = "data/synthetic"
refs_a = [f"{D}/train/A/0000.png", f"{D}/train/A/0001.png"]
refs_b = [f"{D}/train/B/0000.png", f"{D}/train/B/0001.png"]
q = f"{D}/val/B/0000.png"
print("few-shot:", nc.judge(q, refs_a=refs_a, refs_b=refs_b))   # with examples = form (1)
print("zero-shot:", nc.judge(q))                                # no references (ill-posed)
print(nc.explain(q, refs_a=refs_a, refs_b=refs_b).rationale)    # with a rationale
PY
```

---

## 6. Evaluation (produce the method-comparison table)

```bash
.venv/bin/python src/evaluate.py --with-cnn
#   Compares fewshot / zeroshot / cnn on the same val set
#   Outputs results/eval.{json,csv} and confusion.png (confusion-matrix chart)
#   --limit N for a quick check, --refs-per-class K to tune the number of examples
#   --weights PATH swaps the cnn weights (evaluate with alternate weights without overwriting the default cnn.pt)

# Evaluate on the real-photo set (switch data + weights together):
TWIN_DATASET=the_touch .venv/bin/python src/evaluate.py --with-cnn --out results/thetouch
```

Example output (synthetic val n=40, train=300/class, diff=0.7):

| Method | Accuracy | ms/call (median) |
|---|---|---|
| cnn | 100% | ~4 |
| fewshot (VLM with examples) | 97.5% | ~2818 |
| zeroshot (VLM, no references) | 55% | ~764 |

---

## 7. Real-time classification (capture → classify → click)

Choose the screen-capture method with `--capture`: **X11=mss / Wayland (GNOME
etc.)=portal** (`auto` detects automatically). **Dry-run is the default** (logs
only, no clicks) for safety.

```bash
# Explanation demo mode (VLM with examples / form (1)). First do a dry-run to check region and click targets
.venv/bin/python src/realtime.py --mode explain --refs-dir data/synthetic/train \
    --region 100 100 400 400 --a-xy 300 800 --b-xy 900 800

# Speed mode (rapid-fire CNN). Real clicks require --no-dry-run
.venv/bin/python src/realtime.py --mode speed --no-dry-run --interval 0.1 --max-frames 30
```

- `--capture auto|mss|portal`: capture/operation method (default auto; portal on
  Wayland, mss on X11)
- `--region X Y W H`: capture region (default is the whole capture target = full monitor)
- `--a-xy` / `--b-xy`: click targets for an A/B decision (default is the left 1/3
  and right 2/3 of the region)
- `--mode explain` requires `--refs-dir` (containing `A/` and `B/`)

> On Wayland, mss returns all black and pynput clicks don't land, so the
> `portal` method (xdg-desktop-portal ScreenCast capture + RemoteDesktop
> injection) is used. A GNOME "screen share + control" permission dialog appears
> while running — approve it.

---

## 8. Launch the synthetic-twins demo all at once (tmux)

Launch/stop the GUI (synthetic val faces + A/B buttons) and the auto-player
(CNN 40 frames → VLM few-shot 6 frames) together via tmux.

```bash
./start_all.sh        # GUI + (if needed) VLM server + auto-play CNN(40)->VLM(6)
tmux attach -t twin-demo   # Live view (Ctrl-b 0/1/2 to switch, Ctrl-b d to detach)
./stop_all.sh         # Stop everything (an existing server is left running; add ./stop_all.sh --server to stop it too)
```

- The GUI shows CNN/VLM scores separately and prominently indicates the current
  mode (CNN/VLM).
- On Wayland, the "screen share + control" permission prompt appears once each
  for CNN and VLM while running. After approving, don't touch the mouse.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | Export `HSA_OVERRIDE_GFX_VERSION=11.5.1`. Verify with `verify_rocm.sh` |
| Can't connect to llama-server | Start `scripts/serve_nemotron.sh` and check `/health` returns ok |
| CNN val accuracy is 50% | Overfitting from insufficient data. Increase `--n-train` |
| realtime capture is all black | Wayland. Capture with `--capture portal` (auto handles it) and approve the share dialog |
| realtime doesn't actually click | Dry-run is the default; use `--no-dry-run`. Wayland uses portal injection, X11 uses pynput |
| Few results from Openverse | The face filter is strict. Adjust `--no-faces-only` or `--source wikimedia` |

For the implementation status of each milestone, see the "Status" section of the
project overview.
