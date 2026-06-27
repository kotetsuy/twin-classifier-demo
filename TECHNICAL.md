# TECHNICAL — Technical Notes

Design decisions, implementation key points, and measured findings for
`twin-classifier-demo`. For setup steps see [`README.md`](./README.md).

---

## 1. Overall structure

```
input image ─┬─→ [face_align] eye-based 224x224 normalization (for real photos / no mirroring)
             │
             ├─→ [nemotron_client] VLM decision via llama-server (:8080, OpenAI-compatible)
             │        judge()  : fast A|B answer constrained by grammar
             │        explain(): A|B + a Japanese rationale (optional thinking trace too)
             │        * passing refs_a/refs_b enables few-shot matching (form (1))
             │
             └─→ [train_cnn] MobileNetV3-small (trained on synthetic data / sub-ms)

      [classify] unified IF that bundles backend=nemotron|cnn
      [evaluate] method comparison by accuracy, confusion matrix, latency (--weights to swap weights)
      [realtime] screen capture (mss/portal) → classify → click (pynput/portal injection)
      [data_config] TWIN_DATASET switches data + weights together (synthetic/the_touch)
```

Data generation/fetching is done by `scripts/make_synthetic_twins.py` (route A,
synthetic), `scripts/fetch_cc_faces.py` (route C / Openverse CC), and for
building real A/B data,
`fetch_the_touch.py → extract_faces.py → label_faces.py → build_ab_split.py`
(§9 / local only).

---

## 2. The core point: single-image A/B is not solvable as-is (ill-posed)

"Show one photo of a twin and guess A or B" is, **in principle, ill-posed unless
the model is told who A and B are**. "A"/"B" are empty labels with no content;
with one image there is no reference to go by.

The measurements show this clearly:

- **Zero-shot (no references) VLM accuracy ≈ 55%** (near chance). Moreover, the
  output degenerates to one side ("A"), and B recall drops to 10%. The confusion
  matrix becomes a single vertical column of "always answer A."

→ To make this **well-posed**, the model must be told "who A and B are," and
there are two ways to do so. That maps directly onto the "two-tier decision
backend."

| How to teach | Implementation | Characteristics |
|---|---|---|
| **Present examples on the spot** (in-context) | few-shot VLM (form (1)) | no training needed / produces explanations / slow |
| **Train it into weights** | CNN (route A data) | fast / accurate / requires labeled training |

---

## 3. Form (1): VLM matching with examples (few-shot)

The headline demo. A's example images and B's example images are embedded in the
prompt, then the query image is placed last and the model is asked "which is the
last one?" The OpenAI-compatible API lists multiple images in the content array:

```
content = [
  {text: "person A:"}, {image: A example1}, {image: A example2},
  {text: "person B:"}, {image: B example1}, {image: B example2},
  {text: "Is the last image A or B? State your reasoning"},
  {image: query image},          # <- always last (matches "the LAST image" in the prompt)
]
```

This is implemented in `nemotron_client._ref_message()`. `judge()/explain()`
take `refs_a, refs_b` (lists of images each); when given, they run few-shot; when
omitted, they run on a single image (the ill-posed baseline). `classify()` also
passes refs through.

---

## 4. Measured findings from driving Nemotron 3 Nano Omni via llama-server

Model: `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning` (UD-Q4_K_XL ≈ 22 GB) +
`mmproj-F16` (vision). ROCm llama.cpp's `llama-server` resident at
`-ngl 99 -c 8192`.

### 4.1 Separating the reasoning model's output

This model returns its **thinking in `reasoning_content` and the final answer in
`content`** separately.

- **judge (fast decision)**: with reasoning ON, constraining with grammar makes
  the answer appear in `reasoning_content` while `content` is empty. → Turn off
  thinking with `chat_template_kwargs={"enable_thinking": False}` so A/B is
  emitted directly into `content`. `grammars/ab.gbnf`
  (`root ::= "A" | "B"`) forces the output to a single token.
- **explain (explanation)**: with reasoning ON, **the thinking fails to converge
  and hits `max_tokens` with `content` still empty** (thinking exceeds 3000
  characters, `finish_reason: length`). Instructions like "in 3 sentences or
  fewer" are ignored during the thinking phase. → explain also has **thinking
  OFF by default**. A concise, non-empty rationale appears in `content` in about
  2 seconds. `think=True` is an option for obtaining traces (noted: it may not
  reach a conclusion).

### 4.2 Latency

- Single-image judge: ~0.8s/call
- Few-shot judge (2 examples + 1 query = 3 images): ~2.8s/call (grows in
  proportion to the number of images)

Image inputs are passed as a JPEG data URL via `_to_data_url()` from
path / numpy(BGR) / PIL.

---

## 5. The fast CNN (trained on route A data)

The classification head of `MobileNetV3-small` (ImageNet pretrained) is replaced
with a 2-class head.

- **Horizontal-flip augmentation is forbidden.** Twin-discrimination cues are
  subtle left/right asymmetries (moles, eyebrow angle, hairline), so mirroring
  erases the cues. This is the same idea as `face_align`'s "similarity transform
  only / no mirroring (det(R)=s²>0)." Augmentation is limited to slight
  brightness/contrast jitter.
- Inference helpers `load_classifier()/predict_label()` are shared across
  `classify`/`evaluate`/`realtime`. `classify` loads the model once and caches it.

### Effect of data volume (measured)

The amount of synthetic data directly governs generalization:

| train images/class | Behavior |
|---|---|
| 40 | train_loss → 0 but **val 50%** (overfitting / rote memorization; no generalization) |
| 300 | **val 100% by epoch 2** (learns the discriminative features) |

→ You can observe a clean, data-volume-dependent reversal: "with little data the
VLM (few-shot) is stronger, and with enough data the CNN wins decisively."

---

## 6. Evaluation method (`evaluate.py`)

Each method's predict is applied to the entire ImageFolder val set, aggregating
accuracy, per-class accuracy, confusion matrix, and latency (ms/call, median).
Few-shot examples are **taken from train**, so they don't leak with val. Outputs
`results/eval.{json,csv}` and `confusion.png`.

The default of `--data` and the default of cnn's `--weights` follow the
`TWIN_DATASET` switch (§9). `--weights` lets you evaluate weights trained on
other data **without overwriting** the default weights (`results/cnn.pt`).

### 3-method comparison (synthetic val n=40, train=300/class, diff=0.7, seed=0, 2 examples/class)

| Method | Accuracy | A recall | B recall | ms/call (median) | Nature |
|---|---|---|---|---|---|
| cnn | **100%** | 100% | 100% | **~4** | fast + accurate. requires labeled training |
| fewshot VLM | 97.5% | 95% | 100% | ~2818 | no training / with explanation. ~700x slower |
| zeroshot VLM | 55% | 100% | 10% | ~764 | breaks down without references (demonstrates ill-posedness) |

> The median latency excludes the CNN's first model load and reflects
> steady-state inference (the mean includes it). As a weakness of few-shot,
> A recall of 75–95% misses can occur (a bias toward "salient feature = B").

---

## 7. Two data-acquisition routes (design decision)

Research-grade real-photo twin datasets (ND-TWINS-2009-2010, etc.) were
initially considered, but were passed over because they require an institutional
signature, are 250 GB, and are heavy for an individual demo. Instead, a **C+A
two-pronged approach**:

- **route A (synthetic / `make_synthetic_twins.py`)**: A and B share the same
  "genome," and only the cues the HANDOFF lists (moles, eyebrow angle, hairline)
  differ stably. Each image adds shooting jitter (rotation, translation, scale,
  brightness, background, noise) to create within-person variation. Difficulty
  is tunable with `--diff`. Because it's **fully rights-cleared, labeled, and
  seed-reproducible**, it's ideal as the primary data for training and
  evaluation. Photorealism doesn't matter (cartoonish is fine).
- **route C (real / `fetch_cc_faces.py`)**: narrows Openverse to CC0/PDM
  (+ optionally CC-BY), uses face detection (reusing mediapipe FaceLandmarker)
  to exclude non-faces, and records sources in `attribution.csv`. With
  `--source wikimedia` you can hit real twins. However, since these are
  "two-people-per-photo / different pairs / few in number," they are **unsuited
  to supervised A/B**, so they're treated strictly as a qualitative
  explanation gallery for the VLM. Hitting originals directly returns 429, so
  fetch via Openverse thumbnails.
- **To build supervised A/B for a specific real twin pair** (e.g. The Tacchi),
  use a separate, local-only pipeline (`fetch_the_touch` family /
  `TWIN_DATASET=the_touch`). See §9 for details.

> mediapipe is a detector for real-photo faces, so it doesn't detect synthetic
> faces (route A). Synthetic data is already normalized at generation time, so
> face_align is unnecessary.

### Synthetic-twin generation/training (minimal commands)

```bash
# 1) Generate: data/synthetic/{train,val}/{A,B} (seed-reproducible / gitignored)
python scripts/make_synthetic_twins.py --n-train 300 --n-val 20 --diff 0.7 --seed 0
# 2) Train: results/cnn.pt (follows the default TWIN_DATASET=synthetic)
python src/train_cnn.py --epochs 12
```

- `--n-train/--n-val`: images per class. The CNN is stable at 200–300/class or
  more (see "effect of data volume" in §5).
- `--diff`: A/B difference (`0<diff<=1`; smaller is harder). At 0.7 the CNN is
  near 100%; lowering it reveals differences between methods.
- `--size`: output resolution (default 224). `--seed`: for reproducibility.

---

## 8. Going real-time (`realtime.py`) and supporting real hardware (Wayland)

Region capture → `classify` → map A/B to the click targets in `--a-xy`/`--b-xy`
and click. Two modes:

- **explain** (default): decide with the few-shot VLM and show the rationale. You
  control the pace (~3s/frame).
- **speed**: decide with the CNN at sub-ms and fire rapidly.

For safety, **dry-run is the default** (logs, no clicks). Real clicks require
`--no-dry-run`.

### 8.1 Capture/operation backend abstraction (`src/screen_capture.py`)

When the real machine is **GNOME on Wayland**, a naive X11-assuming
implementation gets stuck two ways:

- **`mss` capture is all black.** It can't capture the compositor's screen via
  XWayland.
- **`pynput` (XTEST) clicks don't reach the window** (only cursor movement does).

So an abstraction was added to switch capture/operation method via `--capture`:

| backend | Capture | Click | Use |
|---|---|---|---|
| `mss` | mss (X11 root grab) | pynput (XTEST) | X11/Xorg session |
| `portal` | xdg-desktop-portal **ScreenCast** + PipeWire (`pipewiresrc`) | portal **RemoteDesktop** injection | Wayland (GNOME etc.) |

`auto` (default) selects automatically by session type. `portal` opens
ScreenCast and RemoteDesktop in the **same portal session**, so capture and
injection share the same coordinate system and stay consistent (injection uses
`NotifyPointerMotionAbsolute` + `NotifyPointerButton`, `BTN_LEFT=0x110`). At
runtime a GNOME "screen share + control" permission dialog appears.

### 8.2 Mutter's pitfall: self-reported coordinates aren't trustworthy

Mutter **ignores app-specified window positions**, and the coordinates reported
by `Tk.winfo_rootx` are also far off from the real position (specified +200 →
measured ~1077, varying each launch). The countermeasure exploits the fact that
**capture and injection share the same coordinate system**: the click targets
and face region are **calibrated by color-detecting them from the captured
screen** (`scripts/play_twins.py`). Tk's self-reported coordinates are not used.

### 8.3 Demo orchestration

- `scripts/twin_stream.py`: the Tk stimulus stage (face + A/B buttons + magenta
  frames for detection). Clicking scores and advances. `--loop` for continuous
  playback. Scores are shown **split into CNN / VLM** and the current mode is
  shown prominently (the mode is signaled by `play_twins` through a state file).
- `scripts/play_twins.py`: the auto-player core that calibrates by color
  detection → decides → injects clicks.
- `scripts/mouse_test.py`: debugging for window capture + injected clicks.
- `start_all.sh` / `stop_all.sh`: launch/stop GUI + (if needed) VLM server +
  auto-play **CNN(40)→VLM few-shot(6)** all at once via tmux.

Measured on real hardware (synthetic val / GNOME Wayland): CNN speed **40/40**,
VLM few-shot explain **6/6** completed successfully.

---

## 9. Dataset switching (`TWIN_DATASET`) and real-photo twins (route the_touch)

### 9.1 The single switch `TWIN_DATASET` (`src/data_config.py`)

The tree **defaults to synthetic**. A single env var `TWIN_DATASET` (default
`synthetic`) **switches both the data and the CNN weights** referenced by
training, evaluation, classification, and real-time, all at once.

| `TWIN_DATASET` | Data | Weights |
|---|---|---|
| `synthetic` (default) | `data/synthetic` | `results/cnn.pt` |
| `the_touch` | `data/the_touch` | `results/cnn_thetouch.pt` |

```bash
python src/evaluate.py --with-cnn                          # synthetic (default)
TWIN_DATASET=the_touch python src/evaluate.py --with-cnn   # real photos (switch data + weights together)
TWIN_DATASET=the_touch python src/realtime.py --mode speed --capture portal --no-dry-run ...
```

The `--data`/`--weights` defaults of `train_cnn` / `evaluate`, and the weights of
`classify`/`realtime(speed)`, follow the switch (override with explicit
`--data`/`--weights`). A new dataset can be added in one line in `REGISTRY`.

### 9.2 How to build the real-photo A/B training set (local only / private)

Build the A/B of real twins (e.g. The Tacchi) **locally only**. **The images
are neither published nor committed, and only aggregate figures are reported**
(`data/` is gitignored). A 4-stage pipeline:

1. `fetch_the_touch.py`: ddgs (DuckDuckGo) image search → face filter → source
   CSV. Sets aside noise (English hairstyle photos, etc.) by title relevance.
2. `extract_faces.py`: since most are duo photos, use a landmarker with an
   increased `num_faces` to crop **both faces** with the same similarity
   transform as `face_align` (no mirroring), aligned to 224x224.
3. `label_faces.py`: Tk GUI. Shows the aligned crop alongside the highlighted
   corresponding face in the original photo, and manually labels **A=Takuya (mole
   beside the nose) / B=Kazuya / skip** (identical twins are hard to tell apart,
   so human labeling is essential). Supports resume.
4. `build_ab_split.py`: `labels.csv` → `data/the_touch/{train,val}/{A,B}`.

Then **train and evaluate on the real photos** directly (`TWIN_DATASET=the_touch`
switches data + weights):

```bash
# Train: learn from data/the_touch and save to results/cnn_thetouch.pt (follows the switch default)
TWIN_DATASET=the_touch python src/train_cnn.py --epochs 12
# Evaluate: compare 3 methods on the real-photo val (synthetic results preserved in results/)
TWIN_DATASET=the_touch python src/evaluate.py --with-cnn --out results/thetouch
```

> Real photos are few in number, so the CNN overfits easily (§9.3). The VLM
> few-shot works without training, but doesn't achieve accuracy on real twins
> (measured in §9.3).

### 9.3 Evaluation results for the real The Tacchi (val n=21 / local only)

Methods that won decisively on synthetic data break down on real identical twins:

| Method | Accuracy | A (Takuya) recall | B (Kazuya) recall | ms/call |
|---|---|---|---|---|
| fewshot VLM | 47.6% | 30% | 64% | ~2851 |
| zeroshot VLM | 47.6% | 100% | 0% | ~770 |
| CNN (trained on real photos) | 71.4% | 40% | 100% | ~4 |

- **The few-shot VLM that scored 97.5% on synthetic data is 47.6% ≈ chance on
  real ones.** The *stably effective differences* like the seeded "moles /
  eyebrows" don't work on real twins (at this resolution / count). Zeroshot, as
  with synthetic, collapses to "all A."
- **CNN 71.4%** is also, internally, a bias predicting B for 17 of 21 val images
  (A recall 40%) — a product of small-data overfitting.
- With only 21 val images, the numbers are noisy. Even so, the trend is clear:
  **the real The Tacchi are hard even for AI to tell apart.**
- Conversely, this is also evidence that **synthetic data cannot fully reproduce
  the real difficulty (subtle, unstable differences).**

---

## 10. Known limitations and room for extension

- Since this is still-image classification, when twins **synchronize their
  movements** the cues vanish. Nemotron 3 Nano Omni also supports video input,
  so there is room for improvement with temporal features.
- The few-shot VLM's accuracy depends on the quality of the examples. It varies
  with the number of examples (`--refs-per-class`) and how they are selected.
- Synthetic data is too cleanly separable (CNN 100% at `diff=0.7`). Lowering
  `--diff` approaches realistic difficulty and makes the differences between
  methods more visible.
