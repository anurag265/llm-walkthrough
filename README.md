# LLM Walkthrough

A classroom demo of how an LLM answers a question, one click at a time, with a real model running on your own computer.

Students see the real tokens, the real token IDs, the real next-token probabilities and real attention, and watch the input grow by one token each pass through the network.

## Run

**Windows:** double-click `Start Walkthrough.cmd`. **macOS and Linux:** `./run.sh`.

The first run sets everything up, which needs internet once and takes a few minutes: a Python environment in `.venv`, then the packages, then the model (about 1 GB, into the Hugging Face cache). After that it starts offline and opens http://localhost:4470 by itself. Keep the window open while presenting; closing it stops the walkthrough.

Needs Python 3.11 or newer. No GPU needed; the model runs on the CPU.

By hand:

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe server.py --open
```

On macOS use `.venv/bin/python`, and install torch without the `--index-url` part.

## What is real and what is drawn

- **Real:** tokens, token IDs, vocabulary size, probabilities, logits, the pick, attention heat, timing.
- **Drawn:** the network picture. Its shape is schematic; only the input nodes' brightness comes from the model.
- **Simplified on purpose:** every pass reruns the whole input. Real servers keep a KV cache and compute only the newest token. The "KV-cache note" knob says so on screen.

## Controls

- One screen, no scrolling, sized for a 16:9 projector (checked at 1280x720 and 1920x1080): prompt and response on top, tokens and IDs in one row, network beside the probabilities. Text scales with the screen.
- Phones and tablets (below 900px wide): one scrolling column in pipeline order, a sticky header with the logo and a menu button, the controls in a full-screen menu, and Back and Next fixed at the bottom. Each step scrolls itself into view.
- Keys: N next (or Right arrow, Space), B back (or Left arrow), A auto-play, P pause, S skip to end, R reset, T show or fold hidden template tokens, H hide or show the controls.
- Hide the controls panel to give the content the full width; a small Next button stays at the top right.
- Click a probability bar at stage 4 to pick that token instead of the model's choice.
- Knobs: chat template on/off (resets), show hidden template tokens, spaces as a visible mark, temperature, seed, candidates shown, max new tokens, raw scores (logits), attention heat, KV-cache note, big text, dark theme, auto-play speed.

## Files

- `model.py` - loads `Qwen/Qwen2.5-0.5B-Instruct` on CPU; tokenize and one forward pass.
- `server.py` - FastAPI on 127.0.0.1:4470; `/api/info`, `/api/tokenize`, `/api/step`.
- `static/` - the page: `app.js` (state machine), `render.js` (stages), `diagram.js` (network picture).
- `static/brand/` - Prex Studio wordmark, favicon and the two brand fonts (self-hosted, so the page works offline).

To swap the model, change `MODEL_ID` in `model.py` to another Hugging Face causal LM with a chat template.

## License

Made by Anurag Kumar, [Prex Studio](https://prex.studio). Copyright 2026 Prex Studio.

Licensed under the [Apache License 2.0](LICENSE): free to use, change and share, including commercially, provided as is with no warranty. Keep the [`NOTICE`](NOTICE) file, which carries the credits, with any copy you share. The license does not cover the Prex Studio name, logo or wordmark.
