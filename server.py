"""The web server: it sends the page to the browser, and answers the page's questions.

How the pieces talk:
  browser  --asks for the page-->  this server  --sends-->  static/index.html, app.js, style.css
  page     --fetch("/api/step")->  this server  --calls-->  model.py  --JSON back to the page-->

Run: double-click "Start Walkthrough.cmd" on Windows, or ./run.sh on macOS and Linux.
By hand: .venv/Scripts/python.exe server.py --open, then http://localhost:4470
"""
import sys
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn  # the program that listens on a port and passes requests to FastAPI
from fastapi import FastAPI, HTTPException  # FastAPI: maps web addresses to Python functions
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field  # describes the JSON we expect, and rejects anything else

import model  # importing it loads the model, which takes a few seconds

HERE = Path(__file__).parent  # the folder this file lives in
PORT = 4470

app = FastAPI()

# Any address starting with /static is a file from the static folder: the CSS, JS, fonts and logo.
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.middleware("http")
async def no_cache(request, call_next):
    """Runs on every request. Tells the browser not to keep old copies, because the page
    is edited between classes and a stale copy would show last week's version."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


# ---------------------------------------------------------------------------
# What the page sends. FastAPI checks each request against these shapes, and
# answers with an error by itself if a field is missing or out of range.
# ---------------------------------------------------------------------------
class TokenizeRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    use_template: bool = True  # wrap the prompt in the hidden chat template?


class StepRequest(BaseModel):
    ids: list[int] = Field(min_length=1, max_length=1024)  # every token so far
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)  # ge/le: at least / at most
    top_k: int = Field(default=4, ge=1, le=10)  # how many candidates to show
    seed: int = 0  # makes random picks repeatable


# ---------------------------------------------------------------------------
# The addresses. "@app.get('/x')" means: when the browser asks for /x, run this function.
# Whatever the function returns is sent back as JSON.
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """The page itself."""
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/info")
def info():
    """Facts about the model, for the page header."""
    return model.info()


@app.post("/api/tokenize")
def tokenize(req: TokenizeRequest):
    """Prompt in, tokens out."""
    return {"tokens": model.tokenize(req.prompt, req.use_template)}


@app.post("/api/step")
def step(req: StepRequest):
    """Tokens in, one pass of the model, next-token guesses out."""
    vocab_size = len(model.tokenizer)
    if any(i < 0 or i >= vocab_size for i in req.ids):
        raise HTTPException(400, "token id outside the vocabulary")

    start = time.perf_counter()
    result = model.forward(req.ids, req.temperature, req.top_k, req.seed)
    result["ms"] = round((time.perf_counter() - start) * 1000)  # the page shows how long it took
    return result


# This part runs only when you start the file directly (python server.py),
# not when another file imports it.
if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print(f"Ready: {url}")
    if "--open" in sys.argv:
        # uvicorn.run below never returns, so schedule the browser to open 1.5 s from now,
        # by which time the server is listening.
        threading.Timer(1.5, webbrowser.open, [url]).start()
    # 127.0.0.1 means only this computer can reach the server.
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
