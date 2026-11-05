// The walkthrough as a state machine. Next moves one stage; Back restores the previous snapshot.
// Stages: 0 prompt, 1 tokens, 2 ids, 3 forward pass, 4 softmax, 5 detokenize, 6 append.
const $ = (id) => document.getElementById(id);
const K = {};
let S, history = [], busy = false, autoplay = false, info = null;

function fresh() {
  return { stage: 0, pass: 0, tokens: [], result: null, pick: null, response: [],
    done: false, status: "", info, temp: 0, seed: 0, animate: false };
}

function readKnobs() {
  K.template = $("k-template").checked;
  K.hidden = $("k-hidden").checked;
  K.spaces = $("k-spaces").checked;
  K.temp = +$("k-temp").value;
  K.seed = +$("k-seed").value || 0;
  K.topk = +$("k-topk").value;
  K.max = +$("k-max").value;
  K.logits = $("k-logits").checked;
  K.attn = $("k-attn").checked;
  K.cache = $("k-cache").checked;
  K.speed = +$("k-speed").value;
  $("o-temp").textContent = K.temp;
  $("o-topk").textContent = K.topk;
  $("o-max").textContent = K.max;
  $("o-speed").textContent = K.speed + "s";
  document.body.classList.toggle("big", $("k-big").checked);
  document.body.classList.toggle("dark", $("k-dark").checked);
}

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : {});
  if (!res.ok) throw new Error(`${path}: ${res.status} ${await res.text()}`);
  return res.json();
}

async function advance() {
  const s = S;
  s.animate = false;
  s.tokens.forEach((t) => (t.fresh = false));
  if (s.stage === 0) {
    const prompt = $("prompt").value.trim();
    if (!prompt) throw new Error("Type a prompt first");
    const r = await api("/api/tokenize", { prompt, use_template: K.template });
    s.tokens = r.tokens;
    s.stage = 1;
  } else if (s.stage === 1) {
    s.stage = 2;
  } else if (s.stage === 2) {
    s.temp = K.temp;
    s.seed = K.seed + s.pass;  // a new dice roll each pass, same rolls on every replay
    s.result = await api("/api/step", {
      ids: s.tokens.map((t) => t.id), temperature: s.temp, top_k: 10, seed: s.seed,
    });
    s.seed = K.seed;
    s.pick = s.result.pick;
    s.pass += 1;
    s.stage = 3;
    s.animate = true;
  } else if (s.stage === 3 || s.stage === 4) {
    s.stage += 1;
  } else if (s.stage === 5) {
    append(s);
  } else if (s.stage === 6) {
    s.stage = 2;
  }
}

function append(s) {
  const isEnd = s.result.eos_ids.includes(s.pick.id);
  s.tokens.push({ id: s.pick.id, text: s.pick.text, hidden: false, generated: true, fresh: true });
  if (!isEnd) s.response.push(s.pick.text);
  s.stage = 6;
  if (isEnd) {
    s.done = true;
    s.status = "The model picked its end token. Done.";
  } else if (s.response.length >= K.max) {
    s.done = true;
    s.status = `Stopped at the max of ${K.max} new tokens (set in the panel).`;
  }
}

async function next() {
  if (busy || S.done) return;
  busy = true;
  const snapshot = JSON.stringify(S);
  try {
    await advance();
    history.push(snapshot);
  } catch (e) {
    S = JSON.parse(snapshot);
    S.status = e.message;
    stopAuto();
  } finally {
    busy = false;
    draw();
  }
}

function back() {
  if (busy || !history.length) return;
  stopAuto();
  S = JSON.parse(history.pop());
  S.animate = false;
  draw();
}

function pick(c) {
  if (S.stage !== 4) return;
  S.pick = { id: c.id, text: c.text, prob: c.prob };
  draw();
}

function reset() {
  stopAuto();
  S = fresh();
  history = [];
  draw();
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function runAuto() {
  if (autoplay) return;
  autoplay = true;
  $("auto").innerHTML = "&#10074;&#10074; Pause <kbd>P</kbd>";
  while (autoplay && !S.done) {
    await next();
    await sleep(K.speed * 1000);
  }
  stopAuto();
}

function stopAuto() {
  autoplay = false;
  $("auto").innerHTML = "&#9654; Auto-play <kbd>A</kbd>";
}

// Hidden controls are remembered per browser; storage may be blocked, so failures are ignored.
function setCollapsed(on) {
  document.body.classList.toggle("collapsed", on);
  try { localStorage.setItem("llmw-collapsed", on ? "1" : "0"); } catch (e) { /* not remembered */ }
}

function startCollapsed() {
  try { return localStorage.getItem("llmw-collapsed") === "1"; } catch (e) { return false; }
}

const KEYS = {
  n: next, arrowright: next, " ": next,
  b: back, arrowleft: back,
  a: runAuto, p: stopAuto, s: finish, r: reset, t: toggleHidden,
  h: () => setCollapsed(!document.body.classList.contains("collapsed")),
};

async function finish() {
  stopAuto();
  while (!S.done && !S.status) await next();
}

function toggleHidden() {
  $("k-hidden").checked = !$("k-hidden").checked;
  readKnobs();
  draw();
}

// On a phone the page scrolls, so each step brings its own section into view.
const PHONE = window.matchMedia("(max-width: 900px)");
history.scrollRestoration = "manual";  // a reload starts at the prompt, not mid-page
let lastStage = -1;

function followStage() {
  if (S.stage === lastStage) return;
  lastStage = S.stage;
  if (!PHONE.matches || S.stage === 0) return;
  const sec = document.querySelector(`[data-stage="${S.stage}"]`);
  if (sec) sec.scrollIntoView({ behavior: "smooth", block: "center" });
}

function draw() {
  Render.all(S, K, pick, toggleHidden);
  followStage();
  $("next").disabled = busy || S.done;
  $("m-next").disabled = busy || S.done;
  $("m-back").disabled = busy || !history.length;
  $("back").disabled = busy || !history.length;
  $("finish").disabled = busy || S.done;
}

function wire() {
  $("next").onclick = next;
  $("back").onclick = back;
  $("reset").onclick = reset;
  $("finish").onclick = finish;
  $("auto").onclick = () => (autoplay ? stopAuto() : runAuto());
  document.querySelectorAll(".panel input").forEach((i) => {
    i.addEventListener("input", () => {
      readKnobs();
      if (i.id === "k-template") reset(); else draw();
    });
  });
  $("m-next").onclick = next;
  $("credits-link").onclick = (e) => { e.preventDefault(); stopAuto(); $("credits").showModal(); };
  $("collapse").onclick = () => setCollapsed(true);
  $("m-back").onclick = back;
  $("menu-open").onclick = () => setMenu(true);
  $("menu-close").onclick = () => setMenu(false);
  document.querySelectorAll(".panel .buttons button").forEach((b) =>
    b.addEventListener("click", () => setMenu(false)));
  // on a phone the controls sit below the content, so the menu button scrolls to them
  $("m-open").onclick = () => (PHONE.matches
    ? document.querySelector(".panel").scrollIntoView({ behavior: "smooth" })
    : setCollapsed(false));
  // Enter starts the walkthrough; Shift+Enter is a real line break, which becomes a token of its own.
  $("prompt").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); next(); }
  });
  $("prompt").addEventListener("input", fitPrompt);
  document.addEventListener("keydown", (e) => {
    // typing in the prompt or a number box must not trigger shortcuts
    // on a focused checkbox or slider, Space and the arrows keep their own job; letters still work
    if (e.target.tagName === "TEXTAREA") return;
    if (e.target.tagName === "INPUT" && e.target.type !== "checkbox" && e.target.type !== "range") return;
    if (e.target.tagName === "INPUT" && e.key.length !== 1) return;
    if (e.target.tagName === "INPUT" && e.key === " ") return;
    if (e.key === "Escape" && document.body.classList.contains("menu-open")) { setMenu(false); return; }
    if (e.ctrlKey || e.metaKey || e.altKey || $("credits").open) return;
    const act = KEYS[e.key.toLowerCase()];
    if (!act) return;
    e.preventDefault();
    act();
  });
}

function setModelLine(text) {
  document.querySelectorAll(".model-line").forEach((el) => (el.textContent = text));
}

// Phones: the controls are a full-screen menu. Any action button in it closes it,
// so the student sees the step that button just took.
function setMenu(open) {
  document.body.classList.toggle("menu-open", open);
  $("menu-open").setAttribute("aria-expanded", open);
}

// Token tooltips (render.js sets data-tip): hover on desktop, tap or keyboard focus on phones.
function showTip(e) {
  const el = e.target.closest("[data-tip]");
  const t = $("tip");
  if (!el) { t.hidden = true; return; }
  t.textContent = el.dataset.tip;
  t.hidden = false;
  const r = el.getBoundingClientRect();
  const left = Math.min(Math.max(8, r.left + r.width / 2 - t.offsetWidth / 2), innerWidth - t.offsetWidth - 8);
  const top = r.top - t.offsetHeight - 6;
  t.style.left = left + "px";
  t.style.top = (top < 8 ? r.bottom + 6 : top) + "px";
}

function wireTips() {
  // click too: some phone browsers do not focus a tapped span
  ["mouseover", "focusin", "click"].forEach((ev) => document.addEventListener(ev, showTip));
  document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) $("tip").hidden = true; });
  document.addEventListener("focusout", () => ($("tip").hidden = true));
  document.addEventListener("scroll", () => ($("tip").hidden = true), true);
  window.addEventListener("resize", () => ($("tip").hidden = true));
}

// The prompt box grows with its text (up to the CSS max-height, then scrolls).
function fitPrompt() {
  const p = $("prompt");
  p.style.height = "auto";
  p.style.height = p.scrollHeight + 2 + "px";
}

async function boot() {
  setCollapsed(startCollapsed());
  wireTips();
  fitPrompt();
  window.scrollTo(0, 0);
  readKnobs();
  wire();
  S = fresh();
  draw();
  try {
    info = await api("/api/info");
    S.info = info;
    setModelLine(`Running locally: ${info.model}, ${info.params_millions}M parameters, ` +
      `${info.layers} layers, ${info.vocab_size.toLocaleString()}-token vocabulary`);
  } catch (e) {
    setModelLine("Model server not reachable: " + e.message);
  }
  draw();
}

boot();
