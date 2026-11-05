// Pure rendering: draws every stage from the walkthrough state S and the knobs K.
const Render = (() => {
  const $ = (id) => document.getElementById(id);
  const pct = (p) => (p * 100 >= 10 ? (p * 100).toFixed(0) : (p * 100).toFixed(1)) + "%";

  function show(text, K) {
    let t = text.replace(/\n/g, "↵");
    return K.spaces ? t.replace(/ /g, "␣") : t;
  }

  function visible(S, K) {
    return S.tokens.map((t, i) => ({ ...t, i })).filter((t) => K.hidden || !t.hidden);
  }

  function chip(t, K, withId) {
    const c = document.createElement("span");
    c.className = "chip" + (t.hidden ? " hidden" : "") + (t.generated ? " generated" : "") +
      (t.fresh ? " fresh" : "");
    c.textContent = show(t.text, K);
    if (withId) {
      const id = document.createElement("span");
      id.className = "id";
      id.textContent = t.id;
      c.appendChild(id);
    }
    return c;
  }

  // One row of chips for stages 1 and 2. When hidden tokens are folded, each run of them
  // becomes a single "+N hidden" chip that unfolds on click (or the T key).
  function tokensRow(S, K, withId, heat, onToggleHidden) {
    const target = $("tokens");
    target.innerHTML = "";
    let run = 0, runAttn = 0;
    const flush = () => {
      if (!run) return;
      const f = document.createElement("span");
      f.className = "chip hidden fold";
      f.textContent = `+${run} hidden`;
      f.title = "Template tokens the model reads but you never typed. Click or press T.";
      if (heat) tip(f, `${run} hidden tokens, attention ${pctAttn(runAttn)} combined`);
      f.onclick = onToggleHidden;
      target.appendChild(f);
      run = 0;
      runAttn = 0;
    };
    S.tokens.forEach((t, i) => {
      if (t.hidden && !K.hidden) { run += 1; runAttn += heat ? heat.raw(i) : 0; return; }
      flush();
      const c = chip(t, K, withId);
      if (heat && heat.has(i)) {
        const mix = Math.round(5 + 85 * heat.get(i));
        c.style.background = `color-mix(in srgb, var(--accent) ${mix}%, var(--box))`;
        if (mix > 50) c.style.color = "var(--on-accent)";
        const sink = i === 0 ? " (attention sink, off the colour scale)" : "";
        tip(c, `Attention ${pctAttn(heat.raw(i))}${sink}`);
      }
      target.appendChild(c);
    });
    flush();
    if (S.tokens.some((t) => t.fresh)) target.scrollTop = target.scrollHeight;
  }

  // Attention as a percentage, with enough digits that small values do not read as 0.
  function pctAttn(a) {
    const p = a * 100;
    return (p >= 10 ? p.toFixed(1) : p >= 0.1 ? p.toFixed(2) : p.toFixed(3)) + "%";
  }

  // Tooltip: shown on hover, and on tap or keyboard focus (phones have no hover).
  function tip(el, text) {
    el.dataset.tip = text;
    el.tabIndex = 0;
  }

  // The last position's attention to position i, averaged over all layers (each already
  // averaged over heads by the server). Sums to 1 across positions.
  function rawAttention(S, i) {
    const att = S.result.attention;
    return att.reduce((s, layer) => s + (layer[i] || 0), 0) / att.length;
  }

  // Attention per position, averaged over layers, softened and scaled so the brightest shown = 1.
  // Position 0 is an "attention sink" that soaks up most attention in most models, so it is
  // left out of the scale; otherwise every other token would look dark.
  function weights(S, list) {
    const att = S.result ? S.result.attention : null;
    if (!att) return list.map(() => 0);
    const raw = list.map((t) => Math.sqrt(rawAttention(S, t.i)));
    const max = Math.max(...raw.filter((_, j) => list[j].i !== 0), 1e-9);
    return raw.map((w) => Math.min(w / max, 1));
  }

  // Attention heat for the token row, keyed by position; only while a pass is on screen.
  function heatMap(S, K) {
    if (!K.attn || !S.result || S.stage < 3 || S.stage > 5) return null;
    const list = visible(S, K).filter((t) => t.i < S.result.attention[0].length);
    const w = weights(S, list);
    const map = new Map(list.map((t, j) => [t.i, w[j]]));
    map.raw = (i) => rawAttention(S, i);
    return map;
  }

  function stage3(S, K) {
    const list = visible(S, K).filter((t) => S.result && t.i < S.result.attention[0].length);
    const w = weights(S, list);
    const shown = list.slice(-Diagram.MAX_INPUTS);
    const inputs = shown.map((t, j) => ({
      label: show(t.text, K).slice(0, 12), weight: K.attn ? w[list.length - shown.length + j] : 0.3,
    }));
    Diagram.draw($("diagram"), inputs, S.info ? S.info.layers : "N",
      S.stage === 3 && S.animate, list.length > shown.length);
    $("attn-note").hidden = !heatMap(S, K);
    readsLine(S, K);
  }

  function readsLine(S, K) {
    if (!S.result) { $("reads").textContent = ""; $("cache-note").hidden = true; return; }
    const total = S.result.attention[0].length;
    const gen = S.tokens.slice(0, total).filter((t) => t.generated).length;
    const hidden = S.tokens.slice(0, total).filter((t) => t.hidden).length;
    const hid = hidden ? `, incl. ${hidden} hidden` : "";
    $("reads").textContent = `Pass ${S.pass}: reads prompt (${total - gen} tokens${hid}) + ` +
      `${gen ? gen + " generated" : "nothing generated yet"} = ${total} tokens, ${S.result.ms} ms on CPU`;
    $("cache-note").hidden = !K.cache;
    $("cache-note").textContent = S.pass === 1
      ? "KV cache: on pass 1 every token is computed. Real servers then save each token's work, " +
        "so later passes only compute the 1 newest token. This demo reruns everything so you can see it."
      : `KV cache: a real server would reuse the saved work for the first ${total - 1} tokens ` +
        `and compute only the newest one. Same answer, much faster. This demo reruns all ${total}.`;
  }

  function stage4(S, K, onPick) {
    const bars = $("bars");
    bars.innerHTML = "";
    if (!S.result) return;
    const rows = S.result.candidates.slice(0, K.topk);
    if (!rows.some((c) => c.id === S.pick.id)) rows.push({ ...S.pick, logit: null });
    bars.classList.toggle("two", rows.length > 5);
    for (const c of rows) {
      const r = document.createElement("div");
      r.className = "bar-row" + (c.id === S.pick.id ? " picked" : "");
      const logit = K.logits && c.logit !== null ? `<span class="logit">score ${c.logit.toFixed(2)}</span>` : "";
      r.innerHTML = `<span class="tok"></span><div class="bar-track"><div class="bar"></div></div>` +
        `<span class="pct">${pct(c.prob)}${logit}</span>`;
      r.querySelector(".tok").textContent = show(c.text, K);
      r.querySelector(".bar").style.width = Math.max(c.prob * 100, 0.5) + "%";
      r.onclick = () => onPick(c);
      bars.appendChild(r);
    }
    const rest = 1 - rows.reduce((s, c) => s + c.prob, 0);
    const vocab = S.info ? S.info.vocab_size.toLocaleString() : "";
    const how = S.temp > 0
      ? `Temperature ${S.temp}: the model rolls a weighted dice (seed ${S.seed}).`
      : "Temperature 0: the model takes the likeliest.";
    $("softmax-sub").textContent = `${how} The other ${pct(Math.max(rest, 0))} is spread over the ` +
      `rest of the ${vocab}-token vocabulary. Click a bar to pick it yourself.`;
  }

  function stage5(S, K) {
    const d = $("detok");
    if (!S.pick) { d.textContent = ""; return; }
    const end = S.result.eos_ids.includes(S.pick.id) ? "  (the end token: the model has finished)" : "";
    d.innerHTML = `<span class="chip"></span><span class="arrow">&rarr;</span><span class="txt"></span>`;
    d.querySelector(".chip").textContent = "ID " + S.pick.id;
    d.querySelector(".txt").textContent = `"${show(S.pick.text, K)}"${end}`;
  }

  function stage6(S) {
    const r = $("response");
    r.innerHTML = "";
    S.response.forEach((piece, i) => {
      const s = document.createElement("span");
      s.textContent = piece;
      if (i === S.response.length - 1 && S.stage === 6) s.className = "new";
      r.appendChild(s);
    });
  }

  function all(S, K, onPick, onToggleHidden) {
    document.querySelectorAll("[data-stage]").forEach((sec) => {
      const n = +sec.dataset.stage;
      const to = +(sec.dataset.stageTo || n);
      sec.classList.toggle("active", S.stage >= n && S.stage <= to);
      sec.classList.toggle("seen", S.stage > to || (S.pass > 0 && n < 3));
    });
    document.querySelectorAll("[data-num]").forEach((el) => {
      el.classList.toggle("lit", +el.dataset.num === S.stage);
    });
    $("prompt").disabled = S.stage > 0;
    if (S.stage >= 1) {
      tokensRow(S, K, S.stage >= 2 || S.pass > 0, heatMap(S, K), onToggleHidden);
    } else {
      $("tokens").innerHTML = "";
    }
    $("ids-sub").textContent = S.pass > 0 && S.stage === 2
      ? "The new token's ID joins the input. No re-tokenizing: it is already a number."
      : "Text splits into tokens; each token becomes its number in the vocabulary";
    stage3(S, K);
    // between passes the last pass's bars stay visible; during a forward pass they would spoil it
    stage4(S.stage >= 4 || (S.stage < 3 && S.pass > 0) ? S : { ...S, result: null }, K, onPick);
    stage5(S.stage >= 5 ? S : { ...S, pick: null }, K);
    stage6(S);
    $("loop").classList.toggle("active", S.stage === 6 && !S.done);
    document.querySelectorAll(".status").forEach((el) => (el.textContent = S.status || ""));
    $("pass").textContent = S.pass ? `Pass ${S.pass}, ${S.response.length} tokens generated` : "";
  }

  return { all };
})();
