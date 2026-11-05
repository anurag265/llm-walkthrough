// Schematic of the network. The shape is illustrative; the input brightness is real attention.
const Diagram = (() => {
  const W = 760, H = 360, NS = "http://www.w3.org/2000/svg";
  const MAX_INPUTS = 8;

  function el(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function column(x, n, top = 20, bottom = H - 40) {
    const step = n > 1 ? (bottom - top) / (n - 1) : 0;
    const mid = n > 1 ? top : (top + bottom) / 2;
    return Array.from({ length: n }, (_, i) => ({ x, y: mid + i * step }));
  }

  // inputs: [{label, weight 0..1}] already trimmed to what is shown
  function draw(container, inputs, layers, running, ellipsis) {
    container.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}` }, container);
    const cols = [
      column(150, Math.max(inputs.length, 1)),
      column(290, 6, 35, H - 55),
      column(410, 6, 35, H - 55),
      column(530, 6, 35, H - 55),
      column(660, 5, 55, H - 75),
    ];
    const labels = ["Embeddings", "Layer 1", "Layer 2", `Layer ${layers}`, "Scores"];
    const edges = el("g", {}, svg);
    for (let c = 0; c < cols.length - 1; c++) {
      for (const [ai, a] of cols[c].entries()) {
        const w = c === 0 && inputs[ai] ? inputs[ai].weight : 0.3;
        for (const b of cols[c + 1]) {
          el("line", {
            x1: a.x, y1: a.y, x2: b.x, y2: b.y,
            class: `edge col${c}`,
            "stroke-opacity": (0.08 + 0.7 * w).toFixed(2),
          }, edges);
        }
      }
    }
    cols.forEach((col, c) => {
      col.forEach((p, i) => {
        const w = c === 0 && inputs[i] ? inputs[i].weight : 0;
        el("circle", {
          cx: p.x, cy: p.y, r: 9, class: `node col${c}`,
          "fill-opacity": c === 0 ? (0.12 + 0.88 * w).toFixed(2) : 1,
        }, svg);
      });
      el("text", { x: col[0].x, y: H - 12, class: "dlabel", "text-anchor": "middle" }, svg)
        .textContent = labels[c];
    });
    if (ellipsis) {
      el("text", { x: 130, y: 12, class: "tlabel", "text-anchor": "end" }, svg).textContent = "...";
    }
    cols[0].forEach((p, i) => {
      if (!inputs[i]) return;
      el("text", { x: p.x - 18, y: p.y + 4, class: "tlabel", "text-anchor": "end" }, svg)
        .textContent = inputs[i].label;
    });
    // dots between Layer 2 and Layer N say "many more layers here"
    el("text", { x: 470, y: H / 2 - 10, class: "dlabel", "text-anchor": "middle" }, svg).textContent = "...";
    if (running) svg.classList.add("running");
  }

  return { draw, MAX_INPUTS };
})();
