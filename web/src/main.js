// Bloom UMAP parameter explorer.
// Left: Observable Plot scatter of the 100 fits (NMI vs within-subclass variance).
// Right: Embedding Atlas EmbeddingView of the hovered / pinned fit (15,750 cells).
import "./style.css";
import * as Plot from "@observablehq/plot";
import { EmbeddingView } from "embedding-atlas";

const NN = [5, 10, 15, 20, 25];
const NN_COLOURS = ["#440154", "#3b528b", "#21918c", "#5ec962", "#b5de2b"];
const XLAB = {
  v_span: "Within-subclass variance / displayed span\u00b2",
  v_total: "Within-subclass variance / total variance",
};

const state = {
  fits: [], byId: {}, cells: null, cache: new Map(), grids: new Map(),
  pinned: null, shown: null, xcol: "v_span", colourBy: "subclass", view: null, current: null,
};

// ---------------- data ----------------
async function loadFit(id) {
  if (state.cache.has(id)) return state.cache.get(id);
  const p = fetch(`data/fits/${id}.bin`).then(r => {
    if (!r.ok) throw new Error(`${id}: HTTP ${r.status}`);
    return r.arrayBuffer();
  }).then(buf => {
    const f = state.byId[id], q = new Uint16Array(buf), n = state.cells.n;
    if (q.length !== 2 * n) throw new Error(`${id}: ${q.length} values, expected ${2 * n}`);
    const x = new Float32Array(n), y = new Float32Array(n);
    for (let i = 0; i < n; i++) { x[i] = f.x0 + q[i] * f.scale; y[i] = f.y0 + q[n + i] * f.scale; }
    return { x, y };
  });
  state.cache.set(id, p);
  p.catch(() => state.cache.delete(id));
  return p;
}

// Uniform-grid index (CSR) over one fit, for nearest-cell hover lookups.
function buildGrid(id, x, y) {
  if (state.grids.has(id)) return state.grids.get(id);
  const f = state.byId[id], n = x.length, G = 160, cs = f.frame_span / G;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) { x0 = Math.min(x0, x[i]); x1 = Math.max(x1, x[i]); y0 = Math.min(y0, y[i]); y1 = Math.max(y1, y[i]); }
  const nx = Math.floor((x1 - x0) / cs) + 1, ny = Math.floor((y1 - y0) / cs) + 1;
  const key = new Int32Array(n), counts = new Int32Array(nx * ny + 1);
  for (let i = 0; i < n; i++) {
    const k = Math.floor((y[i] - y0) / cs) * nx + Math.floor((x[i] - x0) / cs);
    key[i] = k; counts[k + 1]++;
  }
  for (let k = 0; k < nx * ny; k++) counts[k + 1] += counts[k];
  const fill = counts.slice(0, nx * ny), items = new Int32Array(n);
  for (let i = 0; i < n; i++) items[fill[key[i]]++] = i;
  const g = { x0, y0, cs, nx, ny, offsets: counts, items };
  state.grids.set(id, g);
  return g;
}

function nearest(g, x, y, X, Y, radius) {
  const r = Math.ceil(radius / g.cs);
  const cx = Math.floor((X - g.x0) / g.cs), cy = Math.floor((Y - g.y0) / g.cs);
  let best = -1, bd = radius * radius;
  for (let j = Math.max(0, cy - r); j <= Math.min(g.ny - 1, cy + r); j++) {
    for (let i = Math.max(0, cx - r); i <= Math.min(g.nx - 1, cx + r); i++) {
      const k = j * g.nx + i;
      for (let t = g.offsets[k]; t < g.offsets[k + 1]; t++) {
        const p = g.items[t], dx = x[p] - X, dy = y[p] - Y, d = dx * dx + dy * dy;
        if (d < bd) { bd = d; best = p; }
      }
    }
  }
  return best;
}

// ---------------- right: embedding view ----------------
function categories() {
  const c = state.cells;
  return state.colourBy === "subclass"
    ? { category: c.subclassCat, colours: c.subclass_slot_colors }
    : { category: c.classCat, colours: c.classes.map(k => k.color) };
}

function viewport(f) { return { x: f.frame_cx, y: f.frame_cy, scale: 2 / f.frame_span }; }

class CellTooltip {
  constructor(target, props) { this.el = document.createElement("div"); this.el.className = "cell-tip"; target.appendChild(this.el); this.update(props); }
  update(props) {
    const t = props.tooltip, c = state.cells;
    if (!t || t.identifier == null) { this.el.innerHTML = ""; this.el.style.display = "none"; return; }
    const i = Number(t.identifier), s = c.subclasses[c.subclass[i]], k = c.classes[c.klass[i]];
    let html = `<div><span class="sw" style="background:${s.color}"></span>${s.name}</div>` +
               `<div><span class="sw" style="background:${k.color}"></span>${k.name}</div>`;
    if (s.klass !== c.klass[i]) html += `<div class="flag">subclass's parent class: ${c.classes[s.klass].name}</div>`;
    this.el.innerHTML = html; this.el.style.display = "block";
  }
  destroy() { this.el.remove(); }
}

async function querySelection(X, Y, unitDistance) {
  const cur = state.current; if (!cur) return null;
  const g = buildGrid(cur.id, cur.x, cur.y);
  const i = nearest(g, cur.x, cur.y, X, Y, 6 * unitDistance);
  if (i < 0) return null;
  const cat = categories().category;
  return { x: cur.x[i], y: cur.y[i], category: cat[i], identifier: i };
}

function viewSize() {
  const el = document.getElementById("umap-plot");
  return { width: el.clientWidth, height: el.clientHeight };
}

// Embedding Atlas renders with WebGPU only (no WebGL path) and needs the shader-f16 feature.
async function webgpuProblem() {
  if (!navigator.gpu) return "this browser has no WebGPU";
  try {
    const a = await navigator.gpu.requestAdapter();
    if (!a) return "no WebGPU adapter is available";
    if (!a.features.has("shader-f16")) return "the WebGPU adapter lacks the shader-f16 feature";
  } catch (e) { return `WebGPU failed (${e.message})`; }
  return null;
}

function showNoWebGPU(reason) {
  const el = document.getElementById("umap-plot");
  el.innerHTML = `<div class="nogpu"><div><b>The map view needs WebGPU</b><br>${reason}.<br>` +
    `Use a recent Chrome or Edge (desktop) or Safari 26+, with hardware acceleration on. ` +
    `The metrics panel still works.</div></div>`;
}

function setFitInfo(id) {
  const f = state.byId[id], pinned = id === state.pinned;
  document.getElementById("fit-info").innerHTML =
    `${pinned ? "<b>pinned</b>" : "<i>preview</i>"}: n_neighbors ${f.n_neighbors}, min_dist ${f.min_dist}, ` +
    `spread ${f.spread} &middot; NMI ${f.nmi_mean.toFixed(3)}`;
}

async function showFit(id) {
  setFitInfo(id);
  if (state.noGPU) return;
  state.shown = id;
  const f = state.byId[id], xy = await loadFit(id);
  if (state.shown !== id) return;  // a newer hover won
  state.current = { id, x: xy.x, y: xy.y };
  const { category, colours } = categories();
  const props = { data: { x: xy.x, y: xy.y, category }, categoryColors: colours,
                  viewportState: viewport(f), tooltip: null };
  if (!state.view) {
    state.view = new EmbeddingView(document.getElementById("umap-plot"), {
      ...props, ...viewSize(), config: { autoLabelEnabled: false, colorScheme: "light" },
      theme: { statusBar: false },
      querySelection, customTooltip: { class: CellTooltip, props: {} },
      onViewportState: v => state.view && state.view.update({ viewportState: v }),
      onTooltip: v => state.view && state.view.update({ tooltip: v }),
    });
  } else {
    state.view.update(props);
  }
}

// ---------------- left: metrics scatter ----------------
function drawMetrics() {
  const host = document.getElementById("metrics-plot"), xcol = state.xcol, pk = `pareto_${xcol}`;
  const fits = state.fits, pin = state.byId[state.pinned];
  const fig = Plot.plot({
    width: Math.max(420, host.clientWidth - 8), height: 520, marginLeft: 58, marginBottom: 44,
    style: { fontSize: "12px" },
    x: { label: `${XLAB[xcol]} \u2192`, nice: true, inset: 8 },
    y: { label: "\u2191 Seeded k-means NMI (mean \u00b1 sd, 3 seeds)", nice: true, inset: 8 },
    color: { type: "ordinal", domain: NN, range: NN_COLOURS },
    symbol: { domain: [1, 1.5, 2], range: ["circle", "square", "triangle"] },
    marks: [
      Plot.ruleX(fits, { x: xcol, y1: d => d.nmi_mean - d.nmi_sd, y2: d => d.nmi_mean + d.nmi_sd, stroke: "#aaa" }),
      Plot.dot(fits.filter(d => !d[pk]), { x: xcol, y: "nmi_mean", fill: "n_neighbors", symbol: "spread", r: 5 }),
      Plot.dot(fits.filter(d => d[pk]), { x: xcol, y: "nmi_mean", fill: "n_neighbors", symbol: "spread", r: 5,
                                          stroke: "black", strokeWidth: 1.6 }),
      Plot.dot(pin ? [pin] : [], { x: xcol, y: "nmi_mean", r: 11, stroke: "#d62728", strokeWidth: 2.2, fill: "none" }),
      Plot.tip(fits, Plot.pointer({
        x: xcol, y: "nmi_mean", maxRadius: 30,
        title: d => `n_neighbors ${d.n_neighbors}, min_dist ${d.min_dist}, spread ${d.spread}\n` +
          `NMI ${d.nmi_mean.toFixed(3)} \u00b1 ${d.nmi_sd.toFixed(3)}\n` +
          `var/span\u00b2 ${d.v_span.toFixed(4)} \u00b7 var/total ${d.v_total.toFixed(3)}` +
          (d[pk] ? "\nPareto front" : ""),
      })),
    ],
  });
  let t = null;
  fig.addEventListener("input", () => {
    const d = fig.value; clearTimeout(t);
    t = setTimeout(() => showFit(d ? d.fit_id : state.pinned), d ? 30 : 150);
  });
  fig.addEventListener("click", () => { if (fig.value) { clearTimeout(t); pinFit(fig.value.fit_id); } });
  const legend = document.createElement("div");
  legend.className = "legend";
  const title = t => Object.assign(document.createElement("span"), { className: "legend-title", textContent: t });
  legend.append(
    title("n_neighbors"), Plot.legend({ color: { type: "ordinal", domain: NN, range: NN_COLOURS } }),
    title("spread"), Plot.legend({ symbol: { domain: [1, 1.5, 2], range: ["circle", "square", "triangle"] } }),
  );
  host.replaceChildren(legend, fig);
}

function pinFit(id) { state.pinned = id; drawMetrics(); showFit(id); }

async function prefetchAll() {
  const ids = state.fits.map(f => f.fit_id), status = document.getElementById("status");
  let k = 0, done = 0;
  const worker = async () => { while (k < ids.length) { const id = ids[k++]; try { await loadFit(id); } catch (e) { console.error(e); } done++; } };
  await Promise.all(Array.from({ length: 6 }, worker));
  status.textContent = `hover a cell to see its subclass and class \u00b7 scroll to zoom, drag to pan \u00b7 ${done}/${ids.length} maps loaded`;
}

async function main() {
  const [mj, cells] = await Promise.all([
    fetch("data/metrics.json").then(r => r.json()), fetch("data/cells.json").then(r => r.json())]);
  state.fits = mj.fits; state.fits.forEach(f => { state.byId[f.fit_id] = f; });
  cells.subclassCat = Uint8Array.from(cells.subclass, s => cells.subclass_slot[s]);
  cells.classCat = Uint8Array.from(cells.klass);
  state.cells = cells;
  const m = mj.meta;
  document.getElementById("meta-line").textContent =
    `${m.n_fits} UMAP fits of one embedding: ${m.embedding}; ${m.dataset}.`;
  document.getElementById("about").innerHTML =
    `<b>Method.</b> ${m.umap}; grid n_neighbors 5\u201325 (step 5), min_dist 0.2\u20131.6 (step 0.2), ` +
    `spread 1, 1.5, 2 with min_dist \u2264 spread. NMI: ${m.nmi}; it uses the labels to start k-means, ` +
    `so use it to rank maps, not as an absolute clustering score. Variance: ${m.v_span} (or ${m.v_total}); ` +
    `it rises when a subclass is split into distant islands, not only when its blob is wider. ` +
    `Each map opens framed like the paper figures (same span on both axes, 0.2\u201399.8 percentile range \u00d7 1.06). ` +
    `Labels are the metadata's own AIT33_subclass and AIT33_class; the tooltip flags cells whose class is not ` +
    `their subclass's parent. The viewer supports at most 256 colours, so ${cells.subclasses.length - cells.subclass_slot_colors.length} ` +
    `pairs of subclasses with near-identical taxonomy colours share one colour; tooltips always name the exact subclass. ` +
    `Rendering: <a href="https://apple.github.io/embedding-atlas/">Embedding Atlas</a> and ` +
    `<a href="https://observablehq.com/plot/">Observable Plot</a>.`;

  const best = state.fits.reduce((a, b) => (b.nmi_mean > a.nmi_mean ? b : a));
  state.pinned = best.fit_id;
  drawMetrics();
  const problem = await webgpuProblem();
  if (problem) { state.noGPU = problem; showNoWebGPU(problem); }
  await showFit(best.fit_id);

  document.getElementById("xsel").addEventListener("change", e => { state.xcol = e.target.value; drawMetrics(); });
  document.getElementById("csel").addEventListener("change", e => {
    state.colourBy = e.target.value; const { category, colours } = categories();
    if (state.view && state.current)
      state.view.update({ data: { x: state.current.x, y: state.current.y, category }, categoryColors: colours });
  });
  new ResizeObserver(() => state.view && state.view.update(viewSize())).observe(document.getElementById("umap-plot"));
  let rt = null;
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(drawMetrics, 150); });
  prefetchAll();
}

window.__dash = state;  // inspection hook (read-only use)
main().catch(e => {
  console.error(e);
  document.getElementById("meta-line").textContent = `Failed to load: ${e.message}`;
});
