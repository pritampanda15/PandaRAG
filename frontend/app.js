const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/<\/?(?:sup|sub|i|b|em|strong|u|small)\b[^>]*>/gi, "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const state = { ws: null, opts: {}, lit: [], graph: null, history: [], lastRun: null, bioLabs: true };

async function api(path, opts) {
  const r = await fetch("/api" + path, {
    method: opts?.body ? "POST" : opts?.method || "GET",
    headers: opts?.body && !(opts.body instanceof Blob) ? { "Content-Type": "application/json" } : undefined,
    body: opts?.body instanceof Blob ? opts.body : opts?.body ? JSON.stringify(opts.body) : undefined,
    ...(opts?.method ? { method: opts.method } : {}),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `${r.status} ${r.statusText}`);
  return data;
}

function toast(msg, bad) {
  const t = Object.assign(document.createElement("div"), { className: "toast" + (bad ? " bad" : ""), textContent: msg });
  document.body.append(t);
  t.animate([{ opacity: 0, transform: "translate(-50%,16px) scale(.94)" }, { opacity: 1, transform: "translate(-50%,0) scale(1)" }],
    { duration: 380, easing: "cubic-bezier(.22,1.2,.3,1)", fill: "both" });
  setTimeout(() => t.animate([{ opacity: 1, transform: "translate(-50%,0)" }, { opacity: 0, transform: "translate(-50%,10px)" }],
    { duration: 260, fill: "both" }).finished.then(() => t.remove()), 2600);
}

/* ---------------- tabs: spring glider, no layout thrash ---------------- */
const tabs = $("#tabs"), glider = $(".glider", tabs);
function moveGlider(btn, animate = true) {
  const r = btn.getBoundingClientRect(), p = tabs.getBoundingClientRect();
  const x = r.left - p.left + tabs.scrollLeft, w = r.width;
  if (!animate) { glider.style.transform = `translateX(${x}px)`; glider.style.width = w + "px"; return; }
  glider.animate([{ transform: glider.style.transform || "translateX(0)", width: glider.style.width || "0px" },
    { transform: `translateX(${x}px)`, width: w + "px" }],
    { duration: 420, easing: "cubic-bezier(.22,1.2,.3,1)", fill: "both" });
  glider.style.transform = `translateX(${x}px)`; glider.style.width = w + "px";
}
const onShow = {};
function show(view, btn) {
  $$(".view").forEach((v) => { v.hidden = v.id !== "view-" + view; });
  $$("button[data-view]", tabs).forEach((b) => b.setAttribute("aria-selected", String(b === btn)));
  const sec = $("#view-" + view);
  sec.classList.remove("enter"); void sec.offsetWidth; sec.classList.add("enter");
  moveGlider(btn);
  onShow[view]?.();
}
tabs.addEventListener("click", (e) => { const b = e.target.closest("button[data-view]"); if (b) show(b.dataset.view, b); });
addEventListener("resize", () => moveGlider($('button[aria-selected="true"]', tabs), false));

$("#theme-btn").onclick = () => {
  const light = document.documentElement.dataset.theme === "light";
  document.documentElement.dataset.theme = light ? "dark" : "light";
  try { localStorage.setItem("pandarag-theme", light ? "dark" : "light"); } catch {}
};
try { const t = localStorage.getItem("pandarag-theme"); if (t) document.documentElement.dataset.theme = t; } catch {}

function setBioLabs(on) {
  state.bioLabs = !!on;
  $$("button.bio-tab", tabs).forEach((b) => { b.hidden = !state.bioLabs; });
  $("#bio-btn").classList.toggle("active", state.bioLabs);
  $("#bio-btn").textContent = state.bioLabs ? "⬡ bio labs" : "⬡ core rag";
  const cur = $('button[aria-selected="true"]', tabs);
  if (cur?.hidden) show("ask", $('button[data-view="ask"]', tabs));
  else if (cur) moveGlider(cur, false);
  try { localStorage.setItem("pandarag-bio", state.bioLabs ? "1" : "0"); } catch {}
}
$("#bio-btn").onclick = () => setBioLabs(!state.bioLabs);
try { const b = localStorage.getItem("pandarag-bio"); if (b === "0") setBioLabs(false); } catch {}

/* ---------------- workspaces ---------------- */
async function loadWorkspaces(select) {
  const ws = await api("/workspaces");
  const sel = $("#ws-select");
  sel.innerHTML = ws.map((w) => `<option value="${w.id}">${esc(w.name)} · ${w.docs} docs</option>`).join("");
  state.ws = select && ws.some((w) => w.id === select) ? select : ws[0]?.id || null;
  sel.value = state.ws || "";
  return ws;
}
$("#ws-select").onchange = (e) => { state.ws = e.target.value; state.history = []; renderChatHistory(); refreshActive(); };
function refreshActive() {
  const v = $$(".view").find((s) => !s.hidden)?.id.replace("view-", "");
  onShow[v]?.();
}

/* ---------------- ASK ---------------- */
const OPTS = [
  ["hybrid", "Hybrid BM25 + dense", true], ["graph", "GraphRAG expansion", true], ["rerank", "MMR rerank", true],
  ["decompose", "Query decomposition", true], ["hyde", "HyDE", true], ["crag", "Corrective grading", true],
  ["auto_pubmed", "Auto-PubMed on gap", false],
];
$("#opts").innerHTML = OPTS.map(([k, label, def]) =>
  `<label class="toggle"><input type="checkbox" data-opt="${k}" ${def ? "checked" : ""}><i></i>${label}</label>`).join("");
OPTS.forEach(([k, , def]) => (state.opts[k] = def));
$("#opts").addEventListener("change", (e) => { state.opts[e.target.dataset.opt] = e.target.checked; });

$("#q-chips").innerHTML = ["What are the main mechanisms described?", "Compare the agents on toxicity",
  "What resistance pathways appear, and what evidence supports them?", "Summarise the open questions"]
  .map((q) => `<button class="chip">${esc(q)}</button>`).join("");
$("#q-chips").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) { $("#q").value = e.target.textContent; $("#ask-btn").click(); }
});

function renderChatHistory() {
  const pairs = [];
  for (let i = 0; i < state.history.length; i += 2) {
    if (state.history[i] && state.history[i + 1]) pairs.push([state.history[i].content, state.history[i + 1].content]);
  }
  $("#chat-turns").innerHTML = pairs.map(([uq, aa]) =>
    `<div class="turn"><div class="tq">Q: ${esc(uq)}</div><div class="ta">${esc(aa)}</div></div>`).join("");
  $("#chat-clear").hidden = !pairs.length;
  if (pairs.length) $("#chat-turns").scrollTop = $("#chat-turns").scrollHeight;
}
$("#chat-clear").onclick = () => { state.history = []; renderChatHistory(); toast("Conversation thread cleared"); };

const cited = (text) => esc(text).replace(/\[(\d+)\]/g, (_, n) => `<cite data-ref="${n}">${n}</cite>`);

function formatMarkdown(q, res) {
  const refs = (res.sources || []).map((s, i) => `[${i + 1}] **${s.title}** (${s.source}, grade ${s.grade}): ${s.text}`).join("\n");
  return `## ${q}\n\n${res.answer}\n\n### References\n${refs || "_No references_"}\n`;
}

async function fetchPubMedAndReask(q) {
  toast("Fetching PubMed abstracts…");
  const lit = await api("/literature", { body: { query: q, sources: ["pubmed"], max_results: 5 } });
  if (!lit.papers?.length) return toast("No PubMed papers found for that query", true);
  const imp = await api(`/workspaces/${state.ws}/import`, { body: { papers: lit.papers } });
  toast(`Imported ${imp.imported} papers (${imp.skipped || 0} duplicates skipped) — re-running…`);
  await loadWorkspaces(state.ws);
  $("#q").value = q;
  $("#ask-btn").click();
}

function renderAnswer(res, queryText = "") {
  state.lastRun = { query: queryText || $("#q").value.trim(), res };
  const wrap = $("#answer-wrap"), body = cited(res.answer);
  const cragAct = res.crag?.action;
  const cragBanner = (cragAct === "knowledge_gap" || cragAct === "augment")
    ? `<div class="banner"><div><b>CRAG ${esc(cragAct.replace("_", " "))}</b> — ${cragAct === "knowledge_gap" ? "no grounded passages passed grading." : "only weak/ambiguous passages found."}</div>
       <button class="mini" id="crag-pubmed-btn">❖ Fetch PubMed &amp; re-ask</button></div>`
    : (res.crag?.augmented_papers ? `<div class="banner"><div><b>CRAG augmented</b> — automatically imported ${res.crag.augmented_papers} PubMed abstracts.</div></div>` : "");
  wrap.innerHTML = `${cragBanner}
    <div class="row" style="justify-content:space-between;margin-bottom:12px">
      <div class="row" style="gap:6px"><span class="tag info">${esc(res.mode)}</span>
        <button class="mini" id="copy-md-btn">copy markdown</button>
        ${res.sources.length ? '<button class="mini" id="to-causal-btn">⇝ draft causal chain</button>' : ""}</div>
      <span class="eyebrow">${res.total_ms} ms · ${res.sources.length} passages</span></div>
    <div class="answer">${body}</div>`;

  $("#crag-pubmed-btn")?.addEventListener("click", () => fetchPubMedAndReask(state.lastRun.query).catch((e) => toast(e.message, true)));
  $("#copy-md-btn")?.addEventListener("click", async () => {
    await navigator.clipboard.writeText(formatMarkdown(state.lastRun.query, res));
    toast("Copied cited Markdown to clipboard");
  });
  $("#to-causal-btn")?.addEventListener("click", async () => {
    try {
      const out = await api(`/workspaces/${state.ws}/causal-extract`, { body: { query: state.lastRun.query, sources: res.sources } });
      if (!out.edges?.length) return toast("No causal hops found in retrieved passages", true);
      if (!state.bioLabs) setBioLabs(true);
      seedEdges.splice(0, seedEdges.length, ...out.edges);
      show("causal", $('button[data-view="causal"]', tabs));
      renderEdges();
      toast(`Drafted ${out.edges.length} causal hops from RAG evidence`);
    } catch (e) { toast(e.message, true); }
  });

  $("#metrics").innerHTML = [["faithfulness", "Faithfulness"], ["answer_relevance", "Answer relevance"],
    ["context_precision", "Context precision"]].map(([k, label]) => {
    const v = res.metrics[k] ?? 0;
    return `<div class="stat"><div class="k">${label}</div><div class="v">${(v * 100).toFixed(0)}<small>%</small></div>
      <div class="meter"><i style="width:${Math.round(v * 100)}%"></i></div></div>`;
  }).join("");

  $("#trace").innerHTML = res.trace.map((t) => `<div class="step"><div class="bul"></div>
    <div><b>${esc(t.stage)}</b><code>${esc(JSON.stringify(t.detail).slice(1, -1).replace(/"/g, ""))}</code></div>
    <div class="ms">${t.ms}ms</div></div>`).join("");

  renderSources(res.sources);
}

// one listener, because streaming replaces the citations on every update
$("#answer-wrap").addEventListener("click", (e) => {
  const c = e.target.closest("cite"), el = c && $$("#sources .src")[+c.dataset.ref - 1];
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
});

function renderSources(sources) {
  $("#sources").innerHTML = sources.length ? sources.map((s, i) => {
    const cls = s.label === "relevant" ? "ok" : s.label === "ambiguous" ? "warn" : "bad";
    return `<div class="src"><h4><em>[${i + 1}]</em> ${esc(s.title)}</h4>
      <p>${esc(s.text)}</p>
      <div class="meta"><span class="tag ${cls}">${s.label} ${s.grade}</span>
        <span class="tag">score ${s.score}</span>${s.via_graph ? '<span class="tag info">via graph</span>' : ""}
        <span class="tag">${esc(s.source)}</span>
        <button class="mini" data-more="${i}">expand</button></div></div>`;
  }).join("") : '<div class="empty"><b>No grounded evidence</b>Add documents, or click Fetch PubMed.</div>';
  $$("#sources [data-more]").forEach((b) => b.onclick = () => {
    const p = b.closest(".src").querySelector("p");
    p.classList.toggle("open"); b.textContent = p.classList.contains("open") ? "collapse" : "expand";
  });
}

async function askStream(body) {
  const r = await fetch(`/api/workspaces/${state.ws}/ask/stream`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || `${r.status} ${r.statusText}`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "", text = "", result = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    const lines = (buf + value).split("\n");
    buf = lines.pop();
    const before = text;
    for (const ev of lines.filter(Boolean).map((l) => JSON.parse(l))) {
      if (ev.error) throw new Error(ev.error);
      if (ev.sources) renderSources(ev.sources);
      if (ev.delta) text += ev.delta;
      if (ev.result) result = ev.result;
    }
    if (text !== before && !result) $("#answer-wrap").innerHTML = `<div class="answer">${cited(text)}</div>`;
  }
  if (!result) throw new Error("The connection closed before the answer finished");
  renderAnswer(result, body.query);
  return result;
}

$("#ask-btn").onclick = async () => {
  const q = $("#q").value.trim();
  if (!q) return toast("Type a question first", true);
  if (!state.ws) return toast("Create a workspace first", true);
  const btn = $("#ask-btn"); btn.disabled = true; btn.textContent = "Retrieving…";
  $("#answer-wrap").innerHTML = '<div class="row"><span class="spin"></span><span style="color:var(--text-3)">running pipeline…</span></div>';
  try {
    $("#answer-wrap").setAttribute("aria-busy", "true");
    const res = await askStream({ query: q, k: 6, ...state.opts, history: state.history.slice(-6) });
    if (res?.answer) {
      state.history.push({ role: "user", content: q }, { role: "assistant", content: res.answer });
      renderChatHistory();
    }
    if (res?.crag?.augmented_papers) loadWorkspaces(state.ws);
  } catch (e) {
    $("#answer-wrap").innerHTML = `<div class="empty"><b>Request failed</b>${esc(e.message)}</div>`;
    toast(e.message, true);
  } finally { $("#answer-wrap").removeAttribute("aria-busy"); btn.disabled = false; btn.textContent = "Retrieve & answer"; }
};
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#ask-btn").click(); });

$("#compare-btn").onclick = async () => {
  const q = $("#q").value.trim();
  if (!q) return toast("Type a question first", true);
  if (!state.ws) return toast("Create a workspace first", true);
  const btn = $("#compare-btn"); btn.disabled = true; btn.textContent = "Comparing…";
  $("#answer-wrap").innerHTML = '<div class="row"><span class="spin"></span><span style="color:var(--text-3)">running A/B ablation (Baseline vs Current Pipeline)…</span></div>';
  try {
    const out = await api(`/workspaces/${state.ws}/compare`, {
      body: {
        query: q, k: 6,
        config_a: { hybrid: true, graph: false, rerank: false, decompose: false, hyde: false, crag: false },
        config_b: { ...state.opts },
      },
    });
    const col = (title, r) => `<div class="ab-col">
      <h4><span>${title}</span><span class="tag">${r.total_ms}ms · ${r.sources.length} hits</span></h4>
      <div class="row" style="gap:6px;margin-bottom:8px">
        <span class="tag ok">faith ${(r.metrics.faithfulness * 100).toFixed(0)}%</span>
        <span class="tag info">rel ${(r.metrics.answer_relevance * 100).toFixed(0)}%</span>
        <span class="tag">prec ${(r.metrics.context_precision * 100).toFixed(0)}%</span></div>
      <div class="ab-ans">${cited(r.answer)}</div>
      <div class="chips">${r.sources.map((s, i) => `<span class="tag">[${i + 1}] ${esc(s.title.slice(0, 28))} (${s.grade})</span>`).join("")}</div></div>`;
    $("#answer-wrap").innerHTML = `<span class="eyebrow">Side-by-side RAG ablation</span>
      <div class="ab-grid">${col("A · Baseline", out.a)}${col("B · Current", out.b)}</div>`;
    $("#metrics").innerHTML = [["faithfulness", "Faithfulness (B)"], ["answer_relevance", "Relevance (B)"],
      ["context_precision", "Precision (B)"]].map(([k, label]) => {
      const v = out.b.metrics[k] ?? 0;
      return `<div class="stat"><div class="k">${label}</div><div class="v">${(v * 100).toFixed(0)}<small>%</small></div>
        <div class="meter"><i style="width:${Math.round(v * 100)}%"></i></div></div>`;
    }).join("");
    $("#trace").innerHTML = out.b.trace.map((t) => `<div class="step"><div class="bul"></div>
      <div><b>${esc(t.stage)}</b><code>${esc(JSON.stringify(t.detail).slice(1, -1).replace(/"/g, ""))}</code></div>
      <div class="ms">${t.ms}ms</div></div>`).join("");
    renderSources(out.b.sources);
  } catch (e) {
    $("#answer-wrap").innerHTML = `<div class="empty"><b>Ablation failed</b>${esc(e.message)}</div>`;
    toast(e.message, true);
  } finally { btn.disabled = false; btn.textContent = "A/B ablation"; }
};

/* ---------------- KNOWLEDGE ---------------- */
onShow.knowledge = async () => {
  if (!state.ws) return ($("#docs").innerHTML = '<div class="empty">No workspace.</div>');
  const docs = await api(`/workspaces/${state.ws}/docs`);
  $("#docs").innerHTML = docs.length ? docs.map((d) => `<div class="item">
      <h4>${esc(d.title)}</h4><p class="clamp">${esc(d.text)}</p>
      <div class="meta"><span class="tag">${d.chunks} chunks</span><span class="tag">${d.chars.toLocaleString()} chars</span>
        <span class="tag">${esc(d.source)}</span><span>${esc(d.added)}</span>
        <button class="mini danger" data-del="${d.id}" style="margin-left:auto">remove</button></div></div>`).join("")
    : '<div class="empty"><b>Empty workspace</b>Ingest a document to begin.</div>';
  $$("#docs [data-del]").forEach((b) => b.onclick = async () => {
    await api(`/workspaces/${state.ws}/docs/${b.dataset.del}`, { method: "DELETE" });
    toast("Document removed"); onShow.knowledge(); loadWorkspaces(state.ws);
  });
};
$("#d-add").onclick = async () => {
  const title = $("#d-title").value.trim() || "Untitled", text = $("#d-text").value.trim();
  if (!text) return toast("Paste some text first", true);
  try {
    const r = await api(`/workspaces/${state.ws}/docs`, { body: { title, text } });
    $("#d-title").value = $("#d-text").value = "";
    toast(r.duplicate ? "Already in workspace (skipped duplicate)" : "Ingested & indexed");
    onShow.knowledge(); loadWorkspaces(state.ws);
  } catch (e) { toast(e.message, true); }
};
$("#d-fetch").onclick = async () => {
  const url = $("#d-url").value.trim();
  if (!url) return toast("Enter a URL or DOI first", true);
  const btn = $("#d-fetch"); btn.disabled = true; btn.textContent = "Fetching…";
  try {
    const r = await api("/url", { body: { url } });
    $("#d-title").value = r.title || url;
    $("#d-text").value = r.text;
    $("#d-url").value = "";
    toast("Fetched — review, then click Ingest");
  } catch (e) { toast(e.message, true); }
  finally { btn.disabled = false; btn.textContent = "fetch"; }
};
$("#d-url").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#d-fetch").click(); });
$("#d-file").onclick = () => $("#d-input").click();
const dropzone = $("#dropzone"), batchLog = $("#batch-log");

async function readSingleFile(f) {
  if (/\.pdf$/i.test(f.name)) {
    const r = await api("/pdf", { body: f });
    return { title: f.name.replace(/\.[^.]+$/, ""), text: r.text, source: "pdf", meta: r };
  }
  return { title: f.name.replace(/\.[^.]+$/, ""), text: await f.text(), source: "file", meta: null };
}

async function handleFiles(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  if (!state.ws) return toast("Create a workspace first", true);

  // Single file: load into the editor for quick review (familiar workflow).
  if (files.length === 1) {
    const f = files[0];
    const btn = $("#d-file"), label = btn.textContent;
    btn.disabled = $("#d-add").disabled = true; btn.textContent = "Reading…";
    try {
      const got = await readSingleFile(f);
      $("#d-text").value = got.text;
      $("#d-title").value = got.title;
      const blank = got.meta?.blank ? `, ${got.meta.blank} without text` : "";
      toast(got.meta ? `${got.meta.pages} page${got.meta.pages > 1 ? "s" : ""}${blank} — review, then ingest` : "File loaded — review, then ingest");
    } catch (err) { toast(err.message, true); }
    finally { btn.disabled = $("#d-add").disabled = false; btn.textContent = label; }
    return;
  }

  // Multiple files: extract with live per-file progress and ingest in one atomic batch.
  const btn = $("#d-file"), label = btn.textContent;
  btn.disabled = $("#d-add").disabled = true;
  batchLog.innerHTML = "";
  const ready = [], failed = [];

  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    btn.textContent = `Reading ${i + 1}/${files.length}…`;
    const row = document.createElement("div");
    row.className = "batch-row";
    row.innerHTML = `<span title="${esc(f.name)}">${esc(f.name)}</span><span class="tag info">reading…</span>`;
    batchLog.append(row);
    batchLog.scrollTop = batchLog.scrollHeight;
    try {
      const got = await readSingleFile(f);
      if (!got.text.trim()) throw new Error("empty file");
      ready.push({ title: got.title, text: got.text, source: got.source });
      row.lastElementChild.className = "tag ok";
      row.lastElementChild.textContent = got.meta ? `${got.meta.pages}p ready` : "ready";
    } catch (err) {
      failed.push(f.name);
      row.lastElementChild.className = "tag bad";
      row.lastElementChild.textContent = err.message.slice(0, 32);
    }
  }

  if (ready.length) {
    btn.textContent = `Ingesting ${ready.length}…`;
    try {
      const res = await api(`/workspaces/${state.ws}/docs/batch`, { body: { docs: ready } });
      const parts = [`Ingested ${res.imported}`];
      if (res.skipped) parts.push(`${res.skipped} duplicate${res.skipped === 1 ? "" : "s"} skipped`);
      if (failed.length) parts.push(`${failed.length} failed`);
      toast(parts.join(" · "), failed.length > 0 && res.imported === 0);
      await onShow.knowledge();
      await loadWorkspaces(state.ws);
    } catch (err) {
      toast(err.message, true);
    }
  } else {
    toast("No readable files in that batch", true);
  }
  btn.disabled = $("#d-add").disabled = false;
  btn.textContent = label;
}

$("#d-input").onchange = async (e) => {
  await handleFiles(e.target.files);
  e.target.value = "";
};
dropzone.onclick = () => $("#d-input").click();
dropzone.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#d-input").click(); } };
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => { if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files); });
$("#w-add").onclick = async () => {
  const name = $("#w-name").value.trim();
  if (!name) return toast("Name the workspace", true);
  const w = await api("/workspaces", { body: { name, domain: $("#w-domain").value.trim() || "General" } });
  $("#w-name").value = $("#w-domain").value = "";
  await loadWorkspaces(w.id); refreshActive(); toast("Workspace created");
};
$("#w-del").onclick = async () => {
  if (!state.ws || !confirm("Delete this workspace and all its documents? This cannot be undone.")) return;
  await api(`/workspaces/${state.ws}`, { method: "DELETE" });
  await loadWorkspaces(); refreshActive(); toast("Workspace deleted");
};

/* ---------------- GRAPH (SVG chord diagram: hover to trace, drag to spin, click to pin) ---------------- */
const chord = $("#gchord");
let nodes = [], edges = [], sel = null, hot = null, rot = 0, raf = null, drag = null;
const PALETTE = ["#9a7b0a", "#3a6b63", "#8a5a3c", "#5a5280", "#a33a32", "#4a6b8a", "#6f7a3a"];
const R_IN = 244, R_OUT = 258, R_BAND = 266, R_LABEL = 276, TAU = Math.PI * 2;
const LINKS_PER_NODE = 3;  // drawn at rest; a traced entity shows every link it has
const color = (n) => PALETTE[n.community % PALETTE.length];
const calm = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const pt = (a, r) => `${(Math.sin(a) * r).toFixed(1)} ${(-Math.cos(a) * r).toFixed(1)}`;
const ring = (a0, a1, r0, r1) => {
  const big = a1 - a0 > Math.PI ? 1 : 0;
  return `M${pt(a0, r1)}A${r1} ${r1} 0 ${big} 1 ${pt(a1, r1)}L${pt(a1, r0)}A${r0} ${r0} 0 ${big} 0 ${pt(a0, r0)}Z`;
};
const ribbon = (s, t) => `M${pt(s[0], R_IN - 3)}A${R_IN - 3} ${R_IN - 3} 0 0 1 ${pt(s[1], R_IN - 3)}Q0 0 ${pt(t[0], R_IN - 3)}` +
  `A${R_IN - 3} ${R_IN - 3} 0 0 1 ${pt(t[1], R_IN - 3)}Q0 0 ${pt(s[0], R_IN - 3)}Z`;

onShow.graph = async () => {
  if (!state.ws) return;
  state.graph = await api(`/workspaces/${state.ws}/graph`);
  const { nodes: N, edges: E, communities } = state.graph;
  cancelAnimationFrame(raf);
  sel = hot = null; rot = 0; chord.classList.remove("tracing");
  $("#gsel").innerHTML = '<div class="empty">Click an entity.</div>';
  if (!N.length) {
    nodes = edges = []; chord.innerHTML = "";
    $("#gcomm").innerHTML = '<div class="empty"><b>No entities yet</b>Ingest documents first.</div>';
    return;
  }
  layoutChord(N, E);
  renderChord();
  $("#gcomm").innerHTML = communities.map((c) => `<button class="item comm" data-c="${c.id}" style="--c:${PALETTE[c.id % PALETTE.length]}"><h4>
      <span class="dot"></span>Cluster ${c.id + 1}</h4><p>${c.members.map(esc).join(" · ")}</p></button>`).join("");
  $$("#gcomm .comm").forEach((b) => {
    b.onpointerenter = b.onfocus = () => trace((n) => n.community === +b.dataset.c);
    b.onpointerleave = b.onblur = () => trace(sel);
  });
};

function layoutChord(N, E) {
  nodes = N.map((n) => ({ ...n, load: 0, ends: [] })).sort((a, b) => a.community - b.community || b.size - a.size);
  const byId = new Map(nodes.map((n) => [n.id, n]));
  edges = E.map((e) => ({ a: byId.get(e.source), b: byId.get(e.target), w: e.weight })).filter((e) => e.a && e.b && e.a !== e.b);
  edges.sort((x, y) => y.w - x.w);
  const kept = new Map(nodes.map((n) => [n, 0]));
  for (const e of edges) {
    // edges are strongest first, so this keeps each entity's top links
    e.drawn = kept.get(e.a) < LINKS_PER_NODE || kept.get(e.b) < LINKS_PER_NODE;
    if (!e.drawn) continue;
    for (const n of [e.a, e.b]) { kept.set(n, kept.get(n) + 1); n.load += e.w; n.ends.push(e); }
  }
  const groups = new Set(nodes.map((n) => n.community)).size;
  const gap = 0.008, groupGap = groups > 1 ? 0.06 : 0;
  const room = TAU - gap * nodes.length - groupGap * groups;
  const total = nodes.reduce((s, n) => s + n.load, 0);
  let a = groupGap / 2;
  nodes.forEach((n, i) => {
    if (i && nodes[i - 1].community !== n.community) a += groupGap;
    // most of the circle is shared equally so every label has room to be read
    const span = room * (total ? 0.7 / nodes.length + 0.3 * n.load / total : 1 / nodes.length);
    n.a0 = a; n.a1 = a + span; n.mid = a + span / 2;
    a += span + gap;
  });
  for (const n of nodes) {
    const far = (e) => ((e.a === n ? e.b : e.a).mid - n.mid + TAU) % TAU;
    let p = n.a0;
    for (const e of n.ends.sort((x, y) => far(y) - far(x))) {
      const q = p + (n.a1 - n.a0) * e.w / n.load;
      e[e.a === n ? "sa" : "sb"] = [p, q]; p = q;
    }
  }
}

function renderChord() {
  const label = (s) => esc(s.length > 18 ? s.slice(0, 17) + "…" : s);
  const bands = [...new Set(nodes.map((n) => n.community))].map((c) => nodes.filter((n) => n.community === c));
  chord.innerHTML = `<g class="spin-g">
    <g class="bands">${bands.map((g) => `<path d="${ring(g[0].a0, g.at(-1).a1, R_OUT + 3, R_BAND)}" fill="${color(g[0])}"/>`).join("")}</g>
    <g class="ribbons">${edges.map((e, i) => e.drawn ? `<path data-e="${i}" d="${ribbon(e.sa, e.sb)}" fill="${color(e.a)}"
      style="--d:${(500 + 600 * e.a.mid / TAU).toFixed(0)}ms"><title>${esc(e.a.id)} – ${esc(e.b.id)}: ${e.w} shared chunks</title></path>` : "").join("")}</g>
    <g class="flows"></g>
    <g class="arcs">${nodes.map((n, i) => `<g data-n="${i}" style="--d:${(650 * n.mid / TAU).toFixed(0)}ms">
      <path class="hit" d="${ring(n.a0, n.a1, R_IN, R_LABEL + 70)}"/>
      <path class="arc" d="${ring(n.a0, n.a1, R_IN, R_OUT)}" fill="${color(n)}"/>
      <text>${label(n.id)}</text></g>`).join("")}</g></g>`;
  chord.classList.remove("draw"); void chord.getBoundingClientRect(); chord.classList.add("draw");
  setRot(0);
  if (!calm()) spinTo(0, -0.7);
}

function setRot(r) {
  rot = r;
  $(".spin-g", chord).setAttribute("transform", `rotate(${(r * 180 / Math.PI).toFixed(2)})`);
  $$(".arcs text", chord).forEach((t, i) => {
    const m = nodes[i].mid, screen = ((m + r) % TAU + TAU) % TAU, flip = screen > Math.PI;
    t.setAttribute("transform", `rotate(${(m * 180 / Math.PI - 90).toFixed(2)}) translate(${R_LABEL} 0)${flip ? " rotate(180)" : ""}`);
    t.setAttribute("text-anchor", flip ? "end" : "start");
  });
}
/* critically damped spring, always starts from the live angle so it can be interrupted */
function spinTo(target, from = rot, v = 0) {
  cancelAnimationFrame(raf);
  if (calm()) return setRot(target);
  let x = from - target, last = performance.now();
  const w = 6;
  const loop = (now) => {
    const dt = Math.max(0, Math.min(0.032, (now - last) / 1000)); last = now;
    v += (-w * w * x - 2 * w * v) * dt; x += v * dt;
    if (Math.abs(x) < 0.0005 && Math.abs(v) < 0.005) return setRot(target);
    setRot(target + x); raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
}
function coast(v) {
  cancelAnimationFrame(raf);
  if (calm()) return;
  let last = performance.now();
  const loop = (now) => {
    const dt = Math.max(0, Math.min(0.032, (now - last) / 1000)); last = now;
    v *= Math.exp(-2.6 * dt); setRot(rot + v * dt);
    if (Math.abs(v) > 0.02) raf = requestAnimationFrame(loop);
  };
  raf = requestAnimationFrame(loop);
}

/* focus: a node, a predicate over nodes (cluster), or null */
function trace(focus) {
  hot = focus;
  const inSet = typeof focus === "function" ? focus : (n) => n === focus;
  const touches = (e) => typeof focus === "function" ? inSet(e.a) && inSet(e.b) : inSet(e.a) || inSet(e.b);
  chord.classList.toggle("tracing", !!focus);
  const lit = new Set();
  if (focus) {
    nodes.filter(inSet).forEach((n) => lit.add(n));
    edges.filter(touches).forEach((e) => { lit.add(e.a); lit.add(e.b); });
  }
  $$(".arcs > g", chord).forEach((g, i) => {
    g.classList.toggle("lit", lit.has(nodes[i]));
    g.classList.toggle("sel", nodes[i] === sel);
  });
  $$(".ribbons path", chord).forEach((p) => p.classList.toggle("lit", !!focus && touches(edges[+p.dataset.e])));
  $(".flows", chord).innerHTML = focus && typeof focus !== "function" ? edges.filter(touches).map((e) => {
    const o = e.a === focus ? e.b : e.a;
    return `<path d="M${pt(focus.mid, R_IN - 3)}Q0 0 ${pt(o.mid, R_IN - 3)}" stroke="${color(focus)}" stroke-width="${Math.min(3, 0.7 + e.w * 0.2)}"/>`;
  }).join("") : "";
}
const angleAt = (e) => {
  const r = chord.getBoundingClientRect();
  return Math.atan2(e.clientX - r.left - r.width / 2, -(e.clientY - r.top - r.height / 2));
};
const nodeAt = (e) => { const g = e.target.closest?.("[data-n]"); return g ? nodes[+g.dataset.n] : null; };
chord.addEventListener("pointerdown", (e) => {
  if (!nodes.length || e.button) return;
  cancelAnimationFrame(raf);
  drag = { n: nodeAt(e), x: e.clientX, y: e.clientY, a: angleAt(e), t: e.timeStamp, v: 0, moved: false };
});
chord.addEventListener("pointermove", (e) => {
  if (!drag) { const n = nodeAt(e) || sel; if (n !== hot) trace(n); return; }
  if (!drag.moved) {
    if (Math.hypot(e.clientX - drag.x, e.clientY - drag.y) < 5) return;
    drag.moved = true; drag.a = angleAt(e); chord.setPointerCapture(e.pointerId);
  }
  const a = angleAt(e), d = Math.atan2(Math.sin(a - drag.a), Math.cos(a - drag.a));
  const dt = Math.max(1, e.timeStamp - drag.t) / 1000;
  drag.v = 0.6 * drag.v + 0.4 * (d / dt); drag.a = a; drag.t = e.timeStamp;
  setRot(rot + d);
});
chord.addEventListener("pointerup", (e) => {
  const was = drag; drag = null;
  if (!was) return;
  if (was.moved) { if (e.timeStamp - was.t < 80) coast(Math.max(-8, Math.min(8, was.v))); }
  else if (was.n && was.n !== sel) selectNode(was.n);
  else clearNode();
});
chord.addEventListener("pointercancel", () => { drag = null; });
chord.addEventListener("pointerleave", () => { if (drag && !drag.moved) drag = null; if (!drag && nodes.length) trace(sel); });
chord.addEventListener("keydown", (e) => {
  if (!nodes.length) return;
  const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
  if (step) { e.preventDefault(); selectNode(nodes[(nodes.indexOf(sel) + step + nodes.length) % nodes.length]); }
  if (e.key === "Escape") clearNode();
});
function clearNode() {
  sel = null; trace(null);
  $("#gsel").innerHTML = '<div class="empty">Click an entity.</div>';
}
function selectNode(n) {
  sel = n; trace(n);
  // bring it to 12 o'clock the short way round
  spinTo(rot + Math.atan2(Math.sin(-n.mid - rot), Math.cos(-n.mid - rot)));
  const deg = edges.filter((e) => e.a === n || e.b === n);
  const nbrs = deg.slice(0, 8).map((e) => [e.a === n ? e.b : e.a, e.w]);
  $("#gsel").innerHTML = `<div class="stat"><div class="k">Entity</div><div class="v" style="font-size:1.15rem">${esc(n.id)}</div></div>
    <div class="row" style="margin-top:12px"><span class="tag ok">${n.size} chunks</span><span class="tag">${deg.length} links</span>
      <span class="tag info">cluster ${n.community + 1}</span></div>
    <p class="sub" style="margin:14px 0 6px">Strongest co-occurrences</p>
    <div class="chips">${nbrs.map(([o, w]) => `<button class="chip" data-n="${nodes.indexOf(o)}">${esc(o.id)} · ${w}</button>`).join("") || "—"}</div>
    <button class="mini" style="margin-top:14px" id="gsel-ask">Ask about ${esc(n.id)}</button>`;
  $$("#gsel [data-n]").forEach((b) => b.onclick = () => selectNode(nodes[+b.dataset.n]));
  $("#gsel-ask").onclick = () => {
    $("#q").value = `What does the corpus say about ${n.id}?`;
    show("ask", $('button[data-view="ask"]', tabs)); $("#ask-btn").click();
  };
}

/* ---------------- LITERATURE ---------------- */
$("#lit-btn").onclick = async () => {
  const q = $("#lit-q").value.trim();
  if (!q) return toast("Enter a search query", true);
  const sources = [$("#lit-pubmed").checked && "pubmed", $("#lit-biorxiv").checked && "biorxiv"].filter(Boolean);
  if (!sources.length) return toast("Pick at least one source", true);
  const btn = $("#lit-btn"); btn.disabled = true; btn.textContent = "Searching…";
  $("#lit-results").innerHTML = '<div class="row"><span class="spin"></span><span style="color:var(--text-3)">querying ' + sources.join(" + ") + "…</span></div>";
  try {
    const days = $("#lit-days").value;
    const res = await api("/literature", { body: { query: q, sources, max_results: +$("#lit-max").value || 15, days_back: days ? +days : null } });
    state.lit = res.papers;
    $("#lit-count").textContent = `${res.papers.length} papers` + (Object.keys(res.errors).length ? ` · errors: ${Object.entries(res.errors).map(([k, v]) => k + " (" + v + ")").join(", ")}` : "");
    $("#lit-import").hidden = !res.papers.length;
    $("#lit-results").innerHTML = res.papers.length ? res.papers.map((p, i) => `<div class="item" style="animation-delay:${i * 28}ms">
        <h4>${esc(p.title)}</h4><p class="clamp">${esc(p.abstract || "No abstract available.")}</p>
        <div class="meta"><span class="tag ${p.source === "pubmed" ? "info" : "warn"}">${esc(p.source)}</span>
          <span>${esc(p.publication_date || "—")}</span><span>${esc((p.authors || []).slice(0, 3).join(", "))}</span>
          <a class="mini" style="margin-left:auto;text-decoration:none" target="_blank" rel="noopener"
             href="${p.source === "pubmed" ? "https://pubmed.ncbi.nlm.nih.gov/" + encodeURIComponent(p.paper_id) : "https://doi.org/" + encodeURIComponent(p.paper_id)}">open ↗</a>
        </div></div>`).join("") : '<div class="empty"><b>No results</b>Try a broader query or another source.</div>';
  } catch (e) {
    $("#lit-results").innerHTML = `<div class="empty"><b>Search failed</b>${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Search"; }
};
$("#lit-q").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#lit-btn").click(); });
$("#lit-import").onclick = async () => {
  const res = await api(`/workspaces/${state.ws}/import`, { body: { papers: state.lit } });
  const skip = res.skipped ? ` (${res.skipped} duplicate${res.skipped === 1 ? "" : "s"} skipped)` : "";
  toast(`Imported ${res.imported} abstracts${skip}`);
  $("#lit-ask").hidden = false;
  loadWorkspaces(state.ws);
};
$("#lit-ask").onclick = () => {
  const q = $("#lit-q").value.trim();
  if (q) $("#q").value = `Summarize the key findings and mechanisms related to ${q}`;
  show("ask", $('button[data-view="ask"]', tabs));
};

/* ---------------- MOLECULES ---------------- */
const EXAMPLES = { Aspirin: "CC(=O)Oc1ccccc1C(=O)O", Caffeine: "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  Imatinib: "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
  Paclitaxel: "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(c1ccccc1)NC(=O)c1ccccc1)O)O)OC(=O)c1ccccc1)(CO4)OC(C)=O)O)C)OC(C)=O",
  Ibuprofen: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" };
$("#mol-chips").innerHTML = Object.keys(EXAMPLES).map((k) => `<button class="chip" data-s="${EXAMPLES[k]}">${k}</button>`).join("");
$("#mol-chips").addEventListener("click", (e) => {
  const b = e.target.closest(".chip"); if (!b) return;
  $("#smiles").value = b.dataset.s; $("#mol-btn").click();
});
$("#mol-btn").onclick = async () => {
  const smiles = $("#smiles").value.trim();
  if (!smiles) return toast("Enter a SMILES string", true);
  const btn = $("#mol-btn"); btn.disabled = true; btn.textContent = "Running…";
  try {
    const r = await api("/admet?smiles=" + encodeURIComponent(smiles));
    const cls = r.permeability === "high" ? "ok" : r.permeability === "moderate" ? "warn" : "bad";
    $("#mol-out").innerHTML = `<div class="grid g3" style="gap:10px">
        <div class="stat" style="grid-column:span 2"><div class="k">${esc(r.endpoint)}</div>
          <div class="v">${r.prediction} <small>${esc(r.unit)}</small></div>
          <div class="row" style="margin-top:10px"><span class="tag ${cls}">${r.permeability} permeability</span>
            <span class="tag">${r.graph.atoms} atoms · ${r.graph.bonds} bonds</span></div></div>
        ${r.lipinski_violations.length
          ? `<div class="stat"><div class="k">Rule-of-five flags</div><div class="row" style="margin-top:10px">${r.lipinski_violations.map((f) => `<span class="tag bad">${esc(f)}</span>`).join("")}</div></div>`
          : `<div class="stat"><div class="k">Rule-of-five</div><div class="v" style="font-size:1.1rem">no violations</div></div>`}
      </div>
      <table style="margin-top:18px"><thead><tr><th>Descriptor</th><th style="text-align:right">Value</th></tr></thead><tbody>
      ${Object.entries(r.descriptors).map(([k, v]) => `<tr><td>${esc(k.replace(/_/g, " "))}</td><td class="num" style="text-align:right">${v}</td></tr>`).join("")}
      </tbody></table>`;
    $("#mol-model").innerHTML = `<div class="stat"><div class="k">Architecture</div><div class="v" style="font-size:1rem">${esc(r.model.architecture)}</div></div>
      <div class="row" style="margin-top:12px"><span class="tag">best epoch ${r.model.trained_epoch ?? "—"}</span></div>
      <table style="margin-top:14px"><thead><tr><th>Validation</th><th style="text-align:right">Value</th></tr></thead><tbody>
      ${Object.entries(r.model.val_metrics).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num" style="text-align:right">${v}</td></tr>`).join("") || '<tr><td colspan="2">—</td></tr>'}
      </tbody></table>`;
  } catch (e) {
    $("#mol-out").innerHTML = `<div class="empty"><b>Prediction failed</b>${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Predict"; }
};
$("#smiles").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#mol-btn").click(); });

/* ---------------- SCREENING ---------------- */
const LIBRARY = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
  "CC(=O)Nc1ccc(O)cc1", "CN1CCC[C@H]1c1cccnc1", "OC(=O)c1ccccc1O",
  "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"];
$("#lib").value = LIBRARY.join("\n");
const heat = (v) => `color-mix(in srgb, var(--bamboo) ${Math.round(v * 92)}%, transparent)`;
$("#lib-btn").onclick = async () => {
  const smiles = $("#lib").value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (!smiles.length) return toast("Add at least one SMILES", true);
  const btn = $("#lib-btn"); btn.disabled = true; btn.textContent = "Screening…";
  try {
    const r = await api("/screen", { body: { smiles, n_select: +$("#lib-pick").value || 0,
      nbits: +$("#lib-bits").value, radius: +$("#lib-radius").value } });
    const picked = new Set(r.selected);
    $("#lib-out").innerHTML = `<div class="row" style="justify-content:space-between;margin-bottom:12px">
        <span class="tag info">${esc(r.backend)}</span>
        <span class="eyebrow">${r.ms} ms · ${r.molecules.length} molecules · ${r.nbits} bits · r=${r.radius}</span></div>
      ${r.failed.length ? `<div class="row" style="margin-bottom:10px">${r.failed.map((f) => `<span class="tag bad">unparsed: ${esc(f)}</span>`).join("")}</div>` : ""}
      <table><thead><tr><th>#</th><th>SMILES</th><th style="text-align:right">Bits</th>
        <th style="text-align:right">Max sim</th><th style="text-align:right">Mean sim</th><th></th></tr></thead><tbody>
      ${r.molecules.map((m, i) => `<tr${picked.has(i) ? ' style="background:color-mix(in srgb,var(--bamboo) 9%,transparent)"' : ""}>
          <td class="num">${i + 1}</td>
          <td style="font-family:var(--font-mono);font-size:.76rem;max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(m.smiles)}">${esc(m.smiles)}</td>
          <td class="num" style="text-align:right">${m.bits_set}</td>
          <td class="num" style="text-align:right">${m.max_similarity}</td>
          <td class="num" style="text-align:right">${m.mean_similarity}</td>
          <td style="text-align:right">${picked.has(i) ? '<span class="tag ok">diverse pick</span>' : ""}</td></tr>`).join("")}
      </tbody></table>`;
    $("#lib-matrix").innerHTML = `<div style="overflow:auto"><table style="min-width:0"><tbody>
      ${r.matrix.map((row, i) => `<tr><td class="num" style="padding:4px 8px;color:var(--text-3)">${i + 1}</td>
        ${row.map((v, j) => `<td class="num" title="${i + 1} ↔ ${j + 1}: ${v}" style="padding:4px;text-align:center;
          background:${i === j ? "transparent" : heat(v)};border-radius:5px;font-size:.72rem">${i === j ? "·" : v.toFixed(2)}</td>`).join("")}
      </tr>`).join("")}</tbody></table></div>`;
    $("#lib-pairs").innerHTML = r.nearest_pairs.length ? r.nearest_pairs.slice(0, 6).map((p) => `<div class="item">
        <div class="row" style="justify-content:space-between"><span class="tag ${p.similarity > 0.5 ? "warn" : "info"}">${p.similarity}</span></div>
        <p style="font-family:var(--font-mono);font-size:.72rem;margin-top:8px">${esc(p.a)}<br>${esc(p.b)}</p></div>`).join("")
      : '<div class="empty">Need at least two molecules.</div>';
  } catch (e) {
    $("#lib-out").innerHTML = `<div class="empty"><b>Screen failed</b>${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Screen"; }
};

/* ---------------- REPURPOSING ---------------- */
onShow.repurposing = async () => {
  if (state.repLoaded) return;
  try {
    const r = await api("/repurposing");
    $("#rep-corpus").innerHTML = `<span class="tag ok">${r.corpus} mechanism notes</span>`;
    if (r.missing.length) {
      $("#rep-status").innerHTML = `<p class="sub" style="margin:0 0 10px">Still needed</p><div class="list">
        ${r.missing.map((m) => `<div class="item"><h4>${esc(m.what)}</h4>
          <p style="font-family:var(--font-mono);font-size:.76rem;color:var(--bamboo)">${esc(m.fix)}</p>
          <div class="meta"><span style="word-break:break-all">${esc(m.note)}</span></div></div>`).join("")}</div>`;
      $("#rep-btn").disabled = true;
      return;
    }
    const c = r.card, h = c.held_out || {}, b = c.baseline_held_out;
    const pct = (v) => (v == null ? "—" : v.toFixed(3));
    $("#rep-status").innerHTML = `<span class="eyebrow">Model card</span>
      <div class="grid g3" style="gap:10px;margin-top:12px">
        <div class="stat"><div class="k">Graph</div><div class="v" style="font-size:1.1rem">${c.graph.nodes.toLocaleString()}<small> nodes</small></div>
          <div class="row" style="margin-top:8px"><span class="tag">${c.graph.compounds} compounds</span><span class="tag">${c.graph.diseases} diseases</span>
          ${c.graph_mode ? `<span class="tag ok">${(c.graph.message_passing_edges / 1e6).toFixed(2)}M msg edges · ${c.graph.relations} relations</span>` : ""}</div></div>
        <div class="stat"><div class="k">Trained on</div><div class="v" style="font-size:1.1rem">${c.training.positive_pairs}<small> treats edges</small></div>
          <div class="row" style="margin-top:8px"><span class="tag">1:${c.training.negatives_per_positive} negatives</span>
            <span class="tag">${c.training.best_epoch ? `best of ${c.training.epochs}: ep ${c.training.best_epoch}` : `${c.training.epochs ?? "—"} epochs`}</span></div></div>
        ${c.ranking && c.ranking.filtered
          ? `<div class="stat"><div class="k">Held-out MRR</div><div class="v" style="font-size:1.1rem">${pct(c.ranking.model.mrr)}</div>
              <div class="meter"><i style="width:${Math.round(c.ranking.model.mrr * 100)}%"></i></div>
              <div class="row" style="margin-top:8px"><span class="tag ok">hits@1 ${c.ranking.model.hits_at_1}</span>
                <span class="tag">${c.ranking.pairs} test pairs</span></div></div>`
          : `<div class="stat"><div class="k">Held-out AUC</div><div class="v" style="font-size:1.1rem">${pct(h.auc)}</div>
              <div class="meter"><i style="width:${Math.round((h.auc || 0) * 100)}%"></i></div>
              <div class="row" style="margin-top:8px"><span class="tag">AP ${pct(h.ap)}</span>
                <span class="tag">${h.test_size ?? "—"} test pairs</span></div></div>`}
      </div>
      <div class="row" style="margin-top:10px"><span class="tag info">${esc(c.objective || "")}</span>
      ${b && !(c.ranking && c.ranking.filtered) ? `<span class="tag ${(h.auc || 0) > (b.auc || 0) ? "ok" : "warn"}">
          R-GCN ${pct(h.auc)} vs embedding baseline ${pct(b.auc)} AUC</span>` : ""}</div>
      ${c.ranking ? (() => {
        const k = c.ranking;
        const rows = [["This model", k.model], ["Popularity only", k.popularity_control], ["Random", k.random_control]]
          .filter(([, m]) => m);
        return `<p class="sub" style="margin:18px 0 8px">Ranking all ${k.diseases} diseases for one drug —
            the task this tab performs, over ${k.pairs} ${k.filtered ? "held-out" : "known"} indications${k.filtered ? ", filtered" : ""}</p>
          <table><thead><tr><th>Scorer</th><th style="text-align:right">Median rank</th>
            <th style="text-align:right">Hits@10</th><th style="text-align:right">MRR</th></tr></thead><tbody>
          ${rows.map(([label, m], i) => `<tr${i === 0 ? ' style="font-weight:600"' : ""}>
            <td>${label}${i === 1 && !k.beats_popularity ? ' <span class="tag warn">wins</span>' : ""}</td>
            <td class="num" style="text-align:right">${m.median_rank}</td>
            <td class="num" style="text-align:right">${m.hits_at_10}</td>
            <td class="num" style="text-align:right">${m.mrr}</td></tr>`).join("")}
          </tbody></table>`;
      })() : ""}
      <div class="item" style="margin-top:12px;border-left:2px solid var(--${c.ranking && !c.ranking.beats_popularity ? "amber" : "bamboo"})">
        <h4>${c.ranking && !c.ranking.beats_popularity ? "Read this before using a score" : "How a score is produced"}</h4>
        <p>${esc(c.caveat)}</p>
        <div class="meta"><span class="tag">${esc(c.architecture)}</span>
          <span class="tag">${c.parameters.toLocaleString()} params</span>
          <span class="tag">${esc(c.checkpoint)}</span></div></div>`;
    const [drugs, dis] = await Promise.all([api("/repurposing/vocab?kind=drug&limit=1600"), api("/repurposing/vocab?kind=disease&limit=200")]);
    $("#rep-drugs").innerHTML = drugs.matches.map((d) => `<option value="${esc(d)}">`).join("");
    $("#rep-diseases").innerHTML = dis.matches.map((d) => `<option value="${esc(d)}">`).join("");
    $("#rep-chips").innerHTML = ["Imatinib", "Metformin", "Aspirin", "Dexamethasone"]
      .filter((d) => drugs.matches.includes(d)).map((d) => `<button class="chip">${esc(d)}</button>`).join("");
    $("#rep-chips").onclick = (e) => {
      if (e.target.classList.contains("chip")) { $("#rep-drug").value = e.target.textContent; $("#rep-btn").click(); }
    };
    state.repGraphMode = c.graph_mode;
    state.repLoaded = true;
  } catch (e) {
    $("#rep-status").innerHTML = `<div class="empty"><b>Module unavailable</b>${esc(e.message)}</div>`;
  }
};
$("#rep-btn").onclick = async () => {
  const drug = $("#rep-drug").value.trim(), disease = $("#rep-disease").value.trim();
  if (!drug) return toast("Name a compound", true);
  const btn = $("#rep-btn"); btn.disabled = true; btn.textContent = "Ranking…";
  try {
    const r = await api(`/repurposing/predict?drug=${encodeURIComponent(drug)}${disease ? "&disease=" + encodeURIComponent(disease) : ""}&top_k=10`);
    $("#rep-out").innerHTML = `<div class="row" style="justify-content:space-between;margin-bottom:12px">
        <span class="tag info">${esc(r.drug)}</span>
        <span class="eyebrow">${r.known_indications.length} known indication${r.known_indications.length === 1 ? "" : "s"} in Hetionet</span></div>
      <table><thead><tr><th>Disease</th><th style="text-align:right">Score</th><th style="text-align:right">In Hetionet</th><th></th></tr></thead><tbody>
      ${r.results.map((x) => `<tr><td>${esc(x.disease)}</td>
        <td class="num" style="text-align:right">${x.score.toFixed(3)}</td>
        <td style="text-align:right">${x.known ? '<span class="tag ok">known treats edge</span>' : '<span class="tag">unseen pair</span>'}</td>
        <td style="text-align:right"><button class="mini" data-ev-drug="${esc(r.drug)}" data-ev-dis="${esc(x.disease)}">evidence →</button></td></tr>`).join("")}
      </tbody></table>
      <p class="sub" style="margin-top:12px">${state.repGraphMode
        ? "Scores come from message passing over Hetionet minus treats edges, but 137 diseases and 528 training edges is a small world — treat a high-scoring unseen pair as a hypothesis to check, not a finding."
        : "Scores lean on how often a disease appears among the 755 training edges, so a high score on an unseen pair is a frequency effect, not a discovery."}</p>`;
    $$("#rep-out [data-ev-drug]").forEach((b) => b.onclick = () => {
      $("#q").value = `What mechanistic or clinical evidence links ${b.dataset.evDrug} to ${b.dataset.evDis}?`;
      const autoBox = $('#opts input[data-opt="auto_pubmed"]');
      if (autoBox && !autoBox.checked) { autoBox.checked = true; state.opts.auto_pubmed = true; }
      show("ask", $('button[data-view="ask"]', tabs));
      $("#ask-btn").click();
    });
  } catch (e) {
    $("#rep-out").innerHTML = `<div class="empty"><b>Prediction failed</b>${esc(e.message)}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Rank"; }
};
$("#rep-import").onclick = async () => {
  if (!state.ws) return toast("No workspace selected", true);
  try {
    const r = await api(`/workspaces/${state.ws}/seed/repurposing`, { body: {} });
    const skip = r.skipped ? ` (${r.skipped} already present)` : "";
    toast(`Imported ${r.imported} mechanism notes${skip}`); loadWorkspaces(state.ws);
  } catch (e) { toast(e.message, true); }
};

/* ---------------- CAUSAL ---------------- */
let schema = null;
const seedEdges = [
  { source: "Imatinib", relation: "INHIBITS", target: "BCR-ABL", evidence: "RCT", year: 2001, confidence: 0.95 },
  { source: "BCR-ABL", relation: "LEADS_TO", target: "CML remission", evidence: "clinical_observational", year: 2003, confidence: 0.9 },
];
function edgeRow(e, i) {
  const rel = (schema?.relations || ["INHIBITS"]).map((r) => `<option ${r === e.relation ? "selected" : ""}>${r}</option>`).join("");
  const ev = (schema?.evidence || [{ type: "literature_mining" }]).map((x) => `<option value="${x.type}" ${x.type === e.evidence ? "selected" : ""}>${x.type}</option>`).join("");
  return `<div class="item" data-i="${i}">
    <div class="row" style="gap:8px">
      <input data-f="source" value="${esc(e.source)}" placeholder="cause" style="flex:1;min-width:110px">
      <select data-f="relation" style="width:170px">${rel}</select>
      <input data-f="target" value="${esc(e.target)}" placeholder="effect" style="flex:1;min-width:110px">
    </div>
    <div class="row" style="gap:8px;margin-top:8px">
      <select data-f="evidence" style="flex:1;min-width:150px">${ev}</select>
      <input data-f="year" value="${esc(e.year)}" placeholder="year" style="width:92px">
      <input data-f="confidence" value="${esc(e.confidence)}" placeholder="conf" style="width:82px">
      <button class="mini danger" data-rm="${i}">remove</button>
    </div></div>`;
}
function renderEdges() {
  $("#edges").innerHTML = seedEdges.map(edgeRow).join("");
  $$("#edges [data-rm]").forEach((b) => b.onclick = () => { seedEdges.splice(+b.dataset.rm, 1); renderEdges(); });
}
$("#edge-add").onclick = () => {
  const last = seedEdges[seedEdges.length - 1];
  seedEdges.push({ source: last?.target || "", relation: "LEADS_TO", target: "", evidence: "literature_mining", year: new Date().getFullYear(), confidence: 0.7 });
  renderEdges();
};
function collectEdges() {
  return $$("#edges .item").map((row) => {
    const o = {};
    $$("[data-f]", row).forEach((el) => (o[el.dataset.f] = el.value));
    return { ...o, year: +o.year || null, confidence: Math.min(1, Math.max(0, +o.confidence || 0.7)) };
  }).filter((e) => e.source && e.target);
}
$("#causal-btn").onclick = async () => {
  const edges = collectEdges();
  if (!edges.length) return toast("Add at least one complete hop", true);
  try {
    const r = await api("/causal/score", { body: { edges } });
    $("#causal-out").innerHTML = `<div class="stat"><div class="k">Causal path score</div>
        <div class="v">${r.score}<small> / 1.0 · ${r.hops} hops</small></div>
        <div class="meter"><i style="width:${Math.round(r.score * 100)}%"></i></div></div>
      <p class="sub" style="margin:14px 0 10px;font-family:var(--font-mono);color:var(--text-2)">${esc(r.chain)}</p>
      <div class="grid g3" style="gap:10px">${Object.entries(r.components).map(([k, v]) =>
        `<div class="stat"><div class="k">${esc(k.replace(/_/g, " "))}</div><div class="v" style="font-size:1.25rem">${v}</div>
          <div class="meter"><i style="width:${Math.round(v * 100)}%"></i></div></div>`).join("")}</div>`;
  } catch (e) { toast(e.message, true); }
};
onShow.causal = async () => {
  if (schema) return;
  try {
    schema = await api("/causal/schema");
    $("#causal-schema").innerHTML = `<table><thead><tr><th>Evidence type</th><th style="text-align:right">Weight</th></tr></thead><tbody>
      ${schema.evidence.map((e) => `<tr><td>${esc(e.type.replace(/_/g, " "))}</td><td class="num" style="text-align:right">${e.weight}</td></tr>`).join("")}</tbody></table>
      <p class="sub" style="margin:16px 0 8px">Scoring weights</p>
      <div class="chips">${Object.entries(schema.weights).map(([k, v]) => `<span class="tag">${esc(k)} ${v}</span>`).join("")}</div>
      <p class="sub" style="margin:16px 0 8px">${schema.relations.length} relation types · ${schema.node_types.length} node types</p>
      <div class="chips">${schema.relations.slice(0, 12).map((r) => `<span class="tag info">${esc(r)}</span>`).join("")}</div>`;
    renderEdges();
  } catch (e) {
    $("#causal-schema").innerHTML = `<div class="empty"><b>Module unavailable</b>${esc(e.message)}</div>`;
    renderEdges();
  }
};

/* ---------------- LABS / health ---------------- */
const STACK = [
  ["Contextual chunking", "Each chunk is indexed with its document title and lead sentence, so retrieval keeps the surrounding context."],
  ["Query planning", "Decomposition into sub-queries plus HyDE — a hypothetical answer used as an extra dense probe."],
  ["Hybrid + RRF", "BM25 lexical and dense vector lists fused with reciprocal rank fusion across every sub-query."],
  ["GraphRAG expansion", "Entity co-occurrence graph: query entities and their strongest neighbours boost their chunks."],
  ["MMR rerank", "Maximal marginal relevance trades relevance against redundancy so passages do not repeat each other."],
  ["Corrective RAG", "Every passage is graded relevant / ambiguous / irrelevant; irrelevant ones are dropped before generation."],
  ["Grounded generation", "Claude, OpenAI, Gemini, or local Ollama answer with inline citations; extractive fallback needs no key."],
  ["Self-reflection", "Polarity-aware & semantic faithfulness, answer relevance and rank-aware context precision per run."],
];
async function health() {
  try {
    const h = await api("/health");
    if (h.plugins === "core" && localStorage.getItem("pandarag-bio") == null) setBioLabs(false);
    $("#llm-pill").innerHTML = h.llm
      ? `<span class="dot"></span>${esc(h.provider)} · ${esc(h.model)}`
      : '<span class="dot off"></span>extractive mode';
    $("#llm-pill").title = h.llm
      ? `Provider: ${h.provider} (${h.model}) · Embedder: ${h.embedder}`
      : `Set ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL (Ollama), or GEMINI_API_KEY · Embedder: ${h.embedder}`;
    $("#labs").innerHTML = h.labs.map((l) => {
      const live = l.state === "live";
      return `<div class="stat"><div class="row" style="justify-content:space-between">
          <span class="k">${esc(l.name)}</span><span class="tag ${live ? "ok" : "warn"}">${esc(l.state)}</span></div>
        <div class="v" style="font-size:.95rem;line-height:1.4">${esc(l.description)}</div>
        <code style="display:block;margin-top:10px;font:500 .68rem/1.4 var(--font-mono);color:var(--text-3)">${esc(l.repo)}</code></div>`;
    }).join("");
  } catch (e) { $("#llm-pill").innerHTML = '<span class="dot off"></span>api offline'; }
}
$("#stack").innerHTML = STACK.map(([k, v], i) =>
  `<div class="stat"><div class="k">${String(i + 1).padStart(2, "0")} · ${esc(k)}</div>
    <div style="margin-top:9px;font-size:.84rem;color:var(--text-2);line-height:1.55">${esc(v)}</div></div>`).join("");

/* ---------------- boot ---------------- */
(async () => {
  moveGlider($('button[aria-selected="true"]', tabs), false);
  await loadWorkspaces();
  await health();
})();
