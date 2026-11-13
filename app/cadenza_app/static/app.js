"use strict";
const $ = (s, el = document) => el.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const e = Object.assign(document.createElement(tag), props);
  kids.forEach(k => e.append(k));
  return e;
};

let cfg = null, songFile = null;
const ears = { l: [], r: [] };          // arrays of the 8 <input> elements
const rowEls = new Map();               // id -> {tr, status}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = "/login"; throw new Error("login"); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `Error ${r.status}`);
  return body;
}

// ---------- upload area ----------
function buildEar(side) {
  const box = $(`#hl${side.toUpperCase()}`);
  cfg.frequencies.forEach(f => {
    const input = el("input", { type: "number", step: "1", min: "-10", max: "120", placeholder: "0" });
    ears[side].push(input);
    box.append(el("div", {}, el("label", { textContent: f >= 1000 ? `${f / 1000}k` : `${f}` }), input));
  });
}

function parseAudiogramText(text) {
  try {                                    // JSON: an 8-number array anywhere inside
    const find = v => {
      if (Array.isArray(v) && v.length === 8 && v.every(x => typeof x === "number")) return v;
      if (v && typeof v === "object") for (const x of Object.values(v)) { const f = find(x); if (f) return f; }
      return null;
    };
    const f = find(JSON.parse(text)); if (f) return f;
  } catch (_) { /* not JSON */ }
  const nums = (text.match(/-?\d+(\.\d+)?/g) || []).map(Number);
  return nums.length >= 8 ? nums.slice(0, 8) : null;
}

function wireDrop(box, onFile) {
  const input = $("input[type=file]", box);
  const open = () => input.click();
  $(".zone", box).addEventListener("click", open);
  input.addEventListener("change", () => input.files[0] && onFile(input.files[0]));
  ["dragenter", "dragover"].forEach(t => box.addEventListener(t, e => { e.preventDefault(); box.classList.add("over"); }));
  ["dragleave", "drop"].forEach(t => box.addEventListener(t, e => { e.preventDefault(); box.classList.remove("over"); }));
  box.addEventListener("drop", e => e.dataTransfer.files[0] && onFile(e.dataTransfer.files[0]));
}

function setupDrops() {
  wireDrop($('[data-kind="song"]'), f => { songFile = f; $("#songName").textContent = f.name; });
  document.querySelectorAll('[data-kind="ear"]').forEach(box => {
    const side = box.dataset.ear;
    wireDrop(box, async f => {
      const vals = parseAudiogramText(await f.text());
      const label = $(side === "l" ? "#fileL" : "#fileR");
      if (!vals) { label.textContent = `${f.name}: could not find 8 numbers`; return; }
      vals.forEach((v, i) => ears[side][i].value = v);
      label.textContent = f.name;
    });
  });
}

function buildChoices() {
  const inst = $("#instruments"), gains = $("#gains");
  inst.append(el("label", { textContent: "Instruments (2 or 3)" }));
  gains.append(el("label", { textContent: "Gain dB (blank = automatic from audiogram)" }));
  const g = el("div"); gains.append(g);
  for (let i = 0; i < 3; i++) {
    const sel = el("select", { className: "inst" }, el("option", { value: "", textContent: "none" }));
    cfg.instruments.forEach(n => sel.append(el("option", { value: n, textContent: n })));
    inst.append(sel, " ");
    g.append(el("input", { className: "gain", type: "number", step: "0.5", min: "-20", max: "20", placeholder: "auto" }), " ");
  }
}

function say(text, isError = true) {
  const m = $("#msg");
  m.textContent = text;
  m.classList.toggle("err", Boolean(text) && isError);
  if (text && isError) m.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

async function go() {
  const names = [...document.querySelectorAll(".inst")].map(s => s.value);
  const chosen = names.filter(Boolean);
  const gainVals = [...document.querySelectorAll("#gains .gain")].filter((_, i) => names[i]).map(g => g.value);
  if (!songFile) return say("Add a song first.");
  if (chosen.length < 2) return say("Choose at least 2 instruments.");
  const vals = side => ears[side].map(i => i.value);
  if ([...vals("l"), ...vals("r")].some(v => v === "")) return say("Fill in all 8 values for both ears.");
  const anyGain = gainVals.some(v => v !== ""), allGain = gainVals.every(v => v !== "");
  if (anyGain && !allGain) return say("Either set every gain or leave them all blank.");

  const fd = new FormData();
  fd.append("song", songFile);
  fd.append("audiogram_l", vals("l").join(","));
  fd.append("audiogram_r", vals("r").join(","));
  fd.append("instruments", chosen.join(","));
  fd.append("gains", anyGain ? gainVals.join(",") : "");
  fd.append("model", $("#model").value);
  fd.append("residual", $("#residual").value || "0.1");
  $("#go").disabled = true; say("Uploading...", false);
  try {
    await api("/api/process", { method: "POST", body: fd });
    say("Processing. This can take a few minutes; the row below updates by itself.", false);
    await refresh();
  } catch (e) { say(e.message); }
  finally { $("#go").disabled = !modelsReady; }
}

// ---------- models ----------
let modelList = [];
const currentModel = () => $("#model").value || "baseline";
const store = {
  get: k => { try { return localStorage.getItem(k); } catch (_) { return null; } },
  set: (k, v) => { try { localStorage.setItem(k, v); } catch (_) { /* private mode */ } },
};

function showModelInfo() {
  const m = modelList.find(x => x.id === $("#model").value);
  if (!m) return;
  const rest = cfg.instruments.filter(i => !m.instruments.includes(i));
  $("#modelInfo").textContent = m.builtin
    ? "Built-in Cadenza pretrained models for all instruments."
    : `Covers: ${m.instruments.join(", ")}` + (rest.length ? ` (${rest.join(", ")} use the baseline)` : "") +
      (m.size_mb ? ` \u00b7 ${m.size_mb} MB` : "");
  $("#delModel").hidden = m.builtin;
  store.set("model", m.id);
  $("#historyTitle").textContent = `History \u00b7 ${m.name}`;
  showHistoryFor(m.id);
}

let shownModel = null;
function showHistoryFor(id) {          // each model has its own table of songs
  if (id === shownModel) return;
  shownModel = id;
  $("#rows").replaceChildren(); rowEls.clear(); $("#empty").hidden = true;
  if (typeof refresh === "function" && cfg) refresh().then(busy => busy && kickPoll()).catch(() => {});
}

async function loadModels(select) {
  modelList = await api("/api/models");
  const sel = $("#model");
  sel.replaceChildren(...modelList.map(m => el("option", { value: m.id, textContent: m.name })));
  const want = select || store.get("model");
  sel.value = modelList.some(m => m.id === want) ? want : "baseline";
  showModelInfo();
}

function uploadModel(file) {
  const bar = $("#modelProg"), msg = $("#modelMsg"), btn = $("#uploadModel");
  const fd = new FormData();
  fd.append("file", file); fd.append("name", $("#modelName").value);
  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/models");
  btn.disabled = true; bar.hidden = false; bar.value = 0; msg.textContent = "Uploading...";
  xhr.upload.onprogress = e => {
    if (!e.lengthComputable) return;
    bar.value = (100 * e.loaded) / e.total;
    msg.textContent = e.loaded < e.total
      ? `Uploading ${(e.loaded / 1e6).toFixed(0)} of ${(e.total / 1e6).toFixed(0)} MB...`
      : "Upload done. Checking the model (this can take a minute)...";
  };
  xhr.onload = async () => {
    btn.disabled = false; bar.hidden = true;
    if (xhr.status === 401) { location.href = "/login"; return; }
    let body = {}; try { body = JSON.parse(xhr.responseText); } catch (_) { /* not JSON */ }
    if (xhr.status >= 200 && xhr.status < 300) {
      msg.textContent = `Added "${body.name}". It is now on the server for everyone.`;
      $("#modelName").value = "";
      await loadModels(body.id);
    } else msg.textContent = body.detail || `Upload failed (${xhr.status})`;
  };
  xhr.onerror = () => { btn.disabled = false; bar.hidden = true; msg.textContent = "Upload failed (network error)."; };
  xhr.send(fd);
}

function setupModels() {
  $("#model").addEventListener("change", showModelInfo);
  $("#uploadModel").addEventListener("click", () => $("#modelFile").click());
  $("#modelFile").addEventListener("change", e => {
    const f = e.target.files[0]; e.target.value = "";
    if (f) uploadModel(f);
  });
  $("#delModel").addEventListener("click", async () => {
    const m = modelList.find(x => x.id === $("#model").value);
    if (!m || m.builtin) return;
    const n = m.songs || 0;
    if (!confirm(`Delete the model "${m.name}" from the server for everyone?` +
        (n ? `\n\nThis also deletes its ${n} song${n > 1 ? "s" : ""} (before, after, ratings and comments).` : ""))) return;
    try { await api(`/api/models/${m.id}`, { method: "DELETE" }); $("#modelMsg").textContent = "Model deleted."; await loadModels(); }
    catch (e) { $("#modelMsg").textContent = e.message; }
  });
  $("#residual").value = cfg.default_residual;
  return loadModels();
}

// ---------- history ----------
function stars(song) {
  const wrap = el("div", { className: "stars" });
  const paint = n => [...wrap.children].forEach((b, i) => b.textContent = i < n ? "★" : "☆");
  for (let n = 1; n <= 5; n++) {
    const b = el("button", { type: "button", title: `${n} star${n > 1 ? "s" : ""}` });
    b.addEventListener("click", async () => {
      const next = (song.rating === n) ? 0 : n;
      song.rating = next; paint(next);
      try { await api(`/api/songs/${song.id}`, jsonOpts("PATCH", { rating: next })); }
      catch (e) { say(e.message); }
    });
    wrap.append(b);
  }
  paint(song.rating);
  return wrap;
}

const jsonOpts = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function comment(song) {
  const t = el("textarea", { value: song.comment, maxLength: 2000, placeholder: "Comment..." });
  let timer = null;
  const save = () => { clearTimeout(timer); return api(`/api/songs/${song.id}`, jsonOpts("PATCH", { comment: t.value })).catch(e => say(e.message)); };
  t.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(save, 600); });
  t.addEventListener("blur", save);
  return t;
}

function audioCell(url, caption) {
  const cell = el("td");
  if (url) cell.append(el("audio", { controls: true, preload: "none", src: url }));
  if (caption) cell.append(el("div", { className: "cap", textContent: caption }));
  return cell;
}

const FREQ_LABEL = ["250", "500", "1k", "2k", "3k", "4k", "6k", "8k"];
function settingsCells(song) {
  const num = n => (Math.round(n * 10) / 10).toString();
  const sign = g => (g > 0 ? "+" : "") + num(g);
  const insts = el("td", { className: "settings" });
  song.instruments.forEach((name, i) => insts.append(el("div", { textContent: `${name} (${sign(song.gains[i])})` })));
  const ear = levels => el("td", { className: "settings aud", title: levels.map((v, i) => `${FREQ_LABEL[i]} Hz: ${v} dB`).join("\n"),
                                   textContent: levels.map(num).join(" ") });
  return [el("td", { className: "settings", textContent: num(song.residual) }), insts, ear(song.audiogram_l), ear(song.audiogram_r)];
}

function buildRow(song) {
  const tr = el("tr");
  tr.append(audioCell(song.before_url, song.name));
  if (song.status === "done") tr.append(audioCell(song.after_url, ""));
  else tr.append(el("td", { className: "status", textContent: song.status === "error" ? `Failed: ${song.error}` : "Processing..." }));
  const commentCell = el("td", {}, comment(song));
  tr.append(el("td", {}, stars(song)), commentCell, ...settingsCells(song));
  const del = el("button", { type: "button", className: "link", textContent: "Delete" });
  del.addEventListener("click", async () => {
    if (!confirm("Delete this song and its processed version?")) return;
    try { await api(`/api/songs/${song.id}`, { method: "DELETE" }); await refresh(); } catch (e) { say(e.message); }
  });
  commentCell.append(el("div", { className: "cap" }, del));
  return tr;
}

async function refresh() {
  const wanted = currentModel();
  const songs = await api(`/api/songs?model=${encodeURIComponent(wanted)}`);
  if (wanted !== currentModel()) return false;       // the user switched model meanwhile
  const body = $("#rows");
  const seen = new Set();
  let prev = null;
  songs.forEach(song => {
    seen.add(song.id);
    const known = rowEls.get(song.id);
    if (!known || known.status !== song.status) {         // new row, or its status changed
      const tr = buildRow(song);
      if (known) known.tr.replaceWith(tr); else body.insertBefore(tr, prev ? prev.nextSibling : body.firstChild);
      rowEls.set(song.id, { tr, status: song.status });
    }
    prev = rowEls.get(song.id).tr;
  });
  for (const [id, r] of rowEls) if (!seen.has(id)) { r.tr.remove(); rowEls.delete(id); }
  $("#empty").hidden = songs.length > 0;
  return songs.some(s => s.status === "processing");
}

let modelsReady = false;
async function checkModels() {
  try {
    const st = await api("/api/status");
    const b = $("#banner");
    modelsReady = st.state === "ready";
    b.hidden = modelsReady;
    b.className = st.state === "error" ? "error" : "";
    b.textContent = st.message;
    $("#go").disabled = !modelsReady;
    if (!modelsReady && st.state !== "error") setTimeout(checkModels, 3000);
  } catch (_) { setTimeout(checkModels, 5000); }
}

let pollTimer = null;
async function poll() {
  clearTimeout(pollTimer);
  let busy = false;
  try { busy = await refresh(); } catch (_) { /* retry next tick */ }
  pollTimer = setTimeout(poll, busy ? 2000 : 8000);
}
const kickPoll = () => { clearTimeout(pollTimer); pollTimer = setTimeout(poll, 2000); };

(async function init() {
  cfg = await api("/api/config");
  buildEar("l"); buildEar("r"); buildChoices(); setupDrops(); await setupModels();
  $("#go").addEventListener("click", go);
  $("#go").disabled = true;   // enabled by checkModels() once the models are on disk
  checkModels();
  poll();
})();
