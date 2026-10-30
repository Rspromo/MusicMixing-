const $ = id => document.getElementById(id);
const state = {stems: [], roles: [], lines: [], terms: [], localMedia: [], clips: [], style: "performance",
               vocals: [], filter: "all", handled: new Set(), openFolders: {}};

async function api(path, data) {
  const opts = {headers: {"X-Token": window.TOKEN, "Content-Type": "application/json"}};
  if (data !== undefined) { opts.method = "POST"; opts.body = JSON.stringify(data); }
  const res = await fetch("/api/" + path, opts);
  const out = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(out.error || "Kuch galat ho gaya");
  return out;
}
function toast(text) {
  const box = $("toast"); box.textContent = text; box.hidden = false;
  clearTimeout(box.timer); box.timer = setTimeout(() => box.hidden = true, 4500);
}
const guard = fn => async (...args) => { try { return await fn(...args); } catch (e) { toast(e.message); } };
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const mediaUrl = p => `/api/media?token=${encodeURIComponent(window.TOKEN)}&path=${encodeURIComponent(p)}`;
const clock = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const openPath = guard(path => api("open", {path}));
const pick = (mode, types) => api("pick", {mode, types});

document.querySelectorAll(".tab").forEach(tab => tab.onclick = () => {
  document.querySelectorAll(".tab,.page").forEach(el => el.classList.remove("active"));
  tab.classList.add("active"); $(tab.dataset.tab).classList.add("active");
});
document.querySelectorAll("[data-pick]").forEach(btn => btn.onclick = guard(async e => {
  e.preventDefault();
  const got = await pick(btn.dataset.pick, btn.dataset.types);
  if (got.length) { const box = $(btn.dataset.into); box.value = got[0]; box.dispatchEvent(new Event("change")); }
}));
function bindRange(id, format) {
  const out = $(id + "Out");
  const show = () => out.textContent = format($(id).value);
  $(id).oninput = show; show();
}

/* Progress queue */
const jobDone = {};
async function pollJobs() {
  try {
    const list = await api("jobs");
    $("jobs").innerHTML = list.length ? list.map(j => `
      <div class="job ${j.status}">
        <div class="title">${esc(j.title)}</div>
        <div class="bar"><i style="width:${(j.progress * 100).toFixed(1)}%"></i></div>
        <div class="stage">${esc(j.stage)}</div>
        ${j.error ? `<div class="err">${esc(j.error)}</div>` : ""}
        <div class="foot"><span>${Math.round(j.progress * 100)}% · ${clock(j.elapsed)}</span>
        ${["queued", "running"].includes(j.status) ? `<button data-cancel="${j.id}">Cancel</button>` : ""}
        ${j.result?.folder ? `<button data-folder="${esc(j.result.folder)}">Folder</button>` : ""}</div>
      </div>`).join("") : '<p class="muted">Abhi koi kaam nahi chal raha.</p>';
    for (const j of list) {
      if (j.status === "done" && !state.handled.has(j.id)) {
        state.handled.add(j.id);
        jobDone[j.id]?.(j.result);
      }
    }
  } catch { /* the next poll retries */ }
  setTimeout(pollJobs, 700);
}
$("jobs").onclick = e => {
  if (e.target.dataset.cancel) api(`jobs/${e.target.dataset.cancel}/cancel`, {});
  if (e.target.dataset.folder) openPath(e.target.dataset.folder);
};
const watch = (reply, done) => { jobDone[reply.job] = done; };

/* Mixing */
function drawStems() {
  $("stemCount").textContent = `(${state.stems.length})`;
  $("stemRows").innerHTML = state.stems.map((s, i) => `<tr><td>${esc(s.name)} ${s.guessed ? '<span class="tag">Guess</span>' : ""}</td>
    <td><select data-stem="${i}">${state.roles.map(r => `<option value="${r}" ${r === s.role ? "selected" : ""}>${r.replace(/_/g, " ")}</option>`).join("")}</select></td>
    <td><div class="row"><button data-remove="${i}">Hatao</button></div></td></tr>`).join("");
}
$("stemRows").onchange = e => { state.stems[e.target.dataset.stem].role = e.target.value; };
$("stemRows").onclick = e => { if (e.target.dataset.remove) { state.stems.splice(+e.target.dataset.remove, 1); drawStems(); } };
$("mixScan").onclick = guard(async () => {
  const out = await api("mix/scan", {folder: $("mixFolder").value});
  if (!out.stems.length) throw new Error("Is folder mein audio files nahi mili.");
  state.stems = out.stems; state.roles = out.roles;
  if (!$("mixName").value) $("mixName").value = $("mixFolder").value.split(/[\\/]/).filter(Boolean).pop();
  $("stemPanel").hidden = false; drawStems();
});
$("mixRun").onclick = guard(async () => {
  const reply = await api("mix/run", {stems: state.stems, name: $("mixName").value, genre: $("mixGenre").value,
    lufs: parseFloat($("mixLufs").value) || null, vocal_boost: $("mixVocal").value, reverb: $("mixReverb").value,
    reference: $("mixRef").value, variants: $("mixVariants").checked, export_stems: $("mixStems").checked});
  watch(reply, r => {
    const box = $("mixResult"); box.hidden = false;
    box.innerHTML = `<h2>Mix taiyar hai</h2><div class="stats"><div><b>${r.lufs}</b><span>LUFS</span></div>
      <div><b>${r.true_peak}</b><span>True Peak dB</span></div><div><b>${r.bpm}</b><span>BPM</span></div><div><b>${r.tracks}</b><span>Tracks</span></div></div>
      ${r.warning ? `<p class="warn">${esc(r.warning)}</p>` : ""}
      <audio controls src="${mediaUrl(r.master)}"></audio><button data-open="${esc(r.folder)}">Folder kholo</button>`;
  });
});
$("mixResult").onclick = e => { if (e.target.dataset.open) openPath(e.target.dataset.open); };

/* Downloader */
async function loadLibrary() {
  const items = await api("library");
  const audio = ["mp3", "wav", "flac", "m4a", "aif", "aiff", "ogg"];
  const shown = items.filter(i => state.filter === "all" || (state.filter === "audio") === audio.includes(i.kind));
  $("libRows").innerHTML = shown.length ? shown.map(i => `<tr>
    <td>${esc(i.title)}${i.exists ? "" : ' <span class="tag">File hat gayi</span>'}</td><td>${esc(i.kind).toUpperCase()}</td>
    <td class="num">${i.bpm || "-"}</td><td>${esc(i.key) || "-"}</td><td class="num">${clock(i.duration)}</td>
    <td><div class="row"><button data-show="${esc(i.path)}">Folder</button><button data-forget="${esc(i.path)}">Hatao</button></div></td></tr>`).join("")
    : '<tr><td colspan="6" class="muted">Abhi kuch nahi hai.</td></tr>';
}
$("libRows").onclick = guard(async e => {
  if (e.target.dataset.show) openPath(e.target.dataset.show);
  if (e.target.dataset.forget) { await api("library/remove", {path: e.target.dataset.forget}); loadLibrary(); }
});
document.querySelectorAll(".chip").forEach(chip => chip.onclick = () => {
  document.querySelectorAll(".chip").forEach(c => c.classList.remove("on"));
  chip.classList.add("on"); state.filter = chip.dataset.filter; loadLibrary();
});
$("dlStart").onclick = guard(async () => {
  const reply = await api("download", {url: $("dlUrl").value, kind: $("dlKind").value, quality: $("dlQuality").value});
  $("dlUrl").value = ""; watch(reply, loadLibrary);
});
$("dlPickFolder").onclick = guard(async () => {
  const got = await pick("folder");
  if (got.length) { await api("settings", {download_dir: got[0]}); $("dlFolder").value = got[0]; }
});
$("dlFolder").onchange = guard(() => api("settings", {download_dir: $("dlFolder").value}));
$("dlOpenFolder").onclick = () => openPath($("dlFolder").value);
$("dlAddLocal").onclick = guard(async () => {
  const got = await pick("files", "audio");
  if (got.length) watch(await api("library/add", {paths: got}), loadLibrary);
});

/* Lyrics */
function drawLines() {
  $("lyEditor").hidden = false;
  $("lyCount").textContent = `(${state.lines.length} lines)`;
  $("lyLines").innerHTML = state.lines.map((l, i) => `<div class="line" data-i="${i}">
    <input data-f="start" type="number" step="0.05" value="${l.start.toFixed(2)}"><input data-f="end" type="number" step="0.05" value="${l.end.toFixed(2)}">
    <input data-f="text" value="${esc(l.text)}"><button data-del="${i}">×</button></div>`).join("");
}
$("lyLines").onchange = e => {
  const line = state.lines[e.target.closest(".line").dataset.i], field = e.target.dataset.f;
  if (field === "text") {
    if (e.target.value !== line.text) {
      const words = e.target.value.split(/\s+/).filter(Boolean), span = (line.end - line.start) / Math.max(words.length, 1);
      line.words = words.map((word, k) => ({word, start: line.start + k * span, end: line.start + (k + 1) * span}));
    }
    line.text = e.target.value;
  } else {
    const before = line[field]; line[field] = parseFloat(e.target.value) || 0;
    if (field === "start") { const d = line.start - before; (line.words || []).forEach(w => { w.start += d; w.end += d; }); }
  }
};
$("lyLines").onclick = e => { if (e.target.dataset.del) { state.lines.splice(+e.target.dataset.del, 1); drawLines(); } };
$("lyAddLine").onclick = () => {
  const last = state.lines.at(-1)?.end ?? 0;
  state.lines.push({start: last, end: last + 3, text: "", words: []}); drawLines();
};
$("lyExtract").onclick = guard(async () => {
  const reply = await api("lyrics/extract", {audio: $("lyAudio").value, vocal: $("lyVocal").value, language: $("lyLang").value,
    engine: $("lyEngine").value, model: $("lyModel").value, polish: $("lyPolish").checked, roman: $("lyRoman").checked});
  watch(reply, r => {
    state.lines = r.lines; drawLines();
    if (r.terms.length && !$("lvTerms").value) $("lvTerms").value = r.terms.join(", ");
    if (!$("lvTitle").value) $("lvTitle").value = $("lyAudio").value.split(/[\\/]/).pop().replace(/\.[^.]+$/, "");
  });
});
$("lvPickLocal").onclick = guard(async () => {
  state.localMedia = await pick("files", "media");
  $("lvLocal").value = state.localMedia.length ? `${state.localMedia.length} files` : "";
});
function lyricOptions(preview) {
  return {source: $("lvSource").value, clip_length: +$("lvClip").value, dim: +$("lvDim").value, blur: +$("lvBlur").value,
    style: $("lvStyle").value, font: $("lvFont").value, font_size: +$("lvSize").value, position: $("lvPosition").value,
    text_color: $("lvColor").value, highlight_color: $("lvHigh").value, outline: +$("lvOutline").value, shadow: +$("lvShadow").value,
    offset: parseFloat($("lvOffset").value) || 0, aspect: $("lvAspect").value, height: +$("lvHeight").value, fps: +$("lvFps").value,
    title: $("lvTitle").value, artist: $("lvArtist").value, logo: $("lvLogo").value, kenburns: $("lvKen").checked,
    visualizer: $("lvViz").checked, uppercase: $("lvUpper").checked, bold: $("lvBold").checked, preview};
}
const startLyricVideo = preview => guard(async () => {
  if (!state.lines.length) throw new Error("Pehle lyrics nikalein.");
  await api("lyrics/render", {audio: $("lyAudio").value, lines: state.lines, local: state.localMedia, options: lyricOptions(preview),
    terms: $("lvTerms").value.split(",").map(t => t.trim()).filter(Boolean)});
  toast("Render shuru. Progress right side mein dikhega.");
});
$("lvPreview").onclick = startLyricVideo(30);
$("lvRender").onclick = startLyricVideo(0);

/* Video editor */
async function loadStyles() {
  const styles = await api("video/styles");
  $("veStyles").innerHTML = Object.entries(styles).map(([k, v]) => `<button class="style ${k === state.style ? "on" : ""}" data-style="${k}"><b>${esc(v.label)}</b><span>${esc(v.about)}</span></button>`).join("");
}
$("veStyles").onclick = e => {
  const btn = e.target.closest("[data-style]");
  if (!btn) return;
  state.style = btn.dataset.style;
  document.querySelectorAll(".style").forEach(s => s.classList.toggle("on", s === btn));
};
$("veFolder").onchange = guard(async () => {
  state.clips = await api("video/scan", {folder: $("veFolder").value});
  $("veCount").textContent = `${state.clips.length} clips mile`;
});
$("veRun").onclick = guard(async () => {
  if (!state.clips.length) throw new Error("Clips ka folder chunein.");
  const reply = await api("video/edit", {song: $("veSong").value, clips: state.clips, style: state.style,
    fps: $("veFps").value, seed: $("veSeed").value, length: $("veLength").value});
  watch(reply, r => {
    const box = $("veResult"); box.hidden = false;
    box.innerHTML = `<h2>Edit taiyar hai</h2><div class="stats"><div><b>${r.cuts}</b><span>Cuts</span></div><div><b>${r.bpm}</b><span>BPM</span></div>
      <div><b>${r.clips_used}/${r.clips_total}</b><span>Clips use hue</span></div><div><b>${r.synced}</b><span>Lip-sync match</span></div></div>
      <video controls src="${mediaUrl(r.file)}"></video>
      <div class="row"><button data-open="${esc(r.premiere)}">Premiere files</button><button data-open="${esc(r.folder)}">Folder kholo</button></div>`;
  });
});
$("veResult").onclick = e => { if (e.target.dataset.open) openPath(e.target.dataset.open); };

/* Tuning */
function drawChart(chart) {
  const c = $("tnChart"), g = c.getContext("2d"), rows = chart.curve;
  g.clearRect(0, 0, c.width, c.height);
  if (!rows.length) return;
  const view = rows.slice(0, 2400), t0 = view[0][0], t1 = view.at(-1)[0] || t0 + 1;
  const low = Math.floor(Math.min(...view.map(r => r[1]))) - 1, high = Math.ceil(Math.max(...view.map(r => r[1]))) + 1;
  const x = t => (t - t0) / (t1 - t0 || 1) * c.width, y = m => c.height - (m - low) / (high - low) * c.height;
  g.font = "12px Segoe UI"; g.fillStyle = "#989ca3"; g.strokeStyle = "#363b42";
  const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
  for (let m = low; m <= high; m++) if (chart.allowed.includes(((m % 12) + 12) % 12)) {
    g.beginPath(); g.moveTo(0, y(m)); g.lineTo(c.width, y(m)); g.stroke(); g.fillText(names[((m % 12) + 12) % 12], 4, y(m) - 2);
  }
  [[1, "#e5736b"], [2, "#6cc08b"]].forEach(([k, color]) => {
    g.fillStyle = color;
    view.forEach(r => g.fillRect(x(r[0]), y(r[k]), 2, 2));
  });
}
$("tnPick").onclick = guard(async () => {
  state.vocals = await pick("files", "audio");
  $("tnVocals").value = state.vocals.length ? `${state.vocals.length} files` : "";
});
$("tnPreset").onchange = () => {
  $("tnStrength").value = {natural: .8, tight: .95, hard: 1}[$("tnPreset").value];
  $("tnStrength").oninput();
};
$("tnBounce").onclick = guard(async () => {
  const reply = await api("ableton/bounce", {path: $("tnAls").value});
  watch(reply, r => {
    state.vocals = r.files.filter(f => /vox|vocal|voice|lead|harm/i.test(f));
    $("tnVocals").value = `${state.vocals.length} vocal tracks`;
    toast(r.warnings.length ? r.warnings.slice(0, 2).join(" | ") : "Tracks nikal gaye");
    openPath(r.folder);
  });
});
$("tnRun").onclick = guard(async () => {
  if (!state.vocals.length) throw new Error("Vocal files chunein.");
  const reply = await api("tune/run", {vocals: state.vocals, music: $("tnMusic").value, key: $("tnKey").value, scale: $("tnScale").value,
    preset: $("tnPreset").value, strength: +$("tnStrength").value, max_shift: +$("tnShift").value});
  watch(reply, r => {
    $("tnResult").hidden = false;
    $("tnSummary").innerHTML = `<div class="stats"><div><b>${esc(r.key)}</b><span>Key</span></div><div><b>${r.bpm}</b><span>BPM</span></div>
      <div><b>${r.files[0].notes}</b><span>Notes</span></div><div><b>${r.files[0].avg_error_before}¢ → ${r.files[0].avg_error_after}¢</b><span>Average error</span></div></div>
      ${r.files.map(f => `<audio controls src="${mediaUrl(f.file)}"></audio>`).join("")}
      <button data-open="${esc(r.folder)}">Folder kholo</button>`;
    drawChart(r.chart);
  });
});
$("tnSummary").onclick = e => { if (e.target.dataset.open) openPath(e.target.dataset.open); };

/* Settings and start-up */
$("setSave").onclick = guard(async () => {
  await api("settings", {output_dir: $("setOutput").value, openai_key: $("setOpenai").value, pexels_key: $("setPexels").value,
    openai_chat_model: $("setChat").value, encoder: $("setEncoder").value});
  $("setStatus").textContent = "Save ho gaya.";
});
guard(async () => {
  const s = await api("settings");
  $("setOutput").value = s.output_dir; $("setOpenai").value = s.openai_key; $("setPexels").value = s.pexels_key;
  $("setChat").value = s.openai_chat_model; $("setEncoder").value = s.encoder; $("dlFolder").value = s.download_dir;
  $("lyModel").value = s.whisper_model;
  const mix = await api("mix/genres");
  $("mixGenre").innerHTML = Object.entries(mix.genres).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  const tune = await api("tune/options");
  $("tnKey").innerHTML += tune.notes.map(n => `<option>${n}</option>`).join("");
  $("tnScale").innerHTML += tune.scales.map(n => `<option>${esc(n)}</option>`).join("");
  await Promise.all([loadStyles(), loadLibrary()]);
  const st = await api("status");
  $("sysinfo").textContent = `Encoder: ${st.encoder} · ${st.cpu} core`;
})();
bindRange("mixVocal", v => `${v > 0 ? "+" : ""}${v} dB`);
bindRange("mixReverb", v => `${Math.round(v * 100)}%`);
bindRange("lvClip", v => `${v} sec`);
bindRange("lvDim", v => `${Math.round(v * 100)}%`);
bindRange("lvBlur", v => v);
bindRange("lvSize", v => v);
bindRange("lvOutline", v => v);
bindRange("lvShadow", v => v);
bindRange("tnStrength", v => `${Math.round(v * 100)}%`);
pollJobs();
