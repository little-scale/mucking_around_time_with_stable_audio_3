"use strict";

const TRACK_IDS = ["melodic", "percussion", "texture"];
const MUSIC_TRACKS = new Set(["melodic", "percussion"]);
const WEIGHTS = ["subtle", "normal", "strong"];
const CROSSFADE = 0.065;
const LOOKAHEAD = 0.24;
const API_BASE = "/api/drift";
const $ = (selector, root = document) => root.querySelector(selector);

const globalElements = {
  statusPill: $("#statusPill"), statusLabel: $("#statusLabel"), playButton: $("#playButton"),
  playIcon: $("#playIcon"), transportState: $("#transportState"), beatReadout: $("#beatReadout"),
  bpm: $("#bpmInput"), master: $("#masterVolume"), freezeAll: $("#freezeAllButton"),
  globalMutation: $("#globalMutation"), globalMutationValue: $("#globalMutationValue"),
  masterMeterValue: $("#masterMeterValue"), masterMeterLeft: $("#masterMeterLeft"),
  masterMeterRight: $("#masterMeterRight"),
  evolve: $("#evolveButton"), queueStrip: $(".queue-strip"), queueTitle: $("#queueTitle"),
  queueDetail: $("#queueDetail"), tempoNote: $("#tempoNote"), toast: $("#toast"),
  refreshHistory: $("#refreshHistory"), sidechainEnabled: $("#sidechainEnabled"),
  sidechainMelodic: $("#sidechainMelodic"), sidechainTexture: $("#sidechainTexture"),
  scThreshold: $("#scThreshold"), scRatio: $("#scRatio"), scAttack: $("#scAttack"),
  scRelease: $("#scRelease"), scDepth: $("#scDepth"), scMakeup: $("#scMakeup"),
  reductionBar: $("#reductionBar"), reductionLabel: $("#reductionLabel"),
  reverbEnabled: $("#reverbEnabled"), rvReturn: $("#rvReturn"), rvDecay: $("#rvDecay"),
  rvPredelay: $("#rvPredelay"), rvDamping: $("#rvDamping"),
  delayEnabled: $("#delayEnabled"), delayDivision: $("#delayDivision"),
  delayReturn: $("#delayReturn"), delayFeedback: $("#delayFeedback"),
  delayLowpass: $("#delayLowpass"), delayWidth: $("#delayWidth"),
  delayTimeReadout: $("#delayTimeReadout"),
  flushAllHistory: $("#flushAllHistory"), clearAllOutputs: $("#clearAllOutputs"),
  backendBadge: $("#backendBadge"),
};

const tracks = Object.fromEntries(TRACK_IDS.map((id) => {
  const root = document.querySelector(`[data-track="${id}"]`);
  return [id, {
    id, root, tags: [], status: null, formRevision: null, dirty: false, syncTimer: null,
    active: null, pending: null, requestedGeneration: null, source: null, sourceGain: null,
    loopAnchor: 0, transition: null, nodes: null,
    cloud: $(".prompt-cloud", root), tagInput: $(".tag-input", root), addTag: $(".add-tag", root),
    sync: $(".sync-state", root), constructed: $(".constructed-prompt", root),
    freeze: $(".freeze-button", root), phase: $(".phase", root), generation: $(".generation", root),
    modelBadge: $(".track-head p", root), modelSelect: $(".model", root),
    duration: $(".duration-readout", root), referenceDrop: $(".reference-drop", root),
    referenceInput: $(".reference-input", root), mute: $(".mute-button", root), solo: $(".solo-button", root),
    volume: $(".volume", root), highpass: $(".highpass", root), lowpass: $(".lowpass", root),
    reverbSend: $(".reverb-send", root), delaySend: $(".delay-send", root),
    beats: $(".beats", root), mutation: $(".mutation", root),
    cfg: $(".cfg", root), steps: $(".steps", root), mutationInterval: $(".mutation-interval", root),
    seedMode: $(".seed-mode", root), seed: $(".seed", root), seedWrap: $(".seed-wrap", root),
    rolePrompt: $(".role-prompt", root), negativePrompt: $(".negative-prompt", root),
    flushHistory: $(".flush-track", root), clearOutput: $(".clear-output", root),
    error: $(".track-error", root), history: document.querySelector(`[data-history="${id}"]`),
  }];
}));

let statusState = null;
let sessionState = null;
let pollBusy = false;
let toastTimer = null;
let sessionSaveTimer = null;
let globalMutationTimer = null;
let uiTickHandle = null;
let backendLabel = "accelerator";
const audioCache = new Map();

const audio = {
  context: null, master: null, limiter: null, reverbInput: null, predelay: null,
  convolver: null, reverbFilter: null, reverbReturn: null, playing: false,
  delayInput: null, delayLeft: null, delayRight: null, delayFilterLeft: null,
  delayFilterRight: null, delayFeedbackLeft: null, delayFeedbackRight: null,
  delayPanLeft: null, delayPanRight: null, delayReturn: null,
  meterSplitter: null, meterLeft: null, meterRight: null, meterFrame: null,
  meterLeftData: null, meterRightData: null, meterLevels: [0, 0],
  anchorTime: 0, pausedBeats: 0, bpm: null, scheduler: null, sidechainTimer: null,
  impulseKey: "", currentReduction: 0,
};

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: options.body && typeof options.body === "string"
      ? { "Content-Type": "application/json", ...(options.headers || {}) }
      : options.headers,
  });
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail : payload;
    throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail || `HTTP ${response.status}`);
  }
  return payload;
}

function toast(message, error = false) {
  clearTimeout(toastTimer);
  globalElements.toast.textContent = message;
  globalElements.toast.classList.toggle("error", error);
  globalElements.toast.classList.add("show");
  toastTimer = setTimeout(() => globalElements.toast.classList.remove("show"), 3200);
}

function cleanText(value) {
  return String(value || "").replace(/\s+/g, " ").trim().slice(0, 120);
}

function weightedPrompt(tags) {
  return tags.map((tag) => {
    if (tag.weight === "subtle") return `subtle hint of ${tag.text}`;
    if (tag.weight === "strong") return `prominent ${tag.text}, ${tag.text}`;
    return tag.text;
  }).join(", ");
}

function modelPrompt(track) {
  const parts = [weightedPrompt(track.tags)];
  if (MUSIC_TRACKS.has(track.id)) parts.push(`${number(globalElements.bpm.value, 120)} BPM`);
  parts.push(track.rolePrompt.value.trim());
  return parts.filter(Boolean).join(", ");
}

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text != null) element.textContent = text;
  return element;
}

function renderTags(track) {
  track.cloud.replaceChildren();
  track.tags.forEach((tag, index) => {
    const chip = node("span", "prompt-tag");
    chip.dataset.weight = tag.weight;
    chip.setAttribute("role", "listitem");
    const text = node("span", "prompt-tag-text", tag.text);
    const increase = node("button", "tag-weight", "+");
    increase.type = "button";
    increase.disabled = tag.weight === "strong";
    increase.setAttribute("aria-label", `Increase influence of ${tag.text}`);
    increase.addEventListener("click", () => changeTagWeight(track, index, 1));
    const decrease = node("button", "tag-weight", "−");
    decrease.type = "button";
    decrease.disabled = tag.weight === "subtle";
    decrease.setAttribute("aria-label", `Reduce influence of ${tag.text}`);
    decrease.addEventListener("click", () => changeTagWeight(track, index, -1));
    const remove = node("button", "tag-remove", "×");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove ${tag.text}`);
    remove.addEventListener("click", () => removeTag(track, index));
    chip.append(text, increase, decrease, remove);
    track.cloud.append(chip);
  });
  track.constructed.textContent = modelPrompt(track) || "—";
}

function addTag(track) {
  const candidates = track.tagInput.value.split(/[,;\n]+/).map(cleanText).filter(Boolean);
  const known = new Set(track.tags.map((tag) => tag.text.toLowerCase()));
  for (const text of candidates) {
    if (!known.has(text.toLowerCase())) track.tags.push({ text, weight: "normal" });
    known.add(text.toLowerCase());
  }
  track.tagInput.value = "";
  if (candidates.length) trackPromptChanged(track);
}

function removeTag(track, index) {
  if (track.tags.length <= 1) return toast("Each track needs at least one prompt idea.", true);
  track.tags.splice(index, 1);
  trackPromptChanged(track);
}

function changeTagWeight(track, index, direction) {
  const tag = track.tags[index];
  const next = Math.max(0, Math.min(2, WEIGHTS.indexOf(tag.weight) + direction));
  if (tag.weight === WEIGHTS[next]) return;
  tag.weight = WEIGHTS[next];
  trackPromptChanged(track);
}

function trackPromptChanged(track) {
  track.dirty = true;
  track.sync.textContent = "UPDATING";
  track.sync.classList.add("syncing");
  renderTags(track);
  clearTimeout(track.syncTimer);
  track.syncTimer = setTimeout(() => saveTrack(track, true), 420);
}

function number(value, fallback) {
  const result = Number(value);
  return Number.isFinite(result) ? result : fallback;
}

function collectTrackSettings(track, promptOnly = false) {
  const base = { prompt_tags: track.tags.map((tag) => ({ ...tag })) };
  if (promptOnly) return base;
  return {
    ...base,
    model: track.modelSelect.value,
    role_prompt: track.rolePrompt.value.trim(),
    negative_prompt: track.negativePrompt.value.trim() || null,
    beats: Math.round(number(track.beats.value, 16)),
    mutation: Math.min(1, Math.max(.01, number(track.mutation.value, .14))),
    cfg: number(track.cfg.value, 2),
    steps: Math.round(number(track.steps.value, 8)),
    mutation_interval_loops: Math.round(number(track.mutationInterval.value, 1)),
    seed_mode: track.seedMode.value,
    seed: track.seedMode.value === "random" ? null : Math.round(number(track.seed.value, 42000)),
  };
}

async function saveTrack(track, promptOnly = false) {
  clearTimeout(track.syncTimer);
  try {
    const result = await api(`/tracks/${track.id}`, { method: "PUT", body: JSON.stringify(collectTrackSettings(track, promptOnly)) });
    track.formRevision = result.config.revision;
    track.dirty = false;
    track.sync.textContent = statusState?.running && !result.config.frozen ? "LIVE" : "SAVED";
    track.sync.classList.remove("syncing");
    track.constructed.textContent = result.config.constructed_prompt;
    updateDuration(track, result.config.beats, result.config.loop_seconds);
  } catch (error) {
    track.sync.textContent = "ERROR";
    toast(error.message, true);
  }
}

function syncGlobalMutationFromTracks() {
  const values = TRACK_IDS.map((id) => number(tracks[id].mutation.value, .14));
  const allEqual = values.every((value) => Math.abs(value - values[0]) < .0001);
  globalElements.globalMutation.dataset.mixed = allEqual ? "false" : "true";
  if (allEqual) globalElements.globalMutation.value = values[0];
  globalElements.globalMutationValue.textContent = allEqual ? values[0].toFixed(2) : "MIXED";
}

function setGlobalMutation(value) {
  const mutation = Math.min(1, Math.max(.01, number(value, .14)));
  globalElements.globalMutationValue.textContent = mutation.toFixed(2);
  globalElements.globalMutation.dataset.mixed = "false";
  for (const id of TRACK_IDS) {
    tracks[id].mutation.value = mutation.toFixed(2);
    tracks[id].dirty = true;
  }
  clearTimeout(globalMutationTimer);
  globalMutationTimer = setTimeout(async () => {
    try {
      await Promise.all(TRACK_IDS.map((id) => saveTrack(tracks[id])));
      toast(`Mutation set to ${mutation.toFixed(2)} on all tracks`);
    } catch (error) {
      toast(error.message, true);
    }
  }, 280);
}

function setTrackForm(track, trackStatus, force = false) {
  const config = trackStatus.config;
  if (track.dirty && !force) return;
  track.tags = (config.prompt_tags || []).map((tag) => ({ text: cleanText(tag.text), weight: WEIGHTS.includes(tag.weight) ? tag.weight : "normal" }));
  track.rolePrompt.value = config.role_prompt || "";
  track.negativePrompt.value = config.negative_prompt || "";
  track.modelSelect.value = config.model || track.status?.model || (track.id === "texture" ? "sm-sfx" : "sm-music");
  track.modelBadge.textContent = track.modelSelect.value === "medium" ? "SA3 MEDIUM · SAME-L" : `SA3 ${track.modelSelect.value === "sm-sfx" ? "SMALL SFX" : "SMALL MUSIC"}`;
  ensureSelectOption(track.beats, config.beats);
  track.beats.value = config.beats;
  track.mutation.value = Math.min(1, Math.max(.01, config.mutation));
  track.cfg.value = config.cfg;
  track.steps.value = config.steps;
  track.mutationInterval.value = config.mutation_interval_loops || 1;
  track.seedMode.value = config.seed_mode;
  track.seed.value = config.seed ?? 42000;
  track.seed.disabled = config.seed_mode === "random";
  track.formRevision = config.revision;
  track.dirty = false;
  renderTags(track);
  updateDuration(track, config.beats, config.loop_seconds);
}

function ensureSelectOption(select, value) {
  if (![...select.options].some((option) => Number(option.value) === Number(value))) {
    select.append(new Option(String(value), String(value)));
  }
}

function updateDuration(track, beats = number(track.beats.value, 16), seconds = null) {
  const duration = seconds ?? beats * 60 / number(globalElements.bpm.value, 120);
  track.duration.textContent = `${beats} BEATS · ${duration.toFixed(1)}S`;
}

function collectMixer() {
  return Object.fromEntries(TRACK_IDS.map((id) => {
    const track = tracks[id];
    return [id, {
      volume: number(track.volume.value, .82), muted: track.mute.classList.contains("active"),
      solo: track.solo.classList.contains("active"), highpass_hz: number(track.highpass.value, 20),
      lowpass_hz: number(track.lowpass.value, 20000), reverb_send: number(track.reverbSend.value, .18),
      delay_send: number(track.delaySend.value, 0),
    }];
  }));
}

function collectSidechain() {
  return {
    enabled: globalElements.sidechainEnabled.checked,
    duck_melodic: globalElements.sidechainMelodic.checked,
    duck_texture: globalElements.sidechainTexture.checked,
    threshold_db: number(globalElements.scThreshold.value, -24), ratio: number(globalElements.scRatio.value, 4),
    attack_ms: number(globalElements.scAttack.value, 12), release_ms: number(globalElements.scRelease.value, 180),
    depth: number(globalElements.scDepth.value, 1), makeup_db: number(globalElements.scMakeup.value, 0),
  };
}

function collectReverb() {
  return {
    enabled: globalElements.reverbEnabled.checked, return_level: number(globalElements.rvReturn.value, .24),
    decay_seconds: number(globalElements.rvDecay.value, 3.5), predelay_ms: number(globalElements.rvPredelay.value, 24),
    damping_hz: number(globalElements.rvDamping.value, 7000),
  };
}

function collectDelay() {
  return {
    enabled: globalElements.delayEnabled.checked,
    return_level: number(globalElements.delayReturn.value, .24),
    division: globalElements.delayDivision.value || "1/4",
    feedback: number(globalElements.delayFeedback.value, .36),
    lowpass_hz: number(globalElements.delayLowpass.value, 6500),
    stereo_width: number(globalElements.delayWidth.value, .7),
  };
}

function syncedDelaySeconds(division = collectDelay().division) {
  const beats = { "1/1": 4, "1/2": 2, "1/4D": 1.5, "1/4": 1, "1/8D": .75, "1/8": .5, "1/8T": 1 / 3, "1/16": .25 };
  return (beats[division] || 1) * 60 / number(globalElements.bpm.value, 120);
}

function queueSessionSave(includeBpm = false) {
  applyMixer();
  updateControlOutputs();
  clearTimeout(sessionSaveTimer);
  sessionSaveTimer = setTimeout(async () => {
    const body = {
      mixer: collectMixer(), sidechain: collectSidechain(), reverb: collectReverb(), delay: collectDelay(),
      master_volume: number(globalElements.master.value, .82),
    };
    if (includeBpm) body.bpm = number(globalElements.bpm.value, 120);
    try {
      const result = await api("/session", { method: "PUT", body: JSON.stringify(body) });
      sessionState = result.session;
      globalElements.tempoNote.textContent = `${result.session.bpm} BPM · QUANTIZED BUFFER SWAPS`;
      TRACK_IDS.forEach((id) => updateDuration(tracks[id]));
    } catch (error) {
      toast(error.message, true);
      if (statusState) setSessionForm(statusState.session, true);
    }
  }, 220);
}

function setSessionForm(session, force = false) {
  if (!session || (!force && sessionState?.revision === session.revision)) return;
  sessionState = session;
  globalElements.bpm.value = session.bpm;
  globalElements.master.value = session.master_volume;
  for (const id of TRACK_IDS) {
    const mix = session.mixer[id];
    const track = tracks[id];
    track.volume.value = mix.volume;
    track.highpass.value = mix.highpass_hz;
    track.lowpass.value = mix.lowpass_hz;
    track.reverbSend.value = mix.reverb_send;
    track.delaySend.value = mix.delay_send ?? 0;
    track.mute.classList.toggle("active", mix.muted);
    track.solo.classList.toggle("active", mix.solo);
  }
  const sc = session.sidechain;
  globalElements.sidechainEnabled.checked = sc.enabled;
  globalElements.sidechainMelodic.checked = sc.duck_melodic ?? true;
  globalElements.sidechainTexture.checked = sc.duck_texture ?? false;
  globalElements.scThreshold.value = sc.threshold_db;
  globalElements.scRatio.value = sc.ratio;
  globalElements.scAttack.value = sc.attack_ms;
  globalElements.scRelease.value = sc.release_ms;
  globalElements.scDepth.value = sc.depth;
  globalElements.scMakeup.value = sc.makeup_db;
  const rv = session.reverb;
  globalElements.reverbEnabled.checked = rv.enabled;
  globalElements.rvReturn.value = rv.return_level;
  globalElements.rvDecay.value = rv.decay_seconds;
  globalElements.rvPredelay.value = rv.predelay_ms;
  globalElements.rvDamping.value = rv.damping_hz;
  const delay = session.delay || { enabled: false, return_level: .24, division: "1/4", feedback: .36, lowpass_hz: 6500, stereo_width: .7 };
  globalElements.delayEnabled.checked = delay.enabled;
  globalElements.delayReturn.value = delay.return_level;
  globalElements.delayDivision.value = delay.division;
  globalElements.delayFeedback.value = delay.feedback;
  globalElements.delayLowpass.value = delay.lowpass_hz;
  globalElements.delayWidth.value = delay.stereo_width;
  updateControlOutputs();
  applyMixer();
}

function outputFor(input, formatter) {
  const output = input.closest("label")?.querySelector("output");
  if (output) output.textContent = formatter(number(input.value, 0));
}

function hz(value) {
  return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}k` : `${Math.round(value)}`;
}

function updateControlOutputs() {
  TRACK_IDS.forEach((id) => {
    const t = tracks[id];
    outputFor(t.volume, (v) => `${Math.round(v * 100)}%`);
    outputFor(t.highpass, (v) => `${hz(v)} Hz`);
    outputFor(t.lowpass, (v) => `${hz(v)} Hz`);
    outputFor(t.reverbSend, (v) => `${Math.round(v * 100)}%`);
    outputFor(t.delaySend, (v) => `${Math.round(v * 100)}%`);
  });
  outputFor(globalElements.scThreshold, (v) => `${v.toFixed(0)} dB`);
  outputFor(globalElements.scRatio, (v) => `${v.toFixed(1)}:1`);
  outputFor(globalElements.scAttack, (v) => `${v.toFixed(v < 10 ? 1 : 0)} ms`);
  outputFor(globalElements.scRelease, (v) => `${v.toFixed(0)} ms`);
  outputFor(globalElements.scDepth, (v) => `${Math.round(v * 100)}%`);
  outputFor(globalElements.scMakeup, (v) => `${v.toFixed(1)} dB`);
  outputFor(globalElements.rvReturn, (v) => `${Math.round(v * 100)}%`);
  outputFor(globalElements.rvDecay, (v) => `${v.toFixed(1)} s`);
  outputFor(globalElements.rvPredelay, (v) => `${v.toFixed(0)} ms`);
  outputFor(globalElements.rvDamping, (v) => `${hz(v)} Hz`);
  outputFor(globalElements.delayReturn, (v) => `${Math.round(v * 100)}%`);
  outputFor(globalElements.delayFeedback, (v) => `${Math.round(v * 100)}%`);
  outputFor(globalElements.delayLowpass, (v) => `${hz(v)} Hz`);
  outputFor(globalElements.delayWidth, (v) => `${Math.round(v * 100)}%`);
  globalElements.delayTimeReadout.textContent = `${Math.round(syncedDelaySeconds() * 1000)} ms`;
}

function ensureAudio() {
  if (audio.context) return audio.context;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) throw new Error("This browser cannot run the multitrack audio mixer.");
  const context = new AudioContextClass({ latencyHint: "interactive" });
  audio.context = context;
  audio.master = context.createGain();
  audio.limiter = context.createDynamicsCompressor();
  audio.limiter.threshold.value = -2;
  audio.limiter.knee.value = 4;
  audio.limiter.ratio.value = 16;
  audio.limiter.attack.value = .003;
  audio.limiter.release.value = .12;
  audio.master.connect(audio.limiter);
  audio.limiter.connect(context.destination);
  audio.meterSplitter = context.createChannelSplitter(2);
  audio.meterLeft = context.createAnalyser();
  audio.meterRight = context.createAnalyser();
  audio.meterLeft.fftSize = 512;
  audio.meterRight.fftSize = 512;
  audio.meterLeftData = new Float32Array(audio.meterLeft.fftSize);
  audio.meterRightData = new Float32Array(audio.meterRight.fftSize);
  audio.limiter.connect(audio.meterSplitter);
  audio.meterSplitter.connect(audio.meterLeft, 0);
  audio.meterSplitter.connect(audio.meterRight, 1);

  audio.reverbInput = context.createGain();
  audio.predelay = context.createDelay(.6);
  audio.convolver = context.createConvolver();
  audio.reverbFilter = context.createBiquadFilter();
  audio.reverbFilter.type = "lowpass";
  audio.reverbReturn = context.createGain();
  audio.reverbInput.connect(audio.predelay).connect(audio.convolver).connect(audio.reverbFilter).connect(audio.reverbReturn).connect(audio.master);

  audio.delayInput = context.createGain();
  audio.delayLeft = context.createDelay(10);
  audio.delayRight = context.createDelay(10);
  audio.delayFilterLeft = context.createBiquadFilter(); audio.delayFilterLeft.type = "lowpass";
  audio.delayFilterRight = context.createBiquadFilter(); audio.delayFilterRight.type = "lowpass";
  audio.delayFeedbackLeft = context.createGain();
  audio.delayFeedbackRight = context.createGain();
  audio.delayPanLeft = context.createStereoPanner();
  audio.delayPanRight = context.createStereoPanner();
  audio.delayReturn = context.createGain();
  audio.delayInput.connect(audio.delayLeft);
  audio.delayLeft.connect(audio.delayFilterLeft);
  audio.delayFilterLeft.connect(audio.delayPanLeft).connect(audio.delayReturn);
  audio.delayFilterLeft.connect(audio.delayFeedbackLeft).connect(audio.delayRight);
  audio.delayRight.connect(audio.delayFilterRight);
  audio.delayFilterRight.connect(audio.delayPanRight).connect(audio.delayReturn);
  audio.delayFilterRight.connect(audio.delayFeedbackRight).connect(audio.delayLeft);
  audio.delayReturn.connect(audio.master);

  for (const id of TRACK_IDS) {
    const input = context.createGain();
    const highpass = context.createBiquadFilter(); highpass.type = "highpass"; highpass.Q.value = .707;
    const lowpass = context.createBiquadFilter(); lowpass.type = "lowpass"; lowpass.Q.value = .707;
    const duck = context.createGain();
    const fader = context.createGain();
    const analyser = context.createAnalyser(); analyser.fftSize = 512; analyser.smoothingTimeConstant = .55;
    const send = context.createGain();
    const delaySend = context.createGain();
    input.connect(highpass).connect(lowpass).connect(duck).connect(fader);
    fader.connect(analyser).connect(audio.master);
    fader.connect(send).connect(audio.reverbInput);
    fader.connect(delaySend).connect(audio.delayInput);
    tracks[id].nodes = { input, highpass, lowpass, duck, fader, analyser, send, delaySend };
  }
  rebuildReverb();
  rebuildDelay();
  applyMixer();
  startSidechainFollower();
  startMasterMeter();
  return context;
}

function startMasterMeter() {
  if (audio.meterFrame) return;
  const draw = () => {
    if (!audio.meterLeft || !audio.meterRight) return;
    audio.meterLeft.getFloatTimeDomainData(audio.meterLeftData);
    audio.meterRight.getFloatTimeDomainData(audio.meterRightData);
    const peaks = [audio.meterLeftData, audio.meterRightData].map((samples) => {
      let peak = 0;
      for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
      return peak;
    });
    audio.meterLevels = audio.meterLevels.map((previous, index) => Math.max(peaks[index], previous * .91));
    const decibels = audio.meterLevels.map((level) => level > .00001 ? 20 * Math.log10(level) : -Infinity);
    const percent = decibels.map((db) => Number.isFinite(db) ? Math.max(0, Math.min(100, (db + 60) / 60 * 100)) : 0);
    globalElements.masterMeterLeft.style.width = `${percent[0]}%`;
    globalElements.masterMeterRight.style.width = `${percent[1]}%`;
    const maximum = Math.max(...decibels);
    globalElements.masterMeterValue.textContent = Number.isFinite(maximum) ? `${maximum.toFixed(1)} dB` : "−∞ dB";
    audio.meterFrame = requestAnimationFrame(draw);
  };
  audio.meterFrame = requestAnimationFrame(draw);
}

function rebuildReverb() {
  if (!audio.context || !sessionState) return;
  const rv = collectReverb();
  const key = `${rv.decay_seconds}:${audio.context.sampleRate}`;
  if (key !== audio.impulseKey) {
    const length = Math.max(1, Math.round(audio.context.sampleRate * rv.decay_seconds));
    const impulse = audio.context.createBuffer(2, length, audio.context.sampleRate);
    for (let channel = 0; channel < 2; channel++) {
      const data = impulse.getChannelData(channel);
      for (let i = 0; i < length; i++) {
        const envelope = Math.pow(1 - i / length, 2.25);
        data[i] = (Math.random() * 2 - 1) * envelope;
      }
    }
    audio.convolver.buffer = impulse;
    audio.impulseKey = key;
  }
  audio.predelay.delayTime.setTargetAtTime(rv.predelay_ms / 1000, audio.context.currentTime, .02);
  audio.reverbFilter.frequency.setTargetAtTime(rv.damping_hz, audio.context.currentTime, .02);
  audio.reverbReturn.gain.setTargetAtTime(rv.enabled ? rv.return_level : 0, audio.context.currentTime, .02);
}

function rebuildDelay() {
  if (!audio.context || !sessionState) return;
  const delay = collectDelay();
  const now = audio.context.currentTime;
  const seconds = syncedDelaySeconds(delay.division);
  audio.delayLeft.delayTime.setTargetAtTime(seconds, now, .02);
  audio.delayRight.delayTime.setTargetAtTime(seconds, now, .02);
  audio.delayFeedbackLeft.gain.setTargetAtTime(delay.feedback, now, .02);
  audio.delayFeedbackRight.gain.setTargetAtTime(delay.feedback, now, .02);
  audio.delayFilterLeft.frequency.setTargetAtTime(delay.lowpass_hz, now, .02);
  audio.delayFilterRight.frequency.setTargetAtTime(delay.lowpass_hz, now, .02);
  audio.delayPanLeft.pan.setTargetAtTime(-delay.stereo_width, now, .02);
  audio.delayPanRight.pan.setTargetAtTime(delay.stereo_width, now, .02);
  audio.delayReturn.gain.setTargetAtTime(delay.enabled ? delay.return_level : 0, now, .02);
}

function applyMixer() {
  if (!audio.context || !sessionState) return;
  const now = audio.context.currentTime;
  const mix = collectMixer();
  const anySolo = TRACK_IDS.some((id) => mix[id].solo);
  audio.master.gain.setTargetAtTime(number(globalElements.master.value, .82), now, .015);
  for (const id of TRACK_IDS) {
    const state = mix[id];
    const nodes = tracks[id].nodes;
    const audible = !state.muted && (!anySolo || state.solo);
    nodes.fader.gain.setTargetAtTime(audible ? state.volume : 0, now, .012);
    nodes.highpass.frequency.setTargetAtTime(state.highpass_hz, now, .015);
    nodes.lowpass.frequency.setTargetAtTime(state.lowpass_hz, now, .015);
    nodes.send.gain.setTargetAtTime(state.reverb_send, now, .015);
    nodes.delaySend.gain.setTargetAtTime(state.delay_send, now, .015);
  }
  rebuildReverb();
  rebuildDelay();
}

function startSidechainFollower() {
  clearInterval(audio.sidechainTimer);
  const waveform = new Float32Array(512);
  audio.sidechainTimer = setInterval(() => {
    if (!audio.context || !tracks.melodic.nodes) return;
    const sc = collectSidechain();
    let reduction = 0;
    if (audio.playing && sc.enabled && tracks.percussion.source) {
      tracks.percussion.nodes.analyser.getFloatTimeDomainData(waveform);
      let power = 0;
      for (const sample of waveform) power += sample * sample;
      const db = 20 * Math.log10(Math.max(1e-6, Math.sqrt(power / waveform.length)));
      const over = Math.max(0, db - sc.threshold_db);
      reduction = (over - over / sc.ratio) * sc.depth;
    }
    const targetDb = sc.enabled ? sc.makeup_db - reduction : 0;
    const target = Math.pow(10, targetDb / 20);
    const falling = reduction > audio.currentReduction;
    const time = (falling ? sc.attack_ms : sc.release_ms) / 1000;
    const smoothing = Math.max(.001, time / 3);
    tracks.melodic.nodes.duck.gain.setTargetAtTime(sc.enabled && sc.duck_melodic ? target : 1, audio.context.currentTime, smoothing);
    tracks.texture.nodes.duck.gain.setTargetAtTime(sc.enabled && sc.duck_texture ? target : 1, audio.context.currentTime, smoothing);
    audio.currentReduction = reduction;
    globalElements.reductionBar.style.width = `${Math.min(100, reduction / 24 * 100)}%`;
    globalElements.reductionLabel.textContent = `${reduction.toFixed(1)} dB REDUCTION`;
  }, 30);
}

function recordUrl(trackId, record) {
  if (record.audio_url) return record.audio_url;
  return record.generation > 0
    ? `${API_BASE}/tracks/${trackId}/audio/versions/${record.generation}.wav`
    : `${API_BASE}/tracks/${trackId}/audio/current.wav`;
}

async function loadEntry(trackId, record) {
  const key = `${trackId}:${record.generation}:${record.created_at || "seed"}`;
  if (audioCache.has(key)) return audioCache.get(key);
  const promise = (async () => {
    const context = ensureAudio();
    const response = await fetch(recordUrl(trackId, record), { cache: record.generation > 0 ? "force-cache" : "no-store" });
    if (!response.ok) throw new Error(`Could not load ${trackId} generation ${record.generation}`);
    const encoded = await response.arrayBuffer();
    const buffer = await context.decodeAudioData(encoded.slice(0));
    return { record, buffer, key };
  })();
  audioCache.set(key, promise);
  try {
    const entry = await promise;
    audioCache.set(key, entry);
    return entry;
  } catch (error) {
    audioCache.delete(key);
    throw error;
  }
}

function statusRecord(trackStatus) {
  if (trackStatus.last_completed) return trackStatus.last_completed;
  if (!trackStatus.current_available) return null;
  return {
    track_id: trackStatus.id, generation: 0, mode: "external-seed", audio_url: trackStatus.current_audio_url,
    prompt: trackStatus.config.constructed_prompt, bpm: statusState.session.bpm, beats: trackStatus.config.beats,
    loop_seconds: trackStatus.config.loop_seconds, created_at: "seed",
  };
}

async function acceptRecord(track, record) {
  if (!record) return;
  const currentGeneration = track.pending?.record.generation ?? track.active?.record.generation;
  const currentCreated = track.pending?.record.created_at ?? track.active?.record.created_at;
  if (Number(currentGeneration) === Number(record.generation) && currentCreated === record.created_at) return;
  if (track.requestedGeneration === `${record.generation}:${record.created_at}`) return;
  track.requestedGeneration = `${record.generation}:${record.created_at}`;
  try {
    const entry = await loadEntry(track.id, record);
    if (track.requestedGeneration !== `${record.generation}:${record.created_at}`) return;
    track.requestedGeneration = null;
    if (!audio.playing) {
      track.active = entry;
      track.pending = null;
      return;
    }
    track.pending = entry;
    schedulerTick();
  } catch (error) {
    track.requestedGeneration = null;
    toast(error.message, true);
  }
}

function createSource(track, entry, initialGain = 1) {
  const source = audio.context.createBufferSource();
  const gain = audio.context.createGain();
  source.buffer = entry.buffer;
  source.loop = true;
  source.loopStart = 0;
  source.loopEnd = entry.buffer.duration;
  gain.gain.value = initialGain;
  source.connect(gain).connect(track.nodes.input);
  return { source, gain };
}

async function startPlayback() {
  if (audio.playing) return pausePlayback();
  globalElements.playButton.disabled = true;
  try {
    ensureAudio();
    await audio.context.resume();
    const available = TRACK_IDS.filter((id) => tracks[id].active);
    if (!available.length) throw new Error("Generate or import at least one track first.");
    const entries = await Promise.all(available.map((id) => loadEntry(id, tracks[id].active.record)));
    const when = audio.context.currentTime + .06;
    audio.bpm = number(sessionState?.bpm, 120);
    audio.anchorTime = when - audio.pausedBeats * 60 / audio.bpm;
    available.forEach((id, index) => {
      const track = tracks[id];
      track.active = entries[index];
      const nodes = createSource(track, track.active, 1);
      const offset = ((audio.pausedBeats * 60 / audio.bpm) % track.active.buffer.duration + track.active.buffer.duration) % track.active.buffer.duration;
      nodes.source.start(when, offset);
      track.source = nodes.source;
      track.sourceGain = nodes.gain;
      track.loopAnchor = when - offset;
    });
    audio.playing = true;
    document.body.classList.add("playing");
    globalElements.playIcon.textContent = "Ⅱ";
    globalElements.playButton.setAttribute("aria-label", "Pause all tracks");
    globalElements.transportState.textContent = "PLAYING";
    audio.scheduler = setInterval(schedulerTick, 25);
    uiTick();
  } catch (error) {
    toast(error.message, true);
  } finally {
    globalElements.playButton.disabled = false;
  }
}

function pausePlayback() {
  if (!audio.playing) return;
  audio.pausedBeats = transportBeats();
  clearInterval(audio.scheduler); audio.scheduler = null;
  for (const id of TRACK_IDS) {
    const track = tracks[id];
    clearTrackTransition(track);
    try { track.source?.stop(); } catch (_) { /* already stopped */ }
    track.source = null;
    track.sourceGain = null;
    if (track.pending) {
      track.active = track.pending;
      track.pending = null;
    }
  }
  audio.playing = false;
  document.body.classList.remove("playing");
  globalElements.playIcon.textContent = "▶";
  globalElements.playButton.setAttribute("aria-label", "Play all tracks");
  globalElements.transportState.textContent = "PAUSED";
}

function transportBeats() {
  if (!audio.context || !audio.playing) return audio.pausedBeats;
  return Math.max(0, (audio.context.currentTime - audio.anchorTime) * audio.bpm / 60);
}

function schedulerTick() {
  if (!audio.playing || !audio.context) return;
  const targetBpm = number(sessionState?.bpm, audio.bpm);
  if (targetBpm !== audio.bpm && tempoEntriesReady(targetBpm)) {
    scheduleTempoSwap(targetBpm);
    return;
  }
  const now = audio.context.currentTime;
  for (const id of TRACK_IDS) {
    const track = tracks[id];
    if (!track.pending || track.transition) continue;
    if (number(track.pending.record.bpm, audio.bpm) !== audio.bpm) continue;
    if (!track.source) {
      const beatSeconds = 60 / audio.bpm;
      const boundary = audio.anchorTime + Math.ceil((now - audio.anchorTime) / beatSeconds) * beatSeconds;
      if (boundary - now <= LOOKAHEAD) scheduleTrackSwap(track, boundary);
      continue;
    }
    const beatSeconds = 60 / audio.bpm;
    const transportPosition = Math.max(0, (now - audio.anchorTime) / beatSeconds);
    const loopBeats = Math.max(1, number(track.active?.record.beats, number(track.beats.value, 16)));
    const nextBoundaryBeat = (Math.floor((transportPosition + 1e-6) / loopBeats) + 1) * loopBeats;
    const boundary = audio.anchorTime + nextBoundaryBeat * beatSeconds;
    if (boundary - now <= LOOKAHEAD) scheduleTrackSwap(track, boundary);
  }
}

function scheduleTrackSwap(track, boundary) {
  if (!track.pending || track.transition) return;
  const entry = track.pending;
  track.pending = null;
  const nodes = createSource(track, entry, 0);
  const fade = Math.min(CROSSFADE, entry.buffer.duration / 6);
  nodes.gain.gain.setValueAtTime(0, boundary);
  nodes.gain.gain.linearRampToValueAtTime(1, boundary + fade);
  nodes.source.start(boundary, 0);
  const oldSource = track.source;
  const oldGain = track.sourceGain;
  if (oldGain) {
    oldGain.gain.setValueAtTime(1, boundary);
    oldGain.gain.linearRampToValueAtTime(0, boundary + fade);
    oldSource.stop(boundary + fade + .01);
  }
  const transition = { entry, source: nodes.source, gain: nodes.gain, boundary, timer: null };
  transition.timer = setTimeout(() => {
    if (track.transition !== transition) return;
    track.active = entry;
    track.source = nodes.source;
    track.sourceGain = nodes.gain;
    track.loopAnchor = boundary;
    track.transition = null;
    document.documentElement.dataset.lastSwap = `${track.id}:${entry.record.generation}`;
  }, Math.max(0, (boundary + fade - audio.context.currentTime) * 1000));
  track.transition = transition;
}

function clearTrackTransition(track) {
  if (!track.transition) return;
  clearTimeout(track.transition.timer);
  try { track.transition.source.stop(); } catch (_) { /* not started */ }
  track.transition = null;
}

function tempoEntriesReady(targetBpm) {
  return TRACK_IDS.every((id) => {
    const entry = tracks[id].pending || tracks[id].active;
    return entry && number(entry.record.bpm, targetBpm) === targetBpm;
  });
}

function scheduleTempoSwap(targetBpm) {
  if (TRACK_IDS.some((id) => tracks[id].transition?.tempo)) return;
  const beatSeconds = 60 / audio.bpm;
  const barSeconds = beatSeconds * 4;
  const now = audio.context.currentTime;
  const boundary = audio.anchorTime + Math.ceil((now - audio.anchorTime) / barSeconds) * barSeconds;
  if (boundary - now > LOOKAHEAD) return;
  const replacements = TRACK_IDS.map((id) => [tracks[id], tracks[id].pending || tracks[id].active]);
  for (const [track, entry] of replacements) {
    track.pending = null;
    const nodes = createSource(track, entry, 0);
    nodes.gain.gain.setValueAtTime(0, boundary);
    nodes.gain.gain.linearRampToValueAtTime(1, boundary + CROSSFADE);
    nodes.source.start(boundary, 0);
    if (track.sourceGain) {
      track.sourceGain.gain.setValueAtTime(1, boundary);
      track.sourceGain.gain.linearRampToValueAtTime(0, boundary + CROSSFADE);
      track.source.stop(boundary + CROSSFADE + .01);
    }
    const transition = { tempo: true, entry, source: nodes.source, gain: nodes.gain, boundary, timer: null };
    transition.timer = setTimeout(() => {
      track.active = entry;
      track.source = nodes.source;
      track.sourceGain = nodes.gain;
      track.loopAnchor = boundary;
      track.transition = null;
    }, Math.max(0, (boundary + CROSSFADE - now) * 1000));
    track.transition = transition;
  }
  setTimeout(() => {
    audio.bpm = targetBpm;
    audio.anchorTime = boundary;
    audio.pausedBeats = 0;
    toast(`Tempo set changed to ${targetBpm} BPM`);
  }, Math.max(0, (boundary - now) * 1000));
}

function uiTick() {
  cancelAnimationFrame(uiTickHandle);
  const draw = () => {
    const beats = transportBeats();
    const bar = Math.floor(beats / 4) + 1;
    const beat = Math.floor(beats % 4) + 1;
    globalElements.beatReadout.textContent = `BAR ${bar} · BEAT ${beat}`;
    if (audio.playing) uiTickHandle = requestAnimationFrame(draw);
  };
  draw();
}

async function toggleEvolution() {
  try {
    globalElements.evolve.disabled = true;
    const result = statusState?.running
      ? await api("/loop/stop", { method: "POST" })
      : await api("/loop/start", { method: "POST" });
    renderStatus(result.status);
    toast(statusState.running ? "Evolution active" : "Evolution stopped");
  } catch (error) {
    toast(error.message, true);
  } finally {
    globalElements.evolve.disabled = false;
  }
}

async function toggleFreeze(track) {
  const frozen = !track.status.config.frozen;
  try {
    await api(`/tracks/${track.id}/freeze`, { method: "PUT", body: JSON.stringify({ frozen }) });
    toast(`${track.status.label} ${frozen ? "frozen" : "evolving"}`);
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function toggleFreezeAll() {
  const allFrozen = TRACK_IDS.every((id) => tracks[id].status?.config.frozen);
  try {
    const result = await api("/loop/freeze", { method: "PUT", body: JSON.stringify({ frozen: !allFrozen }) });
    renderStatus(result.status);
    toast(allFrozen ? "All tracks unfrozen" : "All tracks frozen");
  } catch (error) {
    toast(error.message, true);
  }
}

async function importReference(track, file) {
  if (!file) return;
  if (file.size > 100 * 1024 * 1024) return toast("Reference files are limited to 100 MB.", true);
  track.referenceDrop.classList.add("busy");
  const label = $("strong", track.referenceDrop);
  const previous = label.textContent;
  label.textContent = "PREPARING REFERENCE…";
  try {
    const context = ensureAudio();
    const decoded = await context.decodeAudioData((await file.arrayBuffer()).slice(0));
    const seconds = number(track.status?.config.loop_seconds, number(track.beats.value, 16) * 60 / number(globalElements.bpm.value, 120));
    const wav = audioBufferToWav(decoded, seconds);
    const action = decoded.duration > seconds + .01 ? "trimmed" : decoded.duration < seconds - .01 ? "padded with silence" : "fitted";
    label.textContent = "UPLOADING REFERENCE…";
    const result = await api(`/tracks/${track.id}/reference`, {
      method: "PUT", body: wav,
      headers: { "Content-Type": "audio/wav", "X-File-Name": encodeURIComponent(file.name) },
    });
    toast(`${file.name} ${action} to ${seconds.toFixed(1)}s and loaded into ${track.status.label}`);
    await refreshStatus();
    await refreshHistory();
    return result;
  } catch (error) {
    toast(`Could not import ${file.name}: ${error.message}`, true);
  } finally {
    label.textContent = previous;
    track.referenceDrop.classList.remove("busy", "dragging");
    track.referenceInput.value = "";
  }
}

function audioBufferToWav(buffer, targetSeconds) {
  const sampleRate = buffer.sampleRate;
  const frames = Math.max(1, Math.round(targetSeconds * sampleRate));
  const channels = 2;
  const bytesPerSample = 2;
  const dataLength = frames * channels * bytesPerSample;
  const array = new ArrayBuffer(44 + dataLength);
  const view = new DataView(array);
  const write = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + dataLength, true); write(8, "WAVE"); write(12, "fmt ");
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true); view.setUint16(34, 16, true); write(36, "data"); view.setUint32(40, dataLength, true);
  const left = buffer.getChannelData(0);
  const right = buffer.numberOfChannels > 1 ? buffer.getChannelData(1) : left;
  let offset = 44;
  for (let frame = 0; frame < frames; frame++) {
    const samples = [frame < left.length ? left[frame] : 0, frame < right.length ? right[frame] : 0];
    for (const raw of samples) {
      const sample = Math.max(-1, Math.min(1, raw));
      view.setInt16(offset, sample < 0 ? sample * 32768 : sample * 32767, true);
      offset += 2;
    }
  }
  return array;
}

function renderStatus(status) {
  statusState = status;
  setSessionForm(status.session);
  const phase = status.phase;
  globalElements.statusPill.dataset.phase = phase;
  globalElements.statusLabel.textContent = phase === "generating" ? "RENDERING" : status.running ? "ACTIVE" : "READY";
  globalElements.evolve.textContent = status.running ? "STOP EVOLUTION" : "START EVOLUTION";
  globalElements.evolve.classList.toggle("stop", status.running);
  globalElements.queueStrip.classList.toggle("rendering", phase === "generating");
  if (phase === "generating") {
    const label = status.tracks[status.active_track_id]?.label || status.active_track_id;
    globalElements.queueTitle.textContent = `RENDERING ${String(label).toUpperCase()}`;
    globalElements.queueDetail.textContent = `JOB ${String(status.active_job_id || "—").toUpperCase()}`;
  } else {
    globalElements.queueTitle.textContent = status.running ? "RENDER QUEUE ACTIVE" : "RENDER QUEUE IDLE";
    globalElements.queueDetail.textContent = `One safe ${backendLabel} job at a time`;
  }
  let available = false;
  for (const id of TRACK_IDS) {
    const track = tracks[id];
    const trackStatus = status.tracks[id];
    track.status = trackStatus;
    available ||= trackStatus.current_available;
    if (track.formRevision == null || (track.formRevision !== trackStatus.config.revision && !track.dirty)) setTrackForm(track, trackStatus);
    track.root.classList.toggle("frozen", trackStatus.config.frozen);
    track.freeze.classList.toggle("active", trackStatus.config.frozen);
    track.freeze.textContent = trackStatus.config.frozen ? "UNFREEZE" : "FREEZE";
    track.phase.textContent = trackStatus.config.frozen ? "FROZEN" : String(trackStatus.phase).toUpperCase();
    track.generation.textContent = trackStatus.generation ? `GEN ${String(trackStatus.generation).padStart(3, "0")}` : trackStatus.current_available ? "SEED" : "GEN —";
    track.sync.textContent = trackStatus.config.frozen ? "FROZEN" : status.running ? "LIVE" : "SAVED";
    track.error.classList.toggle("hidden", !trackStatus.last_error);
    track.error.textContent = trackStatus.last_error?.message || "";
    const record = statusRecord(trackStatus);
    if (record) void acceptRecord(track, record);
  }
  syncGlobalMutationFromTracks();
  globalElements.playButton.disabled = !available && !TRACK_IDS.some((id) => tracks[id].active);
  const allFrozen = TRACK_IDS.every((id) => status.tracks[id].config.frozen);
  globalElements.freezeAll.classList.toggle("active", allFrozen);
  globalElements.freezeAll.textContent = allFrozen ? "UNFREEZE ALL" : "FREEZE ALL";
}

async function refreshStatus() {
  if (pollBusy) return;
  pollBusy = true;
  try {
    renderStatus(await api("/status"));
  } catch (error) {
    globalElements.statusPill.dataset.phase = "offline";
    globalElements.statusLabel.textContent = "OFFLINE";
  } finally {
    pollBusy = false;
  }
}

async function refreshHistory() {
  await Promise.all(TRACK_IDS.map(async (id) => {
    try {
      const result = await api(`/tracks/${id}/versions?limit=6`);
      renderHistory(tracks[id], result.versions);
    } catch (error) {
      tracks[id].history.replaceChildren(node("div", "history-empty", "History unavailable"));
    }
  }));
}

function formatBytes(bytes) {
  const value = Math.max(0, Number(bytes) || 0);
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
}

async function flushTrackHistory(track) {
  const confirmed = window.confirm(
    `Permanently delete every archived ${track.status?.label || track.id} version?\n\nThe current loop and all settings will be preserved.`,
  );
  if (!confirmed) return;
  try {
    const result = await api(`/tracks/${track.id}/versions`, { method: "DELETE" });
    toast(`${track.status.label} history cleared · ${formatBytes(result.bytes_freed)} freed`);
    await refreshHistory();
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function flushAllHistory() {
  const confirmed = window.confirm(
    "Permanently delete every archived version for all three tracks?\n\nThe three current loops and all settings will be preserved.",
  );
  if (!confirmed) return;
  try {
    const result = await api("/versions", { method: "DELETE" });
    toast(`All history cleared · ${formatBytes(result.bytes_freed)} freed`);
    await refreshHistory();
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

function discardTrackPlayback(track) {
  clearTrackTransition(track);
  try { track.source?.stop(); } catch (_) { /* already stopped */ }
  track.source = null;
  track.sourceGain = null;
  track.active = null;
  track.pending = null;
  track.requestedGeneration = null;
  audioCache.clear();
}

async function clearTrackOutput(track) {
  const label = track.status?.label || track.id;
  const confirmed = window.confirm(
    `Clear the current ${label} loop and every archived ${label} version?\n\nPrompt, model, mixer and generation settings are preserved. If evolution is active, this track will bootstrap again from text.`,
  );
  if (!confirmed) return;
  try {
    const result = await api(`/tracks/${track.id}/output`, { method: "DELETE" });
    discardTrackPlayback(track);
    toast(`${label} output cleared · ${formatBytes(result.bytes_freed)} freed`);
    await refreshHistory();
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function clearAllOutputs() {
  const confirmed = window.confirm(
    "Clear all three current loops and all archived versions?\n\nAll prompts, models, mixer and generation settings are preserved.",
  );
  if (!confirmed) return;
  try {
    const result = await api("/outputs", { method: "DELETE" });
    TRACK_IDS.forEach((id) => discardTrackPlayback(tracks[id]));
    if (audio.playing && !TRACK_IDS.some((id) => tracks[id].active)) pausePlayback();
    toast(`All outputs cleared · ${formatBytes(result.bytes_freed)} freed`);
    await refreshHistory();
    await refreshStatus();
  } catch (error) {
    toast(error.message, true);
  }
}

async function loadBackendInfo() {
  try {
    const response = await fetch("/api/info");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const info = (await response.json()).backend;
    backendLabel = String(info.backend).toUpperCase();
    globalElements.backendBadge.textContent = `${String(info.backend).toUpperCase()} · ${info.device_name} · ${info.dtype}`;
    globalElements.queueDetail.textContent = `One safe ${backendLabel} job at a time`;
  } catch (_) {
    globalElements.backendBadge.textContent = "BACKEND OFFLINE";
  }
}

function renderHistory(track, versions) {
  track.history.replaceChildren();
  if (!versions.length) return track.history.append(node("div", "history-empty", "No accepted versions yet."));
  for (const version of versions) {
    const item = node("article", "history-item");
    item.append(node("strong", "", String(version.generation).padStart(3, "0")));
    const copy = node("div");
    copy.append(node("p", "", version.mode === "uploaded-reference" ? version.original_filename || "Imported reference" : version.prompt));
    copy.append(node("small", "", `${String(version.mode).toUpperCase()} · ${version.beats} BEATS · ${Number(version.loop_seconds).toFixed(1)}S`));
    const play = node("button", "", audio.playing ? "QUEUE" : "PLAY");
    play.type = "button";
    play.addEventListener("click", async () => {
      const entry = await loadEntry(track.id, version);
      if (audio.playing) track.pending = entry;
      else { track.active = entry; await startPlayback(); }
    });
    item.append(copy, play);
    track.history.append(item);
  }
}

function bindTrack(track) {
  track.addTag.addEventListener("click", () => addTag(track));
  track.tagInput.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); addTag(track); } });
  track.freeze.addEventListener("click", () => toggleFreeze(track));
  track.mute.addEventListener("click", () => { track.mute.classList.toggle("active"); queueSessionSave(); });
  track.solo.addEventListener("click", () => { track.solo.classList.toggle("active"); queueSessionSave(); });
  [track.volume, track.highpass, track.lowpass, track.reverbSend, track.delaySend].forEach((input) => input.addEventListener("input", () => queueSessionSave()));
  [track.modelSelect, track.beats, track.mutation, track.cfg, track.steps, track.mutationInterval, track.seedMode, track.seed, track.rolePrompt, track.negativePrompt].forEach((input) => {
    input.addEventListener("change", () => {
      track.dirty = true;
      track.seed.disabled = track.seedMode.value === "random";
      updateDuration(track);
      renderTags(track);
      void saveTrack(track);
      if (input === track.mutation) syncGlobalMutationFromTracks();
    });
  });
  track.flushHistory.addEventListener("click", () => flushTrackHistory(track));
  track.clearOutput.addEventListener("click", () => clearTrackOutput(track));
  track.referenceDrop.addEventListener("click", () => track.referenceInput.click());
  track.referenceDrop.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) { event.preventDefault(); track.referenceInput.click(); } });
  track.referenceInput.addEventListener("change", () => importReference(track, track.referenceInput.files[0]));
  track.referenceDrop.addEventListener("dragover", (event) => { event.preventDefault(); track.referenceDrop.classList.add("dragging"); });
  track.referenceDrop.addEventListener("dragleave", () => track.referenceDrop.classList.remove("dragging"));
  track.referenceDrop.addEventListener("drop", (event) => {
    event.preventDefault();
    track.referenceDrop.classList.remove("dragging");
    importReference(track, event.dataTransfer.files[0]);
  });
}

TRACK_IDS.forEach((id) => bindTrack(tracks[id]));
globalElements.playButton.addEventListener("click", startPlayback);
globalElements.evolve.addEventListener("click", toggleEvolution);
globalElements.freezeAll.addEventListener("click", toggleFreezeAll);
globalElements.bpm.addEventListener("change", () => queueSessionSave(true));
globalElements.globalMutation.addEventListener("input", () => setGlobalMutation(globalElements.globalMutation.value));
globalElements.master.addEventListener("input", () => queueSessionSave());
globalElements.refreshHistory.addEventListener("click", refreshHistory);
globalElements.flushAllHistory.addEventListener("click", flushAllHistory);
globalElements.clearAllOutputs.addEventListener("click", clearAllOutputs);
[
  globalElements.sidechainEnabled, globalElements.sidechainMelodic, globalElements.sidechainTexture,
  globalElements.scThreshold, globalElements.scRatio,
  globalElements.scAttack, globalElements.scRelease, globalElements.scDepth, globalElements.scMakeup,
  globalElements.reverbEnabled, globalElements.rvReturn, globalElements.rvDecay,
  globalElements.rvPredelay, globalElements.rvDamping,
  globalElements.delayEnabled, globalElements.delayDivision, globalElements.delayReturn,
  globalElements.delayFeedback, globalElements.delayLowpass, globalElements.delayWidth,
].forEach((input) => input.addEventListener("input", () => queueSessionSave()));
document.addEventListener("keydown", (event) => {
  if (event.code === "Space" && !/INPUT|SELECT|TEXTAREA|BUTTON/.test(document.activeElement.tagName)) {
    event.preventDefault();
    void startPlayback();
  }
});

updateControlOutputs();
void refreshStatus();
void refreshHistory();
void loadBackendInfo();
setInterval(refreshStatus, 400);
setInterval(refreshHistory, 12000);
