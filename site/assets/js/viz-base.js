/**
 * Shared visualization helpers — the 4-pane pattern.
 * Each visualization page imports this as a classic script.
 *
 * Pane 1: Visualization
 * Pane 2: Abstract equation (formula)
 * Pane 3: Substituted equation (live values plugged into the formula)
 * Pane 4: Outcome
 */

(function (global) {
  'use strict';

  const palette = ['--c-1', '--c-2', '--c-3', '--c-4', '--c-5', '--c-6'];

  /* Theme toggle (mirrors reader.js) */
  function initTheme() {
    const stored = localStorage.getItem('theme');
    const initial = stored || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    apply(initial);
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      apply(next);
      localStorage.setItem('theme', next);
      window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: next } }));
    });
    function apply(name) {
      document.documentElement.dataset.theme = name;
      const b = document.getElementById('theme-toggle');
      if (b) b.textContent = name === 'dark' ? '☀ Light' : '☾ Dark';
    }
  }

  /* Render KaTeX into an element, fail-soft on errors. */
  function katexRender(el, latex, opts = {}) {
    if (!el) return;
    /* If a `.pane-content` child exists (the new 4-pane layout uses it),
       render into that so the styled flex container is preserved. */
    const target = el.querySelector('.pane-content') || el;
    try {
      // eslint-disable-next-line no-undef
      katex.render(latex, target, { throwOnError: false, displayMode: opts.display !== false, ...opts });
    } catch {
      target.textContent = latex;
    }
  }

  /* Render the abstract formula into pane 2 and the substituted version
     into pane 3 — the two are always updated together. */
  function renderEquations(abstractLatex, substitutedLatex) {
    const eq = document.getElementById('equation-pane');
    const sub = document.getElementById('substituted-pane');
    if (eq) katexRender(eq, abstractLatex);
    if (sub) katexRender(sub, substitutedLatex);
  }

  function color(token) {
    return getComputedStyle(document.documentElement).getPropertyValue(token).trim() || token;
  }
  function paletteColor(i) { return color(palette[i % palette.length]); }

  function debounce(fn, ms) {
    let t;
    return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
  }

  /* Output-pane row builder. */
  function outputRow(label, value, accent = false) {
    const div = document.createElement('div');
    div.className = 'viz-output-row';
    const l = document.createElement('span');
    l.className = 'viz-output-row__label';
    l.textContent = label;
    const v = document.createElement('span');
    v.className = 'viz-output-row__value' + (accent ? ' viz-output-row__value--accent' : '');
    v.textContent = value;
    div.appendChild(l);
    div.appendChild(v);
    return div;
  }
  function setOutput(el, rows) {
    if (!el) return;
    el.innerHTML = '';
    rows.forEach((r) => el.appendChild(outputRow(r.label, r.value, r.accent)));
  }

  function fmt(n, digits = 3) {
    if (!Number.isFinite(n)) return String(n);
    if (Math.abs(n) >= 1000) return n.toFixed(0);
    return n.toFixed(digits);
  }

  /* Format a numeric vector as a LaTeX-friendly bracketed list. */
  function vec(arr, digits = 3, max = 6) {
    const more = arr.length > max ? ',\\;\\dots' : '';
    return '[' + arr.slice(0, max).map((v) => fmt(v, digits)).join(',\\;') + more + ']';
  }

  /* Mulberry32 RNG with seed. */
  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6D2B79F5) >>> 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function softmax(arr, T = 1) {
    const scaled = arr.map((x) => x / T);
    const max = Math.max(...scaled);
    const exps = scaled.map((x) => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }
  function cumsum(arr) {
    const out = new Array(arr.length);
    let s = 0;
    for (let i = 0; i < arr.length; i++) { s += arr[i]; out[i] = s; }
    return out;
  }

  /* Live slider fill — paints the filled portion in CSS variable `--p`. */
  function paintSliderFill(slider) {
    const min = Number(slider.min || 0);
    const max = Number(slider.max || 100);
    const val = Number(slider.value);
    const pct = ((val - min) / (max - min)) * 100;
    slider.style.setProperty('--p', pct.toFixed(2) + '%');
  }
  function bindSliderFills() {
    document.querySelectorAll('input[type="range"]').forEach((s) => {
      paintSliderFill(s);
      s.addEventListener('input', () => paintSliderFill(s));
    });
  }

  /* Replace bare pane content placeholders with `.pane-content` wrappers
     so CSS can flex them, and label them consistently.
     Backward-compatible: if a page already wraps its panes, this is a no-op. */
  function decoratePanes() {
    const labels = {
      'viz-pane':         { idx: '1', text: 'Visualization' },
      'equation-pane':    { idx: '2', text: 'Equation · the formula' },
      'substituted-pane': { idx: '3', text: 'Right now · values plugged in' },
      'output-pane':      { idx: '4', text: 'Outcome' },
    };
    Object.entries(labels).forEach(([id, info]) => {
      const el = document.getElementById(id);
      if (!el) return;
      /* Ensure the pane has a label of the right shape. */
      const card = el.closest('.viz-pane');
      if (card && !card.querySelector('.viz-pane__label--auto')) {
        const existing = card.querySelector('.viz-pane__label');
        if (existing) existing.classList.add('viz-pane__label--auto');
      }
      /* Wrap eq / sub content in a flex container if not already. */
      if ((id === 'equation-pane' || id === 'substituted-pane') &&
          !el.querySelector('.pane-content')) {
        const inner = document.createElement('div');
        inner.className = 'pane-content';
        while (el.firstChild) inner.appendChild(el.firstChild);
        el.appendChild(inner);
      }
    });
  }

  function bindSlider(id, valueId, format = (v) => v) {
    const slider = document.getElementById(id);
    const value = valueId ? document.getElementById(valueId) : null;
    function update() { if (value) value.textContent = format(slider.value); paintSliderFill(slider); }
    if (slider) { slider.addEventListener('input', update); update(); }
    return slider;
  }

  function boot(callback) {
    document.addEventListener('DOMContentLoaded', () => {
      initTheme();
      decoratePanes();
      bindSliderFills();
      try { callback(); } catch (e) {
        console.error(e);
        const out = document.getElementById('output-pane');
        if (out) out.textContent = `Error: ${e.message}`;
      }
    });
  }

  global.Viz = {
    katexRender, renderEquations, color, paletteColor, debounce,
    setOutput, outputRow, fmt, vec, rng, softmax, cumsum,
    bindSlider, paintSliderFill, boot,
  };
})(window);
