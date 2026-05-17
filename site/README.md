# LLM Engineering Bible — Web Site

A complete web reader for `LLM_COMPLETE_GUIDE.md` plus an interactive visualization gallery covering every major LLM concept in the guide.

## Structure

```
site/
├── index.html              # Main reader (loads markdown via fetch)
├── standalone.html         # Self-contained reader (build via build.py)
├── visuals/
│   ├── index.html          # Visualization gallery
│   ├── attention.html
│   ├── softmax-temperature.html
│   ├── gradient-descent.html
│   ├── tokenization-bpe.html
│   ├── embedding-similarity.html
│   ├── rag-pipeline.html
│   ├── top-p-sampling.html
│   ├── layer-norm.html
│   ├── lora.html
│   ├── positional-encoding.html
│   ├── moe-routing.html
│   └── beam-search.html
├── assets/
│   ├── css/
│   │   ├── reader.css
│   │   └── visuals.css
│   └── js/
│       ├── reader.js
│       └── viz-base.js
└── build.py                # Generates standalone.html
```

## Running locally

Browsers block `fetch()` from `file://` for security. To view the site:

### Option 1 — local web server (recommended)

```powershell
# From the llm-guide root (one level above site/)
python -m http.server 8000
# then open http://localhost:8000/site/
```

### Option 2 — self-contained file

```powershell
cd site
python build.py
# opens standalone.html — works by double-click, no server needed
```

## Visualization gallery — the 3-pane pattern

Every visualization follows the same anatomy:

```
┌───────────────────────────────┐
│  Pane 1: Dynamic visual       │  ← graph / heatmap / animation
├───────────────────────────────┤
│  Pane 2: Live equation        │  ← KaTeX-rendered math
├───────────────────────────────┤
│  Pane 3: Output / values      │  ← current values, sampled token, loss…
└───────────────────────────────┘
```

Controls (sliders, buttons, inputs) sit in a bar above pane 1 and update all three panes in real time.

## Adding a new visualization

1. Copy any file from `visuals/` as a template.
2. Implement three functions in your `<script type="module">`:
   - `drawVisual(state)` — renders into `#viz-pane`
   - `renderEquation(state)` — updates KaTeX in `#equation-pane`
   - `renderOutput(state)` — updates `#output-pane`
3. Wire your controls to a single `update()` function that calls all three.
4. Add a card to `visuals/index.html`.

The shared module `assets/js/viz-base.js` exposes:

- `mountViz({ title, eqInit })` — boots the 3-pane shell
- `katexRender(el, latex)` — wrapper around KaTeX with error handling
- `colors` — duotone palette tokens
- `debounce(fn, ms)` — for heavy redraws

## Theme

Duotone with a one-click light/dark toggle (top-right of every page). The
palette lives in CSS custom properties on `:root` and `[data-theme="dark"]`
in `assets/css/reader.css`.

## Dependencies (all CDN)

| Library | Purpose |
|---|---|
| [marked](https://marked.js.org) | Markdown → HTML |
| [KaTeX](https://katex.org) | Math rendering |
| [Prism](https://prismjs.com) | Code highlighting |
| [D3](https://d3js.org) | SVG-based visualizations |

No build step is required for the reader. Only `build.py` (Python stdlib) is
needed to produce the standalone bundle.
