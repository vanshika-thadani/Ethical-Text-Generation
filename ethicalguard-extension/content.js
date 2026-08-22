/**
 * content.js — EthicalGuard browser extension content script
 *
 * What it does:
 *  1. Extracts visible sentence-level text nodes from the page
 *  2. Sends them to POST /analyze-chunks on the EthicalGuard backend
 *  3. Wraps flagged sentences in <mark class="eg-flagged"> with a red wavy underline
 *  4. Shows a tooltip on hover with the reason + safe rewrite
 */

"use strict";

// ── Config ──────────────────────────────────────────────────────────────────
const CHUNK_MIN_WORDS = 4;       // ignore fragments shorter than this
const CHUNK_MAX_CHARS = 400;     // don't send huge chunks
const BATCH_SIZE      = 20;      // sentences per API call
const SCAN_DELAY_MS   = 1500;    // wait after page load before scanning

// Tags whose text content we skip entirely
const SKIP_TAGS = new Set([
  "SCRIPT","STYLE","NOSCRIPT","CODE","PRE","TEXTAREA",
  "INPUT","BUTTON","SELECT","OPTION","LABEL","NAV",
  "HEADER","FOOTER","ASIDE","META","LINK","HEAD",
]);

// ── State ───────────────────────────────────────────────────────────────────
let _enabled  = true;
let _apiUrl   = "http://127.0.0.1:8000";
let _tooltip  = null;   // single shared tooltip element
let _scanning = false;

// ── Init ────────────────────────────────────────────────────────────────────
chrome.storage.sync.get(["enabled", "apiUrl"], (data) => {
  _enabled = data.enabled !== false;
  _apiUrl  = data.apiUrl  || "http://127.0.0.1:8000";
  if (_enabled) {
    setTimeout(scan, SCAN_DELAY_MS);
  }
});

// Listen for toggle / re-scan messages from popup / background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "TOGGLE") {
    _enabled = msg.enabled;
    if (!_enabled) {
      removeAllHighlights();
    } else {
      scan();
    }
  }
  if (msg.type === "RE_SCAN") {
    removeAllHighlights();
    scan();
  }
  if (msg.type === "API_URL_CHANGED") {
    _apiUrl = msg.apiUrl;
    removeAllHighlights();
    scan();
  }
});

// ── DOM helpers ─────────────────────────────────────────────────────────────

/**
 * Walk the DOM and collect visible text nodes that are direct children of
 * block-level elements (p, div, li, td, h1-h6, blockquote, article, section).
 * Returns [{node, text, id}] where id is a stable index string.
 */
function collectTextNodes() {
  const BLOCK = new Set([
    "P","DIV","LI","TD","TH","H1","H2","H3","H4","H5","H6",
    "BLOCKQUOTE","ARTICLE","SECTION","SPAN","FIGCAPTION","SUMMARY",
  ]);

  const results = [];
  let idx = 0;

  function walk(node) {
    if (node.nodeType === Node.ELEMENT_NODE) {
      if (SKIP_TAGS.has(node.tagName)) return;
      if (!isVisible(node)) return;
      for (const child of node.childNodes) walk(child);
    } else if (node.nodeType === Node.TEXT_NODE) {
      const parent = node.parentElement;
      if (!parent || SKIP_TAGS.has(parent.tagName)) return;
      if (!BLOCK.has(parent.tagName) && !BLOCK.has(parent.parentElement?.tagName)) return;

      const text = node.textContent.trim();
      const wordCount = text.split(/\s+/).filter(Boolean).length;
      if (wordCount < CHUNK_MIN_WORDS) return;
      if (text.length > CHUNK_MAX_CHARS) return;

      results.push({ node, text, id: `eg-${idx++}` });
    }
  }

  walk(document.body);
  return results;
}

function isVisible(el) {
  if (!el.getBoundingClientRect) return true;
  const style = window.getComputedStyle(el);
  return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
}

// ── Scanning ────────────────────────────────────────────────────────────────

async function scan() {
  if (_scanning || !_enabled) return;
  _scanning = true;

  try {
    const nodes = collectTextNodes();
    if (!nodes.length) return;

    // Process in batches to avoid huge single requests
    for (let i = 0; i < nodes.length; i += BATCH_SIZE) {
      const batch = nodes.slice(i, i + BATCH_SIZE);
      await processBatch(batch);
    }
  } catch (e) {
    console.warn("[EthicalGuard] Scan failed:", e);
  } finally {
    _scanning = false;
  }
}

async function processBatch(nodes) {
  const payload = {
    chunks: nodes.map(n => ({ id: n.id, text: n.text })),
  };

  let data;
  try {
    const res = await fetch(`${_apiUrl}/analyze-chunks`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    if (!res.ok) return;
    data = await res.json();
  } catch (e) {
    console.warn("[EthicalGuard] API call failed:", e);
    return;
  }

  // Build a quick lookup: id → result
  const byId = {};
  for (const r of data.results || []) {
    byId[r.id] = r;
  }

  // Apply highlights to flagged nodes
  for (const n of nodes) {
    const result = byId[n.id];
    if (!result || result.severity === "LOW") continue;

    // Guard: node must still be in the DOM and not already wrapped
    if (!document.contains(n.node)) continue;
    if (n.node.parentElement?.classList.contains("eg-flagged")) continue;

    wrapNode(n.node, result);
  }
}

// ── Wrapping ─────────────────────────────────────────────────────────────────

function wrapNode(textNode, result) {
  const mark = document.createElement("mark");
  mark.className = "eg-flagged";
  mark.dataset.severity  = result.severity;
  mark.dataset.reason    = result.reason    || "unsafe content";
  mark.dataset.rewritten = result.rewritten || "";
  mark.dataset.egId      = result.id;

  // Replace the text node with the mark
  const parent = textNode.parentNode;
  parent.insertBefore(mark, textNode);
  mark.appendChild(textNode);

  // Tooltip events
  mark.addEventListener("mouseenter", showTooltip);
  mark.addEventListener("mouseleave", scheduleHideTooltip);
}

// ── Tooltip ──────────────────────────────────────────────────────────────────

let _hideTimer = null;

function getOrCreateTooltip() {
  if (!_tooltip) {
    _tooltip = document.createElement("div");
    _tooltip.className = "eg-tooltip";
    _tooltip.style.display = "none";
    _tooltip.addEventListener("mouseenter", () => clearTimeout(_hideTimer));
    _tooltip.addEventListener("mouseleave", scheduleHideTooltip);
    document.body.appendChild(_tooltip);
  }
  return _tooltip;
}

function showTooltip(e) {
  clearTimeout(_hideTimer);

  const mark     = e.currentTarget;
  const severity = mark.dataset.severity;
  const reason   = mark.dataset.reason   || "unsafe content";
  const rewrite  = mark.dataset.rewritten;

  const tip = getOrCreateTooltip();

  const badgeClass = severity === "HIGH" ? "eg-badge-high" : "eg-badge-medium";
  const icon       = severity === "HIGH" ? "⚠️" : "🔶";

  tip.innerHTML = `
    <div class="eg-tooltip-header">
      <span class="${badgeClass}">${icon} ${severity} RISK</span>
    </div>
    <div class="eg-tooltip-reason">Detected: ${reason}</div>
    ${rewrite ? `
      <hr class="eg-tooltip-divider" />
      <div class="eg-tooltip-label">✦ Suggested replacement</div>
      <div class="eg-tooltip-rewrite" id="eg-rewrite-text">${escapeHtml(rewrite)}</div>
      <button class="eg-copy-btn" id="eg-copy-btn">Copy replacement</button>
    ` : `
      <hr class="eg-tooltip-divider" />
      <div class="eg-tooltip-loading">Generating safe replacement…</div>
    `}
  `;

  // Position tooltip just above the mark
  const rect = mark.getBoundingClientRect();
  const scrollY = window.scrollY;
  const scrollX = window.scrollX;

  tip.style.display = "block";
  tip.style.left    = `${scrollX + rect.left}px`;
  tip.style.top     = `${scrollY + rect.top - tip.offsetHeight - 10}px`;

  // If tooltip goes off-screen top, show below instead
  if (rect.top - tip.offsetHeight - 10 < 0) {
    tip.style.top = `${scrollY + rect.bottom + 8}px`;
  }

  // Copy button handler
  const copyBtn = document.getElementById("eg-copy-btn");
  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      navigator.clipboard.writeText(rewrite).then(() => {
        copyBtn.textContent = "Copied!";
        setTimeout(() => { copyBtn.textContent = "Copy replacement"; }, 1500);
      });
    });
  }
}

function scheduleHideTooltip() {
  _hideTimer = setTimeout(() => {
    if (_tooltip) _tooltip.style.display = "none";
  }, 200);
}

// ── Cleanup ──────────────────────────────────────────────────────────────────

function removeAllHighlights() {
  document.querySelectorAll(".eg-flagged").forEach(mark => {
    const parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
  });
  if (_tooltip) _tooltip.style.display = "none";
}

// ── Utils ─────────────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
