// popup.js — EthicalGuard popup script

const $toggle      = document.getElementById("toggle");
const $btnRescan   = document.getElementById("btn-rescan");
const $btnClear    = document.getElementById("btn-clear");
const $apiUrl      = document.getElementById("api-url");
const $status      = document.getElementById("status");
const $countHigh   = document.getElementById("count-high");
const $countMedium = document.getElementById("count-medium");
const $countSafe   = document.getElementById("count-safe");

// ── Init ────────────────────────────────────────────────────────────────────

chrome.storage.sync.get(["enabled", "apiUrl"], (data) => {
  $toggle.checked = data.enabled !== false;
  $apiUrl.value   = data.apiUrl  || "http://127.0.0.1:8000";
  updateStatus();
  fetchStats();
});

// ── Toggle ──────────────────────────────────────────────────────────────────

$toggle.addEventListener("change", () => {
  const enabled = $toggle.checked;
  chrome.runtime.sendMessage({ type: "TOGGLE", enabled });
  updateStatus();
});

// ── Buttons ─────────────────────────────────────────────────────────────────

$btnRescan.addEventListener("click", () => {
  $status.textContent = "Re-scanning page…";
  $status.className   = "status-bar scanning";
  chrome.runtime.sendMessage({ type: "RE_SCAN" });
  setTimeout(fetchStats, 2000);
});

$btnClear.addEventListener("click", () => {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      func:   clearHighlightsInPage,
    });
  });
  $countHigh.textContent   = "0";
  $countMedium.textContent = "0";
  $countSafe.textContent   = "—";
  $status.textContent = "Highlights cleared";
});

function clearHighlightsInPage() {
  document.querySelectorAll(".eg-flagged").forEach(mark => {
    const parent = mark.parentNode;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
  });
}

// ── API URL ─────────────────────────────────────────────────────────────────

$apiUrl.addEventListener("change", () => {
  const url = $apiUrl.value.trim();
  chrome.storage.sync.set({ apiUrl: url });
  chrome.runtime.sendMessage({ type: "API_URL_CHANGED", apiUrl: url });
  $status.textContent = "Backend URL updated";
  $status.className   = "status-bar";
});

// ── Stats ───────────────────────────────────────────────────────────────────

function fetchStats() {
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.scripting.executeScript({
      target: { tabId: tabs[0].id },
      func:   getStatsFromPage,
    }, (results) => {
      if (!results || !results[0]) return;
      const { high, medium, safe } = results[0].result || { high: 0, medium: 0, safe: 0 };
      $countHigh.textContent   = high;
      $countMedium.textContent = medium;
      $countSafe.textContent   = safe || "—";
    });
  });
}

function getStatsFromPage() {
  const marks = document.querySelectorAll(".eg-flagged");
  let high = 0, medium = 0;
  marks.forEach(m => {
    if (m.dataset.severity === "HIGH")   high++;
    if (m.dataset.severity === "MEDIUM") medium++;
  });
  return { high, medium, safe: marks.length - high - medium };
}

// ── Status ──────────────────────────────────────────────────────────────────

function updateStatus() {
  if ($toggle.checked) {
    $status.textContent = "Scanning for unsafe text…";
    $status.className   = "status-bar scanning";
  } else {
    $status.textContent = "Paused";
    $status.className   = "status-bar";
  }
}
