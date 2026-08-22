// background.js — EthicalGuard service worker
// Relays messages between popup and content script, manages enabled state.

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({ enabled: true, apiUrl: "http://127.0.0.1:8000" });
});

// Forward toggle messages from popup to the active tab's content script
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "TOGGLE") {
    chrome.storage.sync.set({ enabled: msg.enabled });
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, msg);
      }
    });
    sendResponse({ ok: true });
  }
  if (msg.type === "RE_SCAN") {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]?.id) {
        chrome.tabs.sendMessage(tabs[0].id, msg);
      }
    });
    sendResponse({ ok: true });
  }
  return true;
});
