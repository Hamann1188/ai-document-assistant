"use strict";

// --- helpers ----------------------------------------------------------------

const $ = (selector) => document.querySelector(selector);

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

function truncate(text, max) {
  return text.length > max ? text.slice(0, max - 1).trimEnd() + "…" : text;
}

function pdfLink(documentId, page) {
  return `/documents/${encodeURIComponent(documentId)}/file#page=${page}`;
}

async function errorMessage(response) {
  try {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  } catch { /* not JSON */ }
  return `Request failed (HTTP ${response.status}).`;
}

// Minimal Markdown for answers: paragraphs, line breaks, bullet and numbered lists,
// **bold** and *italic*. Input must already be HTML-escaped.
function renderMarkdown(src) {
  const inline = (s) => s
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  let html = "";
  let list = null;
  let para = [];
  const flushPara = () => {
    if (para.length) html += `<p>${inline(para.join("<br>"))}</p>`;
    para = [];
  };
  const closeList = () => {
    if (list) html += `</${list}>`;
    list = null;
  };
  for (const line of src.split("\n")) {
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const item = bullet || numbered;
    if (item) {
      flushPara();
      const kind = bullet ? "ul" : "ol";
      if (list !== kind) { closeList(); html += `<${kind}>`; list = kind; }
      html += `<li>${inline(item[1])}</li>`;
    } else if (!line.trim()) {
      flushPara();
      closeList();
    } else {
      closeList();
      para.push(line);
    }
  }
  flushPara();
  closeList();
  return html;
}

// Server-sent events from a fetch() response (EventSource can't POST).
async function* readEvents(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      yield { event, data: data ? JSON.parse(data) : {} };
    }
  }
}

// --- documents panel --------------------------------------------------------

const docList = $("#doc-list");
const docEmpty = $("#doc-empty");
const uploadError = $("#upload-error");
let pollTimer = null;

function renderDocuments(documents) {
  docEmpty.hidden = documents.length > 0;
  docList.innerHTML = documents.map((doc) => {
    const name = escapeHtml(doc.filename);
    let status;
    if (doc.status === "ready") {
      status = `<span class="badge ready">ready</span> ${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`;
    } else if (doc.status === "processing") {
      status = `<span class="spinner" aria-hidden="true"></span> processing…`;
    } else {
      status = `<span class="badge failed">failed</span>`;
    }
    const title = doc.title ? ` title="${escapeHtml(doc.title)}"` : "";
    const nameHtml = doc.status === "ready"
      ? `<a class="doc-name" href="${pdfLink(doc.id, 1)}" target="_blank" rel="noopener"${title}>${name}</a>`
      : `<span class="doc-name"${title}>${name}</span>`;
    const error = doc.status === "failed" && doc.error
      ? `<div class="doc-error">${escapeHtml(doc.error)}</div>` : "";
    return `<li class="doc">
      ${nameHtml}
      <button type="button" class="doc-delete" data-id="${escapeHtml(doc.id)}"
              data-name="${name}" aria-label="Delete ${name}" title="Delete">×</button>
      <div class="doc-meta">${status}</div>
      ${error}
    </li>`;
  }).join("");
}

async function refreshDocuments() {
  clearTimeout(pollTimer);
  try {
    const response = await fetch("/documents");
    if (!response.ok) throw new Error(await errorMessage(response));
    const documents = await response.json();
    renderDocuments(documents);
    if (documents.some((d) => d.status === "processing")) {
      pollTimer = setTimeout(refreshDocuments, 1500);
    }
  } catch (err) {
    showUploadError(`Couldn't load documents: ${err.message}`);
  }
}

function showUploadError(message) {
  uploadError.textContent = message;
  uploadError.hidden = !message;
}

async function uploadFiles(files) {
  showUploadError("");
  const problems = [];
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    try {
      const response = await fetch("/documents", { method: "POST", body: form });
      if (!response.ok) problems.push(`${file.name}: ${await errorMessage(response)}`);
    } catch (err) {
      problems.push(`${file.name}: ${err.message}`);
    }
    refreshDocuments();
  }
  showUploadError(problems.join(" "));
}

$("#file-input").addEventListener("change", (event) => {
  uploadFiles([...event.target.files]);
  event.target.value = "";
});

const dropzone = $("#dropzone");
for (const type of ["dragenter", "dragover"]) {
  document.addEventListener(type, (event) => {
    event.preventDefault();
    dropzone.classList.add("over");
  });
}
for (const type of ["dragleave", "drop"]) {
  document.addEventListener(type, (event) => {
    event.preventDefault();
    if (type === "drop" || !event.relatedTarget) dropzone.classList.remove("over");
  });
}
document.addEventListener("drop", (event) => {
  const files = [...event.dataTransfer.files];
  if (files.length) uploadFiles(files);
});

docList.addEventListener("click", async (event) => {
  const button = event.target.closest(".doc-delete");
  if (!button) return;
  if (!confirm(`Delete "${button.dataset.name}"? Its answers can no longer cite it.`)) return;
  const response = await fetch(`/documents/${encodeURIComponent(button.dataset.id)}`, { method: "DELETE" });
  if (!response.ok && response.status !== 404) showUploadError(await errorMessage(response));
  refreshDocuments();
});

// --- chat -------------------------------------------------------------------

const thread = $("#thread");
const form = $("#ask-form");
const input = $("#question");
const sendButton = $("#send");
let current = null; // the answer being streamed

const CITE_OPEN = "";
const CITE_CLOSE = "";

class AnswerView {
  constructor(question) {
    this.blocks = new Map(); // content block index -> {text, numbers}
    this.numbers = new Map(); // source index -> citation number
    this.cited = []; // [{number, source, quotes}]
    this.context = []; // all pages sent to Claude
    this.renderQueued = false;

    this.el = document.createElement("article");
    this.el.className = "turn";
    this.el.innerHTML = `
      <div class="q">${escapeHtml(question)}</div>
      <div class="a">
        <div class="a-body typing"></div>
        <div class="sources" hidden><h3>Sources</h3><ol></ol></div>
        <div class="notice"></div>
        <div class="meta"></div>
      </div>`;
    this.body = this.el.querySelector(".a-body");
    this.sourcesEl = this.el.querySelector(".sources");
    this.noticeEl = this.el.querySelector(".notice");
    this.metaEl = this.el.querySelector(".meta");
    thread.append(this.el);
    this.el.scrollIntoView({ block: "end" });
  }

  block(index) {
    if (!this.blocks.has(index)) this.blocks.set(index, { text: "", numbers: [] });
    return this.blocks.get(index);
  }

  handle({ event, data }) {
    switch (event) {
      case "sources":
        this.context = data.sources;
        break;
      case "text":
        this.block(data.block).text += data.text;
        break;
      case "citation":
        this.addCitation(data);
        break;
      case "reset":
        this.blocks.clear();
        this.numbers.clear();
        this.cited = [];
        break;
      case "done":
        this.finish(data);
        break;
      case "error":
        this.notice(data.message, "error");
        break;
    }
    this.queueRender();
  }

  addCitation(c) {
    if (!this.numbers.has(c.source)) {
      this.numbers.set(c.source, this.numbers.size + 1);
      this.cited.push({ number: this.numbers.size, source: c, quotes: [] });
    }
    const number = this.numbers.get(c.source);
    const entry = this.cited[number - 1];
    if (!entry.quotes.includes(c.cited_text)) entry.quotes.push(c.cited_text);
    const block = this.block(c.block);
    if (!block.numbers.includes(number)) block.numbers.push(number);
  }

  finish({ stop_reason: stopReason, model, usage, cost_usd: cost }) {
    if (stopReason === "refusal") {
      this.notice("The assistant declined to answer this question. Try rephrasing it.", "warn");
    } else if (stopReason === "max_tokens") {
      this.notice("The answer was cut off because it reached the length limit.", "warn");
    }
    const parts = [model];
    if (usage) parts.push(`${(usage.input_tokens / 1000).toFixed(1)}k in · ${usage.output_tokens} out`);
    if (cost != null) parts.push(`$${cost.toFixed(3)}`);
    parts.push(`${this.context.length} page${this.context.length === 1 ? "" : "s"} searched`);
    this.metaEl.textContent = parts.join(" · ");
  }

  notice(message, kind) {
    this.noticeEl.innerHTML = `<p class="alert ${kind === "warn" ? "warn" : ""}">${escapeHtml(message)}</p>`;
  }

  queueRender() {
    if (this.renderQueued) return;
    this.renderQueued = true;
    requestAnimationFrame(() => {
      this.renderQueued = false;
      this.render();
    });
  }

  render() {
    const markers = (numbers) => numbers.length
      ? CITE_OPEN + numbers.join(",") + CITE_CLOSE : "";
    const text = [...this.blocks.entries()]
      .sort(([a], [b]) => a - b)
      .map(([, block]) => block.text + markers(block.numbers))
      .join("")
      // A quote the server dropped as a restatement leaves only its marker, after the
      // previous sentence's trailing space: pull the marker up to the punctuation.
      .replace(new RegExp(`(\\s+)(${CITE_OPEN}[\\d,]+${CITE_CLOSE})`, "g"), "$2$1");
    const html = renderMarkdown(escapeHtml(text)).replace(
      new RegExp(`${CITE_OPEN}([\\d,]+)${CITE_CLOSE}`, "g"),
      (_, list) => list.split(",").map((n) => this.citeLink(Number(n))).join(""),
    );
    this.body.innerHTML = html;

    this.sourcesEl.hidden = this.cited.length === 0;
    this.sourcesEl.querySelector("ol").innerHTML = this.cited.map(({ number, source, quotes }) => `
      <li class="source">
        <a href="${pdfLink(source.document_id, source.page)}" target="_blank" rel="noopener">
          [${number}] ${escapeHtml(source.filename)} · p. ${source.page}</a>
        ${quotes.map((q) => `<blockquote>${escapeHtml(truncate(q, 240))}</blockquote>`).join("")}
      </li>`).join("");

    const nearBottom = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 120;
    if (nearBottom) thread.scrollTop = thread.scrollHeight;
  }

  citeLink(number) {
    const entry = this.cited[number - 1];
    if (!entry) return "";
    const { source, quotes } = entry;
    const tip = `${source.filename}, p. ${source.page}: ${truncate(quotes.join(" "), 300)}`;
    return `<a class="cite" href="${pdfLink(source.document_id, source.page)}" target="_blank"
      rel="noopener" title="${escapeHtml(tip)}">${number}</a>`;
  }

  done() {
    this.body.classList.remove("typing");
    if (!this.body.textContent.trim() && !this.noticeEl.textContent) {
      this.notice("No answer was returned.", "error");
    }
  }
}

function setBusy(busy) {
  sendButton.disabled = busy;
  sendButton.textContent = busy ? "Answering…" : "Ask";
  input.disabled = busy;
}

async function ask(question) {
  $("#empty")?.remove();
  const view = new AnswerView(question);
  current = view;
  setBusy(true);
  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      view.handle({ event: "error", data: { message: await errorMessage(response) } });
    } else {
      for await (const event of readEvents(response)) view.handle(event);
    }
  } catch (err) {
    view.handle({ event: "error", data: { message: `Connection problem: ${err.message}` } });
  } finally {
    view.render();
    view.done();
    current = null;
    setBusy(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || current) return;
  input.value = "";
  autosize();
  ask(question);
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

function autosize() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}
input.addEventListener("input", autosize);

$("#examples")?.addEventListener("click", (event) => {
  const chip = event.target.closest(".chip");
  if (!chip || current) return;
  ask(chip.textContent.trim());
});

refreshDocuments();
input.focus();
