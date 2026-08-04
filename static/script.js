(() => {
  const feed = document.getElementById("chat-feed");
  const input = document.getElementById("query-input");
  const sendBtn = document.getElementById("send-btn");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadInput = document.getElementById("upload-input");
  const howBtn = document.getElementById("how-it-works-btn");
  const modal = document.getElementById("how-it-works-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");

  function scrollToBottom() {
    feed.scrollTop = feed.scrollHeight;
  }

  function addMessage(text, cls) {
    const el = document.createElement("div");
    el.className = `msg ${cls}`;
    el.textContent = text;
    feed.appendChild(el);
    scrollToBottom();
    return el;
  }

  function addSystem(text) {
    return addMessage(text, "system");
  }

  function addBotAnswer(answer, confidence, elapsed) {
    const el = addMessage(answer, "bot");
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `Confidence: ${confidence.toFixed(2)} | Time: ${elapsed.toFixed(2)}s`;
    el.appendChild(meta);
    return el;
  }

  function addPlanCard(plan) {
    const el = document.createElement("div");
    el.className = "plan-card";
    el.innerHTML = `
      <strong>Query Plan</strong>
      <dl>
        <dt>Category</dt><dd>${escapeHtml(plan.category ?? "")}</dd>
        <dt>Entities</dt><dd>${escapeHtml((plan.entities || []).join(", ") || "-")}</dd>
        <dt>Agentic</dt><dd>${plan.requires_agentic_loop ? "yes" : "no"}</dd>
        <dt>Top K</dt><dd>${escapeHtml(String(plan.top_k ?? ""))}</dd>
      </dl>`;
    feed.appendChild(el);
    scrollToBottom();
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function fmt(v) {
    return typeof v === "number" ? v.toFixed(2) : "-";
  }

  function addCompareCard(data) {
    const wrap = document.createElement("div");
    wrap.className = "compare-card";
    const order = ["Knowledge Graph", "BM25", "Semantic (Vector)"];
    let rows = "";
    for (const name of order) {
      const r = data.results[name];
      if (!r) continue;
      if (r.is_error) {
        rows += `<tr class="errored"><td>${name}</td><td colspan="6">ERROR: ${escapeHtml(r.answer)}</td></tr>`;
        continue;
      }
      const isWinner = name === data.winner;
      rows += `<tr class="${isWinner ? "winner" : ""}">
        <td>${name}${isWinner ? " \u{1F3C6}" : ""}</td>
        <td>${fmt(r.retrieval_score)}</td>
        <td>${fmt(r.groundedness_score)}</td>
        <td>${fmt(r.answer_quality)}</td>
        <td>${fmt(r.consensus_score)}</td>
        <td>${fmt(r.final_score)}</td>
        <td>${fmt(r.time)}</td>
      </tr>`;
    }
    wrap.innerHTML = `
      <strong>Comparison</strong>
      <div class="compare-table-wrap">
        <table class="compare-table">
          <thead><tr><th>Model</th><th>Retrieval</th><th>Groundedness</th><th>Answer Qual.</th><th>Consensus</th><th>Final Score</th><th>Time (s)</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>`;
    feed.appendChild(wrap);

    if (data.winner) {
      addBotAnswer(
        `Answer (via ${data.winner}): ${data.results[data.winner].answer}`,
        data.results[data.winner].answer_quality,
        data.results[data.winner].time
      );
    } else {
      addMessage("All sources failed to generate an answer.", "bot error");
    }
    scrollToBottom();
  }

  async function postJson(url, body) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
    return data;
  }

  function setBusy(busy) {
    sendBtn.disabled = busy;
    input.disabled = busy;
  }

  async function handleSubmit() {
    const raw = input.value.trim();
    if (!raw) return;
    input.value = "";
    addMessage(raw, "user");

    if (/^(q|quit|exit)$/i.test(raw)) {
      addSystem("This is a web session - just close the tab (or refresh) to leave. There's no process to exit.");
      return;
    }

    setBusy(true);
    try {
      if (raw.startsWith("/ingest")) {
        await postJson("/api/ingest");
        addSystem("Ingestion started in the background - watch for the 'Reindex complete' message.");
        pollReindex();
      } else if (raw.startsWith("/scrape")) {
        const url = raw.slice("/scrape".length).trim();
        if (!url) { addSystem("Usage: /scrape <URL>"); return; }
        addSystem(`Scraping PDFs from ${url}...`);
        const data = await postJson("/api/scrape", { url });
        addSystem(`Saved ${data.saved_files.length} file(s).` + (data.reindexing ? " Reindexing in the background..." : ""));
        if (data.reindexing) pollReindex();
      } else if (raw.startsWith("/plan")) {
        const query = raw.slice("/plan".length).trim();
        if (!query) { addSystem("Usage: /plan <query>"); return; }
        const data = await postJson("/api/plan", { query });
        addPlanCard(data.plan);
        addBotAnswer(data.answer, data.confidence, data.elapsed);
      } else if (raw.startsWith("/compare")) {
        const query = raw.slice("/compare".length).trim();
        if (!query) { addSystem("Usage: /compare <query>"); return; }
        addSystem("Running /compare across Knowledge Graph, BM25 and Semantic Search...");
        const data = await postJson("/api/compare", { query });
        addCompareCard(data);
      } else {
        const data = await postJson("/api/query", { query: raw });
        addBotAnswer(data.answer, data.confidence, data.elapsed);
      }
    } catch (err) {
      addMessage(String(err.message || err), "bot error");
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  let pollTimer = null;
  function pollReindex() {
    if (pollTimer) return;
    pollTimer = setInterval(async () => {
      try {
        const res = await fetch("/api/reindex-status");
        const data = await res.json();
        if (data.status !== "running") {
          clearInterval(pollTimer);
          pollTimer = null;
          addSystem(`Reindex complete: ${data.detail}`);
        }
      } catch {
        clearInterval(pollTimer);
        pollTimer = null;
      }
    }, 2000);
  }

  sendBtn.addEventListener("click", handleSubmit);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") handleSubmit();
  });

  uploadBtn.addEventListener("click", () => uploadInput.click());
  uploadInput.addEventListener("change", async () => {
    const files = uploadInput.files;
    if (!files || !files.length) return;
    const names = Array.from(files).map((f) => f.name).join(", ");
    addMessage(`Uploading: ${names}`, "user");

    const formData = new FormData();
    for (const f of files) formData.append("files", f);

    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.error || `Upload failed (${res.status})`);
      addSystem(`Saved ${data.saved_files.length} file(s).` + (data.rejected_files.length ? ` Rejected: ${data.rejected_files.join(", ")}.` : "") + (data.reindexing ? " Indexing..." : ""));
      if (data.reindexing) pollReindex();
    } catch (err) {
      addMessage(String(err.message || err), "bot error");
    } finally {
      uploadInput.value = "";
    }
  });

  howBtn.addEventListener("click", () => { modal.hidden = false; });
  modalCloseBtn.addEventListener("click", () => { modal.hidden = true; });
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.hidden = true; });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.hidden = true; });

  addSystem("Unified Multi-RAG Pipeline ready. Type a question, or try /compare, /plan, /ingest, /scrape <URL>.");
})();
