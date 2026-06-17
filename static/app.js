const statusPill = document.querySelector("#status-pill");
const messagesEl = document.querySelector("#messages");
const searchStatus = document.querySelector("#search-status");
const chatForm = document.querySelector("#chat-form");
const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-button");

const starterPrompts = [
  "What is your return policy?",
  "My order is delayed. What should I do?",
  "How does warranty support work?",
  "Can you help me troubleshoot a product issue?",
];

let history = [];
let initialized = false;

init();

async function init() {
  addAssistantMessage(
    "Hi, I'm your store support assistant. I can help with returns, shipping, warranties, product troubleshooting, and store policies. What can I help you with today?",
    true
  );
  await refreshStatus();
  window.setInterval(refreshStatus, 5000);
}

async function refreshStatus() {
  try {
    const status = await getJson("/api/status");
    updateStatus(status);
  } catch {
    initialized = false;
    statusPill.textContent = "Offline";
    statusPill.className = "status-pill unavailable";
    setChatEnabled(false);
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = chatInput.value.trim();
  if (!message || !initialized) {
    return;
  }

  chatInput.value = "";
  addMessage("user", message);
  const assistantMessage = addAssistantMessage("");
  setChatEnabled(false);
  showSearchStatus("Checking store support documents...");

  try {
    const data = await postStream("/api/chat/stream", {
      message,
      history,
    }, ({ type, payload }) => {
      if (type === "search") {
        showSearchStatus(`Checking: ${payload.query}`);
      }

      if (type === "text") {
        assistantMessage.textContent += payload;
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    });

    if (data.searches && data.searches.length) {
      showSearchStatus(`Sources checked: ${data.searches.join("; ")}`);
    } else {
      hideSearchStatus();
    }

    history.push({ role: "user", content: message });
    history.push({ role: "assistant", content: data.answer });
  } catch (error) {
    assistantMessage.textContent =
      "I'm sorry, I couldn't answer that right now. Please try again in a moment.";
    showSearchStatus(error.message);
  } finally {
    setChatEnabled(initialized);
    chatInput.focus();
  }
});

function updateStatus(status) {
  initialized = Boolean(status.initialized);

  if (status.indexing) {
    statusPill.textContent = "Updating";
    statusPill.className = "status-pill updating";
  } else if (initialized) {
    statusPill.textContent = "Online";
    statusPill.className = "status-pill ready";
  } else {
    statusPill.textContent = "Preparing";
    statusPill.className = "status-pill unavailable";
  }

  if (status.error && !initialized) {
    showSearchStatus(status.error);
  } else if (status.indexing) {
    showSearchStatus("Knowledge base is updating. Answers may take a moment.");
  } else if (searchStatus.textContent.startsWith("Knowledge base")) {
    hideSearchStatus();
  }

  setChatEnabled(initialized);
}

function addAssistantMessage(content, showPrompts = false) {
  const message = addMessage("assistant", content);
  if (showPrompts) {
    const promptList = document.createElement("div");
    promptList.className = "starter-prompts";

    for (const prompt of starterPrompts) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "prompt-chip";
      button.textContent = prompt;
      button.addEventListener("click", () => {
        chatInput.value = prompt;
        chatInput.focus();
      });
      promptList.append(button);
    }

    message.append(promptList);
  }
  return message;
}

function addMessage(role, content) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = content;
  messagesEl.append(message);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return message;
}

function setChatEnabled(isEnabled) {
  chatInput.disabled = !isEnabled;
  sendButton.disabled = !isEnabled;
  chatInput.placeholder = isEnabled
    ? "Ask about returns, shipping, warranty, or product help"
    : "Support assistant is preparing";
}

function showSearchStatus(text) {
  searchStatus.hidden = false;
  searchStatus.textContent = text;
}

function hideSearchStatus() {
  searchStatus.hidden = true;
  searchStatus.textContent = "";
}

async function getJson(url) {
  const response = await fetch(url);
  return readJsonResponse(response);
}

async function postStream(url, body, onEvent) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    return readJsonResponse(response);
  }

  if (!response.body) {
    throw new Error("Streaming is not supported by this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const searches = [];
  let answer = "";
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() || "";

    for (const eventText of events) {
      const event = parseStreamEvent(eventText);
      if (!event) {
        continue;
      }

      if (event.type === "error") {
        throw new Error(event.payload || "Stream failed");
      }

      if (event.type === "search" && event.payload.query) {
        searches.push(event.payload.query);
      }

      if (event.type === "text") {
        answer += event.payload;
      }

      onEvent(event);
    }
  }

  return { answer, searches };
}

function parseStreamEvent(eventText) {
  const dataLine = eventText
    .split("\n")
    .find((line) => line.startsWith("data: "));
  if (!dataLine) {
    return null;
  }

  return JSON.parse(dataLine.slice("data: ".length));
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readJsonResponse(response);
}

async function readJsonResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}
