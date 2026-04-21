const messages = document.getElementById("messages");
const input = document.getElementById("input");
const history = document.getElementById("history");
const stopBtn = document.getElementById("stop");

let chats = [];
let current = [];
let stop = false;

/* START */
function startApp() {
  document.getElementById("landing").style.display = "none";
  document.getElementById("app").classList.remove("hidden");
}

/* FORMAT TEXT */
function formatText(text) {
  return text
    .replace(/\n/g, "<br>")
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");
}

/* ADD MESSAGE */
function addMessage(text, type, typing = false) {
  const msg = document.createElement("div");
  msg.className = "message " + type;

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  msg.appendChild(bubble);
  messages.appendChild(msg);

  if (typing) {
    typeWriter(formatText(text), bubble);
  } else {
    bubble.innerHTML = formatText(text);
  }

  messages.scrollTop = messages.scrollHeight;
  current.push({ text, type });
}

/* TYPE EFFECT */
function typeWriter(text, el, i = 0) {
  stop = false;
  stopBtn.style.display = "inline";

  if (i < text.length && !stop) {
    el.innerHTML = text.substring(0, i + 1);
    setTimeout(() => typeWriter(text, el, i + 1), 8);
  } else {
    stopBtn.style.display = "none";
  }
}

/* STOP */
function stopTyping() {
  stop = true;
}

/* SEND */
async function sendMessage() {
  const msg = input.value.trim();
  if (!msg) return;

  addMessage(msg, "user");
  input.value = "";

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: msg })
  });

  const data = await res.json();

  addMessage(data.response, "bot", true);
}

/* ENTER */
input.addEventListener("keydown", e => {
  if (e.key === "Enter") sendMessage();
});

/* NEW CHAT */
function newChat() {
  if (current.length > 0) {
    chats.push([...current]);
    addHistory(`Chat ${chats.length}`, chats.length - 1);
  }

  current = [];
  messages.innerHTML = "";
}

/* HISTORY */
function addHistory(name, i) {
  const li = document.createElement("li");
  li.innerText = name;
  li.onclick = () => loadChat(i);
  history.appendChild(li);
}

function loadChat(i) {
  messages.innerHTML = "";
  current = chats[i];

  current.forEach(m => {
    addMessage(m.text, m.type);
  });
}

/* MENU */
function showChat() {
  document.getElementById("ideas-panel").classList.add("hidden");
}

function showIdeas() {
  document.getElementById("ideas-panel").classList.remove("hidden");
}

function quickAsk(q) {
  input.value = q;
  sendMessage();
}