import os
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="DamperBot")


class ChatRequest(BaseModel):
    message: str


HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>DamperBot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui; max-width: 720px; margin: 40px auto; padding: 0 16px; background: #0f0f0f; color: #eee; }
    h1 { text-align: center; }
    h1 span { color: #4ade80; }
    #chat { border: 1px solid #333; border-radius: 12px; padding: 16px; height: 60vh; overflow-y: auto; background: #181818; }
    .msg { margin: 10px 0; padding: 10px 14px; border-radius: 12px; max-width: 80%; white-space: pre-wrap; line-height: 1.5; }
    .user { background: #2563eb; color: white; margin-left: auto; }
    .ai { background: #262626; color: #eee; }
    form { display: flex; gap: 8px; margin-top: 12px; }
    input { flex: 1; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #eee; }
    button { padding: 12px 20px; border-radius: 8px; border: none; background: #4ade80; color: #0f0f0f; font-weight: bold; cursor: pointer; }
    button:hover { background: #22c55e; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
  </style>
</head>
<body>
  <h1><span>Damper</span>Bot 🤖</h1>
  <div id="chat"></div>
  <form id="form">
    <input id="input" placeholder="Ask DamperBot anything..." autocomplete="off" required>
    <button id="send">Send</button>
  </form>
  <script>
    const chat = document.getElementById('chat');
    const form = document.getElementById('form');
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('send');

    function addMsg(text, cls) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    addMsg("Hi! I'm DamperBot. How can I help you today?", 'ai');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const msg = input.value.trim();
      if (!msg) return;

      addMsg(msg, 'user');
      input.value = '';
      sendBtn.disabled = true;

      const aiDiv = addMsg('', 'ai');

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value, { stream: true });
          aiDiv.textContent += chunk;
          chat.scrollTop = chat.scrollHeight;
        }
      } catch (err) {
        aiDiv.textContent = 'Error: ' + err.message;
      }

      sendBtn.disabled = false;
      input.focus();
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/api/chat")
async def chat(req: ChatRequest):
    def stream():
        try:
            completion = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are DamperBot, a friendly and helpful AI assistant."},
                    {"role": "user", "content": req.message},
                ],
                stream=True,
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            yield f"\\n[Error: {e}]"

    return StreamingResponse(stream(), media_type="text/plain")