import os
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

MODEL_PRIMARY = "openai/gpt-oss-120b"
MODEL_FALLBACK = "openai/gpt-oss-20b"

app = FastAPI(title="DamperBot")


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

class DetectRequest(BaseModel):
    message: str


@app.get("/config")
def config():
    return {"supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}


@app.post("/api/detect-intent")
async def detect_intent(req: DetectRequest):
    system = """Classify the user's message. Reply ONLY with valid JSON.
{"intent": "image"|"video"|"chat", "prompt": "cleaned prompt"}
- "image" = user wants you to MAKE/DRAW/CREATE/SHOW an image/picture/photo
- "video" = user wants you to MAKE/CREATE/ANIMATE/SHOW a video
- "chat" = everything else
For image/video: extract ONLY the visual description.
For chat: use the original message unchanged.
Examples:
- "draw a cat" -> {"intent": "image", "prompt": "a cat"}
- "make a video of waves" -> {"intent": "video", "prompt": "waves"}
- "what is 2+2?" -> {"intent": "chat", "prompt": "what is 2+2?"}
- "how do I draw a cat?" -> {"intent": "chat", "prompt": "how do I draw a cat?"}
"""
    try:
        response = groq_client.chat.completions.create(
            model=MODEL_FALLBACK,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": req.message[:400]},
            ],
            temperature=0.1,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        intent = parsed.get("intent", "chat")
        prompt = parsed.get("prompt", req.message)
        if intent not in ("image", "video", "chat"):
            intent = "chat"
        return {"intent": intent, "prompt": prompt}
    except Exception as e:
        print("Intent error:", e)
        return {"intent": "chat", "prompt": req.message}


HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>DamperBot</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>&#129302;</text></svg>">
<script src="https://cdn.tailwindcss.com"></script>
<script>tailwind.config = { darkMode: 'class' };</script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<style>
  html, body { height: 100%; margin: 0; overflow: hidden; }
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif; -webkit-font-smoothing: antialiased; }

  .scroll-thin::-webkit-scrollbar { width: 6px; height: 6px; }
  .scroll-thin::-webkit-scrollbar-thumb { background: rgba(120,120,120,0.25); border-radius: 3px; }
  .scroll-thin::-webkit-scrollbar-thumb:hover { background: rgba(120,120,120,0.5); }

  /* ---------- Markdown ---------- */
  .msg-md { line-height: 1.7; font-size: 15px; word-wrap: break-word; }
  .msg-md > *:first-child { margin-top: 0 !important; }
  .msg-md > *:last-child { margin-bottom: 0 !important; }
  .msg-md p { margin: 0 0 0.8em 0; }
  .msg-md ul, .msg-md ol { margin: 0.5em 0 0.8em 1.5em; }
  .msg-md li { margin: 0.25em 0; }
  .msg-md code { font-family: 'SF Mono', monospace; font-size: 0.87em; padding: 0.15em 0.42em; border-radius: 5px; background: rgba(249,115,22,0.12); color: #c2410c; font-weight: 500; }
  .dark .msg-md code { background: rgba(251,146,60,0.15); color: #fdba74; }
  .msg-md pre { margin: 0.9em 0; padding: 1em 1.15em; border-radius: 10px; overflow-x: auto; background: #0d1117; border: 1px solid rgba(255,255,255,0.06); }
  .msg-md pre code { padding: 0; background: transparent; color: #e6edf3; font-size: 13.2px; line-height: 1.65; }
  .msg-md blockquote { border-left: 3px solid #f97316; padding: 0.2em 0 0.2em 1em; margin: 0.9em 0; opacity: 0.9; font-style: italic; }
  .msg-md a { color: #ea580c; text-decoration: underline; font-weight: 500; }
  .dark .msg-md a { color: #fb923c; }
  .msg-md hr { border: none; border-top: 1px solid rgba(127,127,127,0.18); margin: 1.4em 0; }
  .msg-md table { border-collapse: collapse; margin: 0.9em 0; display: block; overflow-x: auto; font-size: 14px; }
  .msg-md th, .msg-md td { border: 1px solid rgba(127,127,127,0.22); padding: 0.55em 0.95em; text-align: left; }
  .msg-md th { background: rgba(249,115,22,0.08); font-weight: 600; }
  .msg-md img { max-width: 100%; border-radius: 12px; margin: 0.6em 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .msg-md video { max-width: 100%; border-radius: 12px; margin: 0.5em 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .msg-md h1, .msg-md h2, .msg-md h3 { margin: 1.2em 0 0.5em; font-weight: 700; line-height: 1.3; }
  .msg-md h1 { font-size: 1.5em; } .msg-md h2 { font-size: 1.3em; } .msg-md h3 { font-size: 1.13em; }

  /* ---------- Code blocks ---------- */
  .code-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5em 0.9em; background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 11.5px; color: #8b949e; font-weight: 500;
  }
  .code-copy { display: inline-flex; align-items: center; gap: 4px; color: #8b949e; background: transparent; border: none; cursor: pointer; padding: 4px 8px; border-radius: 6px; transition: all 0.15s; font-size: 11.5px; font-weight: 500; }
  .code-copy:hover { background: rgba(255,255,255,0.08); color: #e6edf3; }
  .code-copy.copied { color: #4ade80; }

  /* ---------- Animations ---------- */
  .typing-cursor::after { content: "\258D"; display: inline-block; margin-left: 1px; color: #f97316; animation: blink 1s step-start infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
  .fade-in { animation: fadeInUp 0.32s cubic-bezier(0.16, 1, 0.3, 1); }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---------- Utilities ---------- */
  textarea { resize: none; }
  .grad-text { background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
  .logo-grad { background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%); }
  #fatal { display: none; padding: 40px; text-align: center; font-family: Inter, sans-serif; color: #ef4444; }

  /* ---------- Mic active ---------- */
  .mic-active { color: #f97316 !important; }
  #micBtn { position: relative; }

  /* ---------- Media loading ---------- */
  .media-loading {
    width: 100%; max-width: 512px; aspect-ratio: 1/1; border-radius: 12px;
    background: linear-gradient(90deg, rgba(120,120,120,0.08) 0%, rgba(120,120,120,0.18) 50%, rgba(120,120,120,0.08) 100%);
    background-size: 200% 100%; animation: shimmer 1.5s infinite;
    display: flex; align-items: center; justify-content: center;
    color: #888; font-size: 13px;
  }
  @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }

  /* ---------- SIDEBAR (mobile-first) ---------- */
  #sidebar {
    position: fixed;
    top: 0; left: 0; bottom: 0;
    width: 280px;
    z-index: 50;
    transform: translateX(-100%);
    transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    will-change: transform;
  }
  #sidebar.open { transform: translateX(0); }

  #overlay {
    position: fixed; inset: 0; z-index: 40;
    background: rgba(0,0,0,0.6);
    backdrop-filter: blur(4px);
    opacity: 0; pointer-events: none;
    transition: opacity 0.3s ease;
  }
  #overlay.open { opacity: 1; pointer-events: auto; }

  /* Desktop: sidebar always visible */
  @media (min-width: 1024px) {
    #sidebar { transform: translateX(0); position: relative; }
    #overlay { display: none !important; }
    #menuBtn { display: none !important; }
  }

  /* ---------- Chat bubbles ---------- */
  .bubble-user {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
    color: white;
    border-radius: 20px 20px 4px 20px;
    padding: 10px 16px;
    box-shadow: 0 4px 15px rgba(249,115,22,0.2);
    max-width: 100%;
    word-wrap: break-word;
    white-space: pre-wrap;
  }
  .bubble-ai {
    background: white;
    color: #1f2937;
    border-radius: 20px 20px 20px 4px;
    padding: 12px 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.04);
    max-width: 100%;
  }
  .dark .bubble-ai {
    background: #1a1a1a;
    color: #e5e7eb;
    border: 1px solid rgba(255,255,255,0.06);
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }

  /* ---------- CALL MODE ---------- */
  #callScreen {
    position: fixed; inset: 0; z-index: 100; display: none;
    background: radial-gradient(circle at 50% 30%, #1a1a1a 0%, #0a0a0a 60%);
    align-items: center; justify-content: center; flex-direction: column;
    padding: 24px;
  }
  #callScreen.active { display: flex; }
  .call-avatar {
    width: 140px; height: 140px; border-radius: 50%;
    background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 64px; box-shadow: 0 20px 60px rgba(249,115,22,0.4);
    transition: all 0.3s ease;
  }
  .call-avatar.listening { animation: pulse 1.6s ease-in-out infinite; }
  .call-avatar.speaking { box-shadow: 0 20px 80px rgba(249,115,22,0.7); }
  .call-avatar.thinking { animation: spin-glow 2s linear infinite; }
  @keyframes pulse {
    0%, 100% { transform: scale(1); box-shadow: 0 20px 60px rgba(249,115,22,0.4); }
    50% { transform: scale(1.08); box-shadow: 0 20px 80px rgba(249,115,22,0.8); }
  }
  @keyframes spin-glow {
    0% { box-shadow: 0 20px 60px rgba(249,115,22,0.4); }
    50% { box-shadow: 0 20px 60px rgba(251,191,36,0.8); }
    100% { box-shadow: 0 20px 60px rgba(249,115,22,0.4); }
  }
  .call-status { font-size: 18px; margin-top: 32px; color: #9ca3af; font-weight: 500; letter-spacing: 0.5px; text-align: center; }
  .call-transcript { max-width: 500px; margin-top: 24px; font-size: 15px; color: #e5e7eb; text-align: center; min-height: 60px; line-height: 1.5; opacity: 0.9; padding: 0 16px; }
  .call-controls { position: fixed; bottom: 40px; left: 0; right: 0; display: flex; justify-content: center; gap: 16px; }
  .call-btn { width: 60px; height: 60px; border-radius: 50%; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.2s; border: none; }
  .call-btn:hover { transform: scale(1.05); }
  .call-btn:active { transform: scale(0.95); }
  .call-btn.end { background: #ef4444; color: white; }
  .call-btn.mute { background: rgba(255,255,255,0.1); color: white; backdrop-filter: blur(10px); }
  .call-btn.mute.muted { background: #ef4444; }

  /* ---------- Mobile fixes ---------- */
  @media (max-width: 640px) {
    .msg-md { font-size: 14.5px; }
    .bubble-user, .bubble-ai { font-size: 14.5px; padding: 9px 14px; }
  }
</style>
</head>
<body class="bg-gray-50 dark:bg-[#0a0a0a] text-gray-900 dark:text-gray-100">

<div id="fatal"></div>

<!-- ========== AUTH ========== -->
<div id="authScreen" class="hidden fixed inset-0 z-[60] bg-gray-50 dark:bg-[#0a0a0a] flex items-center justify-center p-5">
  <div class="w-full max-w-[420px] fade-in">
    <div class="flex flex-col items-center mb-6">
      <div class="w-16 h-16 rounded-2xl logo-grad flex items-center justify-center text-white text-3xl mb-3 shadow-xl shadow-orange-500/30">&#129302;</div>
      <h1 class="text-2xl font-bold tracking-tight">Welcome to <span class="grad-text">DamperBot</span></h1>
      <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Sign in to save your chats across devices</p>
    </div>
    <div class="rounded-2xl border border-gray-200 dark:border-[#262626] bg-white dark:bg-[#141414] p-5 shadow-xl">
      <div class="flex gap-1 p-1 rounded-xl bg-gray-100 dark:bg-[#1a1a1a] mb-5">
        <button id="tabLogin" class="flex-1 py-2 rounded-lg text-sm font-semibold bg-white dark:bg-[#262626] text-orange-600 dark:text-orange-400 shadow-sm">Log in</button>
        <button id="tabSignup" class="flex-1 py-2 rounded-lg text-sm font-semibold text-gray-500 dark:text-gray-400">Sign up</button>
      </div>
      <form id="authForm" class="space-y-3">
        <div>
          <label class="block text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 uppercase tracking-wider">Email</label>
          <input id="emailInput" type="email" required placeholder="you@example.com" class="w-full px-3.5 py-3 rounded-xl border border-gray-300 dark:border-[#2a2a2a] bg-white dark:bg-[#1a1a1a] text-gray-900 dark:text-white outline-none focus:border-orange-500 transition" />
        </div>
        <div>
          <label class="block text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1.5 uppercase tracking-wider">Password</label>
          <input id="passwordInput" type="password" required minlength="6" placeholder="At least 6 characters" class="w-full px-3.5 py-3 rounded-xl border border-gray-300 dark:border-[#2a2a2a] bg-white dark:bg-[#1a1a1a] text-gray-900 dark:text-white outline-none focus:border-orange-500 transition" />
        </div>
        <div id="authError" class="hidden text-xs px-3 py-2 rounded-lg"></div>
        <button id="authSubmit" type="submit" class="w-full py-3 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-semibold text-sm transition shadow-lg shadow-orange-500/25 mt-2 flex items-center justify-center">
          <span id="authSubmitText">Log in</span>
        </button>
      </form>

      <div class="flex items-center gap-3 my-5">
        <div class="flex-1 h-px bg-gray-200 dark:bg-[#2a2a2a]"></div>
        <span class="text-[11px] text-gray-400 dark:text-gray-600 uppercase tracking-wider font-semibold">or</span>
        <div class="flex-1 h-px bg-gray-200 dark:bg-[#2a2a2a]"></div>
      </div>

      <button id="guestBtn" class="w-full py-3 rounded-xl border border-gray-300 dark:border-[#2a2a2a] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50 dark:hover:bg-orange-950/20 text-gray-700 dark:text-gray-300 font-semibold text-sm transition flex items-center justify-center gap-2">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        Continue as guest
      </button>
      <p class="text-[11px] text-gray-400 dark:text-gray-600 text-center mt-3">Chats stay in this browser only</p>
    </div>
  </div>
</div>

<!-- ========== CALL SCREEN ========== -->
<div id="callScreen">
  <div id="callAvatar" class="call-avatar">&#129302;</div>
  <div id="callStatus" class="call-status">Starting...</div>
  <div id="callTranscript" class="call-transcript"></div>
  <div class="call-controls">
    <button id="callMuteBtn" class="call-btn mute" title="Mute">
      <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
    </button>
    <button id="callEndBtn" class="call-btn end" title="End call">
      <svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-3.33-2.67m-2.67-3.34a19.79 19.79 0 0 1-3.07-8.63A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91"/><line x1="23" y1="1" x2="1" y2="23"/></svg>
    </button>
  </div>
</div>

<!-- ========== APP ========== -->
<div id="app" class="hidden h-screen flex overflow-hidden">

  <!-- Mobile overlay -->
  <div id="overlay"></div>

  <!-- SIDEBAR -->
  <aside id="sidebar" class="flex flex-col bg-white dark:bg-[#111111] border-r border-gray-200 dark:border-[#1a1a1a] shadow-2xl lg:shadow-none">

    <div class="p-4 flex items-center gap-2.5 shrink-0">
      <div class="w-9 h-9 rounded-xl logo-grad flex items-center justify-center text-white text-lg shadow-md shadow-orange-500/20">&#129302;</div>
      <div class="min-w-0 flex-1">
        <div class="font-bold text-[15px] leading-tight tracking-tight">DamperBot</div>
        <div class="text-[11px] text-gray-500">AI Assistant</div>
      </div>
      <button id="closeSidebarBtn" class="lg:hidden p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-[#1f1f1f] text-gray-500">
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>

    <div class="px-3 pb-3 shrink-0">
      <button id="newChatBtn" class="w-full flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-semibold text-sm transition shadow-md shadow-orange-500/20 active:scale-[0.98]">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New chat
      </button>
    </div>

    <div id="chatList" class="flex-1 overflow-y-auto scroll-thin px-2 pb-3 space-y-0.5"></div>

    <div id="guestBanner" class="hidden mx-3 mb-2 p-3 rounded-xl bg-orange-50 dark:bg-orange-950/20 border border-orange-200 dark:border-orange-900/40 shrink-0">
      <div class="text-[12px] font-medium text-orange-700 dark:text-orange-300 mb-2">You're browsing as guest</div>
      <button id="signupFromBanner" class="w-full py-2 rounded-lg bg-gradient-to-r from-orange-500 to-amber-500 text-white text-[12px] font-semibold hover:from-orange-600 hover:to-amber-600 transition">Sign up to save chats</button>
    </div>

    <div class="p-3 border-t border-gray-200 dark:border-[#1a1a1a] flex items-center gap-2 shrink-0">
      <div id="userAvatar" class="w-8 h-8 rounded-full logo-grad flex items-center justify-center text-white text-xs font-bold">?</div>
      <div class="min-w-0 flex-1">
        <div id="userEmail" class="text-[12px] font-medium truncate">-</div>
      </div>
      <button id="logoutBtn" class="p-2 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 text-gray-500 hover:text-red-500 transition" title="Log out">
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      </button>
    </div>
  </aside>

  <!-- MAIN -->
  <main class="flex-1 flex flex-col min-w-0 bg-gray-50 dark:bg-[#0a0a0a] relative">

    <header class="h-14 flex items-center gap-3 px-3 sm:px-4 border-b border-gray-200 dark:border-[#1a1a1a] bg-white/80 dark:bg-[#0a0a0a]/80 backdrop-blur-md z-20 shrink-0">
      <button id="menuBtn" class="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-[#1a1a1a] text-gray-600 dark:text-gray-300">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <div class="flex items-center gap-2.5 min-w-0 flex-1">
        <span class="font-semibold text-[15px] truncate" id="headerTitle">DamperBot</span>
        <span class="hidden sm:inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full bg-orange-50 dark:bg-orange-950/40 text-orange-600 dark:text-orange-400 font-medium border border-orange-200 dark:border-orange-900/50">
          <span class="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></span>
          <span id="connStatus">ready</span>
        </span>
      </div>
    </header>

    <div id="chat" class="flex-1 overflow-y-auto scroll-thin">
      <div id="emptyState" class="min-h-full flex flex-col items-center justify-center text-center px-6 py-10">
        <div class="w-20 h-20 rounded-3xl logo-grad flex items-center justify-center text-white text-4xl mb-5 shadow-2xl shadow-orange-500/30 fade-in">&#129302;</div>
        <h1 class="text-3xl font-extrabold mb-2 fade-in">Hi, I'm <span class="grad-text">DamperBot</span></h1>
        <p class="text-gray-500 dark:text-gray-400 max-w-md text-[15px] mb-8 fade-in">Chat, speak, make images and videos — or start a voice call.</p>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-xl w-full px-2">
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all active:scale-[0.98]" data-prompt="Draw a majestic white tiger in a misty forest">
            <div class="text-lg mb-1.5">&#127912;</div>
            <div class="font-semibold mb-0.5">Draw something</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">Draw a majestic white tiger...</div>
          </button>
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all active:scale-[0.98]" data-prompt="Make a video of a sunset over the ocean, waves crashing">
            <div class="text-lg mb-1.5">&#127909;</div>
            <div class="font-semibold mb-0.5">Make a video</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">Make a video of a sunset...</div>
          </button>
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all active:scale-[0.98]" data-prompt="Explain how neural networks work in detail">
            <div class="text-lg mb-1.5">&#128187;</div>
            <div class="font-semibold mb-0.5">Explain something</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">How neural networks work...</div>
          </button>
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all active:scale-[0.98]" data-prompt="Tell me a fun fact about space">
            <div class="text-lg mb-1.5">&#128172;</div>
            <div class="font-semibold mb-0.5">Just chat</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">Tell me a fun fact about space</div>
          </button>
        </div>
      </div>
      <div id="messages" class="max-w-3xl mx-auto px-3 sm:px-4 py-6 space-y-5 hidden"></div>
    </div>

    <div class="pt-2 pb-3 sm:pb-4 px-3 sm:px-4 shrink-0 bg-gradient-to-t from-gray-50 dark:from-[#0a0a0a] via-gray-50 dark:via-[#0a0a0a] to-transparent">
      <div class="max-w-3xl mx-auto">
        <form id="form" class="flex items-end gap-1.5 sm:gap-2 rounded-2xl border border-gray-300 dark:border-[#262626] bg-white dark:bg-[#141414] focus-within:border-orange-500 transition px-2 sm:px-3 py-2 shadow-lg shadow-black/[0.03] dark:shadow-black/30">
          <textarea id="input" rows="1" placeholder="Ask anything..." class="flex-1 bg-transparent outline-none py-2 px-1 sm:px-2 max-h-40 scroll-thin text-[15px] min-w-0" autocomplete="off"></textarea>

          <button id="callModeBtn" type="button" title="Voice call" class="shrink-0 w-9 h-9 rounded-xl text-orange-500 hover:bg-orange-50 dark:hover:bg-orange-950/30 flex items-center justify-center transition">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.42 19.42 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
          </button>

          <button id="micBtn" type="button" title="Voice input" class="shrink-0 w-9 h-9 rounded-xl text-gray-500 hover:bg-gray-200 dark:hover:bg-[#2a2a2a] flex items-center justify-center transition">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
          </button>

          <button id="speakBtn" type="button" title="Toggle voice reply" class="shrink-0 w-9 h-9 rounded-xl text-gray-500 hover:bg-gray-200 dark:hover:bg-[#2a2a2a] flex items-center justify-center transition">
            <svg id="speakerOff" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>
            <svg id="speakerOn" class="hidden" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
          </button>

          <button id="send" type="submit" class="shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 disabled:opacity-30 text-white flex items-center justify-center transition shadow-md shadow-orange-500/25 active:scale-95">
            <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </form>
        <p class="text-[10px] sm:text-[11px] text-center text-gray-400 dark:text-gray-600 mt-2 hidden sm:block">
          Say <em>"draw a..."</em>, <em>"make a video of..."</em>, or tap the phone icon for a voice call
        </p>
      </div>
    </div>
  </main>
</div>

<script>
(async function() {
  "use strict";

  function fatal(msg) {
    const f = document.getElementById('fatal');
    f.style.display = 'block';
    f.innerHTML = '<div style="max-width:500px;margin:0 auto">' +
      '<div style="font-size:48px;margin-bottom:12px">!</div>' +
      '<h2 style="color:#ef4444;margin:0 0 8px 0">Startup Error</h2>' +
      '<p style="color:#9ca3af;font-size:14px">' + msg + '</p></div>';
  }

  if (typeof window.supabase === 'undefined') { fatal('Supabase SDK failed to load.'); return; }

  let config;
  try { config = await (await fetch('/config')).json(); }
  catch (e) { fatal('Could not fetch /config.'); return; }

  if (!config.supabase_url || !config.supabase_anon_key) {
    fatal('Backend missing SUPABASE_URL or SUPABASE_ANON_KEY.'); return;
  }

  const sb = window.supabase.createClient(config.supabase_url, config.supabase_anon_key);

  let chats = [];
  let activeId = null;
  let isStreaming = false;
  let currentUser = null;
  let isGuest = false;
  let speakReplies = false;
  let callModeActive = false;
  let callMuted = false;

  const GUEST_STORAGE_KEY = 'damperbot_guest_chats_v2';
  const GUEST_ACTIVE_KEY = 'damperbot_guest_active_v2';
  const GUEST_FLAG_KEY = 'damperbot_guest_flag_v2';

  const $ = (id) => document.getElementById(id);
  const els = {
    authScreen: $('authScreen'), app: $('app'),
    tabLogin: $('tabLogin'), tabSignup: $('tabSignup'),
    authForm: $('authForm'), emailInput: $('emailInput'),
    passwordInput: $('passwordInput'), authError: $('authError'),
    authSubmit: $('authSubmit'), authSubmitText: $('authSubmitText'),
    guestBtn: $('guestBtn'), signupFromBanner: $('signupFromBanner'),
    guestBanner: $('guestBanner'),
    userEmail: $('userEmail'), userAvatar: $('userAvatar'), logoutBtn: $('logoutBtn'),
    newChatBtn: $('newChatBtn'), chatList: $('chatList'),
    messages: $('messages'), emptyState: $('emptyState'), chat: $('chat'),
    form: $('form'), input: $('input'), send: $('send'),
    headerTitle: $('headerTitle'), connStatus: $('connStatus'),
    micBtn: $('micBtn'), speakBtn: $('speakBtn'),
    speakerOff: $('speakerOff'), speakerOn: $('speakerOn'),
    callScreen: $('callScreen'), callAvatar: $('callAvatar'),
    callStatus: $('callStatus'), callTranscript: $('callTranscript'),
    callModeBtn: $('callModeBtn'), callEndBtn: $('callEndBtn'), callMuteBtn: $('callMuteBtn'),
    sidebar: $('sidebar'), overlay: $('overlay'),
    menuBtn: $('menuBtn'), closeSidebarBtn: $('closeSidebarBtn'),
  };

  // ---------- SIDEBAR ----------
  function openSidebar() {
    els.sidebar.classList.add('open');
    els.overlay.classList.add('open');
  }
  function closeSidebar() {
    els.sidebar.classList.remove('open');
    els.overlay.classList.remove('open');
  }
  els.menuBtn.addEventListener('click', openSidebar);
  els.closeSidebarBtn.addEventListener('click', closeSidebar);
  els.overlay.addEventListener('click', closeSidebar);

  function setStatus(text, color) {
    els.connStatus.textContent = text;
    // Also style the pill wrapper
    const pill = els.connStatus.parentElement;
    if (pill) {
      pill.className = 'hidden sm:inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full font-medium border ' +
        (color === 'green' ? 'bg-orange-50 dark:bg-orange-950/40 text-orange-600 dark:text-orange-400 border-orange-200 dark:border-orange-900/50' :
         color === 'red' ? 'bg-red-50 dark:bg-red-950/40 text-red-600 dark:text-red-400 border-red-200 dark:border-red-900/50' :
         'bg-yellow-50 dark:bg-yellow-950/40 text-yellow-600 dark:text-yellow-400 border-yellow-200 dark:border-yellow-900/50');
    }
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  // ---------- VOICE ----------
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const synth = window.speechSynthesis;
  let recognition = null;
  let isListening = false;

  if (SR) {
    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
  }

  function stripMarkdown(text) {
    return text
      .replace(/!\[[^\]]*\]\([^)]+\)/g, ' image ')
      .replace(/```[\s\S]*?```/g, ' code block ')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/#{1,6}\s/g, '')
      .replace(/\*\*([^*]+)\*\*/g, '$1')
      .replace(/\*([^*]+)\*/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/^\s*[-*+]\s/gm, '')
      .replace(/^\s*\d+\.\s/gm, '')
      .replace(/\n{2,}/g, '. ')
      .replace(/\n/g, ' ');
  }

  function speakText(text, onEnd) {
    if (!synth) { if (onEnd) onEnd(); return; }
    synth.cancel();
    const clean = stripMarkdown(text).slice(0, 3000);
    const u = new SpeechSynthesisUtterance(clean);
    u.rate = 1.0; u.pitch = 1.0; u.volume = 1.0;
    const voices = synth.getVoices();
    const preferred = voices.find(v => v.name.includes('Google US English') || v.name.includes('Samantha') || v.name.includes('Microsoft Aria'));
    if (preferred) u.voice = preferred;
    if (onEnd) { u.onend = onEnd; u.onerror = onEnd; }
    synth.speak(u);
  }

  if (SR) {
    recognition.onstart = () => {
      isListening = true;
      els.micBtn.classList.add('mic-active');
      els.input.placeholder = 'Listening...';
    };
    recognition.onend = () => {
      isListening = false;
      els.micBtn.classList.remove('mic-active');
      els.input.placeholder = 'Ask anything...';
    };
    recognition.onresult = (event) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += t; else interim += t;
      }
      els.input.value = final || interim;
      els.input.dispatchEvent(new Event('input'));
      if (final) setTimeout(() => { if (els.input.value.trim()) els.form.requestSubmit(); }, 250);
    };
    recognition.onerror = (event) => {
      console.error('Speech error:', event.error);
      isListening = false;
      els.micBtn.classList.remove('mic-active');
      els.input.placeholder = 'Ask anything...';
      if (event.error === 'not-allowed' && !callModeActive) {
        alert('Microphone blocked.\n\nFix:\n1. Click the padlock in the address bar\n2. Site settings -> Microphone -> Allow\n3. Reload');
      }
    };
    els.micBtn.addEventListener('click', () => {
      if (isListening) { recognition.stop(); return; }
      els.input.value = '';
      try { recognition.start(); } catch (e) { console.error(e); }
    });
  } else {
    els.micBtn.style.display = 'none';
  }

  els.speakBtn.addEventListener('click', () => {
    speakReplies = !speakReplies;
    els.speakerOn.classList.toggle('hidden', !speakReplies);
    els.speakerOff.classList.toggle('hidden', speakReplies);
    els.speakBtn.classList.toggle('text-orange-500', speakReplies);
    if (!speakReplies && synth) synth.cancel();
  });

  // ---------- CALL MODE ----------
  let callRecognition = null;

  function setCallStatus(text, avatarClass) {
    els.callStatus.textContent = text;
    els.callAvatar.className = 'call-avatar' + (avatarClass ? ' ' + avatarClass : '');
  }

  function startCallRecognition() {
    if (!SR || !callModeActive) return;
    if (callRecognition) { try { callRecognition.abort(); } catch (e) {} }
    callRecognition = new SR();
    callRecognition.continuous = false;
    callRecognition.interimResults = true;
    callRecognition.lang = 'en-US';

    let finalText = '';

    callRecognition.onstart = () => {
      setCallStatus('Listening...', 'listening');
      els.callTranscript.textContent = '';
    };

    callRecognition.onresult = (event) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += t; else interim += t;
      }
      els.callTranscript.textContent = final || interim;
      if (final) finalText = final;
    };

    callRecognition.onend = () => {
      if (!callModeActive) return;
      if (finalText.trim()) {
        handleCallMessage(finalText.trim());
      } else {
        setTimeout(() => { if (callModeActive) startCallRecognition(); }, 300);
      }
    };

    callRecognition.onerror = (event) => {
      console.error('Call speech error:', event.error);
      if (event.error === 'not-allowed') {
        setCallStatus('Microphone blocked — check permissions', '');
        return;
      }
      if (callModeActive && event.error !== 'aborted') {
        setTimeout(() => { if (callModeActive) startCallRecognition(); }, 500);
      }
    };

    try { callRecognition.start(); } catch (e) { console.error(e); }
  }

  async function handleCallMessage(text) {
    if (!callModeActive) return;

    setCallStatus('Thinking...', 'thinking');
    els.callTranscript.textContent = 'You: ' + text;

    if (!currentChat()) {
      const chat = { id: null, title: 'Voice call', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
      const row = await dbInsert(chat);
      if (row) {
        chats.unshift(chat);
        activeId = chat.id;
        renderSidebar();
      }
    }
    const c = currentChat();
    if (!c) return;

    c.messages.push({ role: 'user', content: text });
    if (c.title === 'New chat' || c.title === 'Voice call') {
      c.title = text.length > 40 ? text.slice(0, 40) + '...' : text;
    }
    c.updatedAt = Date.now();
    renderSidebar();

    const detected = await detectIntent(text);
    if (detected.intent === 'image' || detected.intent === 'video') {
      const spoken = "I've created a " + detected.intent + " for you. Check it in the chat.";
      c.messages.push({ role: 'assistant', content: spoken });
      await dbUpdate(c);
      if (callModeActive) {
        speakText(spoken, () => {
          if (callModeActive) setTimeout(startCallRecognition, 400);
        });
      }
      return;
    }

    let aiText = '';
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: c.messages }),
      });
      if (!res.ok) throw new Error('Server ' + res.status);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        aiText += decoder.decode(value, { stream: true });
        els.callTranscript.textContent = 'DamperBot: ' + aiText;
      }
    } catch (err) {
      aiText = "Sorry, I couldn't process that.";
    }

    c.messages.push({ role: 'assistant', content: aiText });
    c.updatedAt = Date.now();
    await dbUpdate(c);
    renderMessages();
    renderSidebar();

    if (callModeActive) {
      setCallStatus('Speaking...', 'speaking');
      speakText(aiText, () => {
        if (callModeActive) {
          setTimeout(() => { if (callModeActive) startCallRecognition(); }, 500);
        }
      });
    }
  }

  async function startCallMode() {
    if (!SR) { alert('Your browser does not support voice input. Try Chrome.'); return; }
    callModeActive = true;
    callMuted = false;
    els.callScreen.classList.add('active');
    els.callMuteBtn.classList.remove('muted');
    els.callTranscript.textContent = '';
    setCallStatus('Starting...', '');
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (e) {
      setCallStatus('Microphone access denied', '');
      return;
    }
    setTimeout(() => startCallRecognition(), 400);
  }

  function endCallMode() {
    callModeActive = false;
    if (callRecognition) { try { callRecognition.abort(); } catch (e) {} }
    if (synth) synth.cancel();
    els.callScreen.classList.remove('active');
    renderMessages();
  }

  els.callModeBtn.addEventListener('click', startCallMode);
  els.callEndBtn.addEventListener('click', endCallMode);
  els.callMuteBtn.addEventListener('click', () => {
    callMuted = !callMuted;
    els.callMuteBtn.classList.toggle('muted', callMuted);
    if (callMuted) {
      if (callRecognition) { try { callRecognition.abort(); } catch (e) {} }
      setCallStatus('Muted', '');
    } else {
      if (callModeActive) startCallRecognition();
    }
  });

  // ---------- Intent ----------
  async function detectIntent(text) {
    try {
      const res = await fetch('/api/detect-intent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text }),
      });
      if (!res.ok) throw new Error('Server ' + res.status);
      return await res.json();
    } catch (err) {
      console.error('Intent detection failed:', err);
      return { intent: 'chat', prompt: text };
    }
  }

  function buildImageUrl(prompt) {
    const seed = Math.floor(Math.random() * 1000000);
    const encoded = encodeURIComponent(prompt);
    return `https://gen.pollinations.ai/image/${encoded}?width=1024&height=1024&nologo=true&seed=${seed}&model=flux&enhance=true&private=true`;
  }

  function buildVideoUrl(prompt) {
    const seed = Math.floor(Math.random() * 1000000);
    const encoded = encodeURIComponent(prompt);
    return `https://gen.pollinations.ai/video/${encoded}?seed=${seed}`;
  }

  // ---------- STORAGE ----------
  function saveGuestChats() {
    try {
      localStorage.setItem(GUEST_STORAGE_KEY, JSON.stringify(chats));
      localStorage.setItem(GUEST_ACTIVE_KEY, activeId || '');
    } catch (e) { console.error('Guest save failed:', e); }
  }

  function loadGuestChats() {
    try {
      chats = JSON.parse(localStorage.getItem(GUEST_STORAGE_KEY) || '[]');
      activeId = localStorage.getItem(GUEST_ACTIVE_KEY) || null;
      if (chats.length > 0 && !activeId) activeId = chats[0].id;
    } catch (e) { chats = []; }
  }

  async function loadChats() {
    if (isGuest) {
      loadGuestChats();
      renderSidebar();
      renderMessages();
      setStatus('ready', 'green');
      return;
    }
    setStatus('loading...', 'yellow');
    const { data, error } = await sb.from('chats').select('*').order('updated_at', { ascending: false });
    if (error) { console.error(error); setStatus('db error', 'red'); return; }
    chats = (data || []).map(r => ({
      id: r.id, title: r.title, messages: r.messages || [],
      createdAt: new Date(r.created_at).getTime(),
      updatedAt: new Date(r.updated_at).getTime(),
    }));
    if (chats.length > 0 && !activeId) activeId = chats[0].id;
    renderSidebar();
    renderMessages();
    setStatus('ready', 'green');
  }

  async function dbInsert(chat) {
    if (isGuest) {
      chat.id = 'g_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
      saveGuestChats();
      return { id: chat.id };
    }
    const { data, error } = await sb.from('chats').insert({
      user_id: currentUser.id, title: chat.title, messages: chat.messages,
    }).select().single();
    if (error) { console.error(error); return null; }
    chat.id = data.id;
    return data;
  }

  async function dbUpdate(chat) {
    if (isGuest) { saveGuestChats(); return; }
    const { error } = await sb.from('chats').update({ title: chat.title, messages: chat.messages }).eq('id', chat.id);
    if (error) console.error(error);
  }

  async function dbDelete(id) {
    if (isGuest) { saveGuestChats(); return; }
    const { error } = await sb.from('chats').delete().eq('id', id);
    if (error) console.error(error);
  }

  // ---------- AUTH UI ----------
  let authMode = 'login';
  function setAuthMode(mode) {
    authMode = mode;
    if (mode === 'login') {
      els.tabLogin.className = 'flex-1 py-2 rounded-lg text-sm font-semibold bg-white dark:bg-[#262626] text-orange-600 dark:text-orange-400 shadow-sm';
      els.tabSignup.className = 'flex-1 py-2 rounded-lg text-sm font-semibold text-gray-500 dark:text-gray-400';
      els.authSubmitText.textContent = 'Log in';
    } else {
      els.tabLogin.className = 'flex-1 py-2 rounded-lg text-sm font-semibold text-gray-500 dark:text-gray-400';
      els.tabSignup.className = 'flex-1 py-2 rounded-lg text-sm font-semibold bg-white dark:bg-[#262626] text-orange-600 dark:text-orange-400 shadow-sm';
      els.authSubmitText.textContent = 'Create account';
    }
    els.authError.classList.add('hidden');
  }
  els.tabLogin.addEventListener('click', () => setAuthMode('login'));
  els.tabSignup.addEventListener('click', () => setAuthMode('signup'));

  function showAuthError(msg, isGood) {
    els.authError.textContent = msg;
    els.authError.className = 'text-xs px-3 py-2 rounded-lg ' +
      (isGood ? 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-950/30'
              : 'text-red-500 dark:text-red-400 bg-red-50 dark:bg-red-950/30');
  }

  els.authForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const email = els.emailInput.value.trim();
    const password = els.passwordInput.value;
    els.authError.classList.add('hidden');
    els.authSubmit.disabled = true;
    els.authSubmitText.innerHTML = '<span class="spinner"></span>';
    try {
      let result;
      if (authMode === 'login') result = await sb.auth.signInWithPassword({ email, password });
      else result = await sb.auth.signUp({ email, password });
      if (result.error) throw result.error;
      if (authMode === 'signup' && !result.data.session) {
        showAuthError('Account created. Check your email to confirm.', true);
      }
    } catch (err) {
      showAuthError(err.message || 'Something went wrong', false);
    } finally {
      els.authSubmit.disabled = false;
      els.authSubmitText.textContent = authMode === 'login' ? 'Log in' : 'Create account';
    }
  });

  els.guestBtn.addEventListener('click', () => {
    isGuest = true;
    currentUser = null;
    localStorage.setItem(GUEST_FLAG_KEY, '1');
    enterApp();
  });

  els.signupFromBanner.addEventListener('click', () => {
    localStorage.removeItem(GUEST_FLAG_KEY);
    isGuest = false;
    els.authScreen.classList.remove('hidden');
    els.app.classList.add('hidden');
    setAuthMode('signup');
  });

  async function enterApp() {
    els.authScreen.classList.add('hidden');
    els.app.classList.remove('hidden');
    if (isGuest) {
      els.userEmail.textContent = 'Guest';
      els.userAvatar.textContent = 'G';
      els.guestBanner.classList.remove('hidden');
    } else {
      els.userEmail.textContent = currentUser.email;
      els.userAvatar.textContent = (currentUser.email[0] || '?').toUpperCase();
      els.guestBanner.classList.add('hidden');
    }
    setStatus('ready', 'green');
    await loadChats();
  }

  async function exitToAuth() {
    isGuest = false;
    currentUser = null;
    chats = [];
    activeId = null;
    localStorage.removeItem(GUEST_FLAG_KEY);
    els.authScreen.classList.remove('hidden');
    els.app.classList.add('hidden');
    if (callModeActive) endCallMode();
    setAuthMode('login');
  }

  els.logoutBtn.addEventListener('click', async () => {
    if (!confirm(isGuest ? 'Leave guest mode?' : 'Log out?')) return;
    if (synth) synth.cancel();
    if (callModeActive) endCallMode();
    if (isGuest) {
      exitToAuth();
    } else {
      await sb.auth.signOut();
    }
  });

  sb.auth.onAuthStateChange(async (event, session) => {
    if (session?.user) {
      isGuest = false;
      localStorage.removeItem(GUEST_FLAG_KEY);
      currentUser = session.user;
      await enterApp();
    } else {
      const wasGuest = localStorage.getItem(GUEST_FLAG_KEY) === '1';
      if (wasGuest) {
        isGuest = true;
        currentUser = null;
        await enterApp();
      } else {
        await exitToAuth();
      }
    }
  });

  // ---------- CHATS ----------
  async function newChat() {
    const cur = currentChat();
    if (cur && cur.messages.length === 0) { els.input.focus(); closeSidebar(); return; }
    const chat = { id: null, title: 'New chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
    const row = await dbInsert(chat);
    if (!row) return;
    chats.unshift(chat);
    activeId = chat.id;
    if (isGuest) saveGuestChats();
    renderSidebar();
    renderMessages();
    closeSidebar();
    els.input.focus();
  }

  async function deleteChat(id) {
    await dbDelete(id);
    chats = chats.filter(c => c.id !== id);
    if (activeId === id) activeId = chats[0]?.id || null;
    if (isGuest) saveGuestChats();
    renderSidebar();
    renderMessages();
  }

  function selectChat(id) {
    if (isStreaming) return;
    if (synth) synth.cancel();
    activeId = id;
    if (isGuest) saveGuestChats();
    renderSidebar();
    renderMessages();
    closeSidebar();
  }

  function currentChat() { return chats.find(c => c.id === activeId); }

  function renderSidebar() {
    els.chatList.innerHTML = '';
    if (chats.length === 0) {
      els.chatList.innerHTML = '<div class="text-xs text-gray-400 dark:text-gray-600 px-3 py-6 text-center">No chats yet</div>';
      return;
    }
    chats.forEach(c => {
      const row = document.createElement('div');
      const active = c.id === activeId;
      row.className = 'group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer text-[13px] ' +
        (active ? 'bg-orange-50 dark:bg-orange-950/30 text-orange-700 dark:text-orange-300 font-medium'
                : 'hover:bg-gray-100 dark:hover:bg-[#1a1a1a] text-gray-700 dark:text-gray-300');
      row.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="shrink-0 ' + (active ? 'text-orange-500' : 'opacity-60') + '"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>' +
        '<span class="flex-1 truncate">' + escapeHtml(c.title) + '</span>' +
        '<button class="del opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-200 dark:hover:bg-[#2a2a2a] text-xs" title="Delete">x</button>';
      row.addEventListener('click', (e) => { if (!e.target.closest('.del')) selectChat(c.id); });
      row.querySelector('.del').addEventListener('click', (e) => { e.stopPropagation(); deleteChat(c.id); });
      els.chatList.appendChild(row);
    });
  }

  // ---------- MESSAGES (redesigned bubbles) ----------
  function makeUserMsg(content) {
    const w = document.createElement('div');
    w.className = 'flex gap-2 sm:gap-3 justify-end fade-in';
    w.innerHTML = '<div class="bubble-user max-w-[85%] sm:max-w-[75%] text-[15px]">' + escapeHtml(content) + '</div>' +
      '<div class="shrink-0 w-8 h-8 rounded-full bg-gray-200 dark:bg-[#262626] flex items-center justify-center text-[11px] font-semibold text-gray-600 dark:text-gray-300">You</div>';
    return w;
  }

  function makeAiMsg() {
    const w = document.createElement('div');
    w.className = 'flex gap-2 sm:gap-3 fade-in';
    w.innerHTML = '<div class="shrink-0 w-8 h-8 rounded-full logo-grad flex items-center justify-center text-white text-[14px] shadow-md shadow-orange-500/25">&#129302;</div>' +
      '<div class="flex-1 min-w-0"><div class="bubble-ai msg-md typing-cursor"></div></div>';
    return w;
  }

  function renderMessages() {
    const c = currentChat();
    els.messages.innerHTML = '';
    if (!c || c.messages.length === 0) {
      els.emptyState.classList.remove('hidden');
      els.messages.classList.add('hidden');
      els.headerTitle.textContent = 'DamperBot';
      return;
    }
    els.emptyState.classList.add('hidden');
    els.messages.classList.remove('hidden');
    els.headerTitle.textContent = c.title;
    c.messages.forEach(m => {
      if (m.role === 'user') els.messages.appendChild(makeUserMsg(m.content));
      else {
        const w = makeAiMsg();
        const t = w.querySelector('.msg-md');
        t.classList.remove('typing-cursor');
        t.innerHTML = marked.parse(m.content || '');
        enhanceCodeBlocks(t);
        t.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
        els.messages.appendChild(w);
      }
    });
    els.chat.scrollTop = els.chat.scrollHeight;
  }

  function enhanceCodeBlocks(container) {
    container.querySelectorAll('pre').forEach(pre => {
      if (pre.dataset.enhanced) return;
      pre.dataset.enhanced = '1';
      const code = pre.querySelector('code');
      if (!code) return;
      const lang = (code.className.match(/language-(\w+)/) || [])[1] || 'code';
      const header = document.createElement('div');
      header.className = 'code-header';
      header.innerHTML = '<span>' + lang + '</span>';
      const btn = document.createElement('button');
      btn.className = 'code-copy';
      btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>';
      btn.addEventListener('click', () => {
        navigator.clipboard.writeText(code.innerText).then(() => {
          btn.classList.add('copied');
          btn.querySelector('span').textContent = 'Copied';
          setTimeout(() => { btn.classList.remove('copied'); btn.querySelector('span').textContent = 'Copy'; }, 1500);
        });
      });
      header.appendChild(btn);
      pre.insertBefore(header, code);
    });
  }

  function updateSendState() {
    els.send.disabled = isStreaming || els.input.value.trim().length === 0;
  }
  els.input.addEventListener('input', () => {
    els.input.style.height = 'auto';
    els.input.style.height = Math.min(els.input.scrollHeight, 160) + 'px';
    updateSendState();
  });
  els.input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); els.form.requestSubmit(); }
  });
  els.newChatBtn.addEventListener('click', newChat);

  document.querySelectorAll('.suggestion').forEach(b => {
    b.addEventListener('click', () => {
      els.input.value = b.dataset.prompt;
      els.input.dispatchEvent(new Event('input'));
      els.form.requestSubmit();
    });
  });

  async function handleMediaGeneration(c, prompt, type) {
    const aiW = makeAiMsg();
    els.messages.appendChild(aiW);
    const aiT = aiW.querySelector('.msg-md');
    aiT.classList.remove('typing-cursor');

    if (type === 'image') {
      aiT.innerHTML = '<p>Creating your image: <em>"' + escapeHtml(prompt) + '"</em></p><div class="media-loading">&#127912; painting pixels...</div>';
    } else {
      aiT.innerHTML = '<p>Creating your video: <em>"' + escapeHtml(prompt) + '"</em></p><div class="media-loading">&#127909; rendering video...</div>';
    }
    els.chat.scrollTop = els.chat.scrollHeight;

    const url = type === 'image' ? buildImageUrl(prompt) : buildVideoUrl(prompt);

    if (type === 'image') {
      return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
          const markdown = 'Here is your image:\n\n![' + prompt.replace(/[\[\]]/g, '') + '](' + url + ')';
          aiT.innerHTML = marked.parse(markdown);
          c.messages.push({ role: 'assistant', content: markdown });
          resolve(markdown);
        };
        img.onerror = () => {
          const err = 'Could not generate image. Please try again.';
          aiT.innerHTML = '<p>' + err + '</p>';
          c.messages.push({ role: 'assistant', content: err });
          resolve(err);
        };
        img.src = url;
      });
    } else {
      return new Promise((resolve) => {
        setTimeout(() => {
          const markdown = 'Here is your video:\n\n<video controls autoplay loop muted playsinline style="max-width:100%;border-radius:12px">\n<source src="' + url + '" type="video/mp4">\n</video>';
          aiT.innerHTML = markdown;
          c.messages.push({ role: 'assistant', content: markdown });
          resolve(markdown);
        }, 500);
      });
    }
  }

  els.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isStreaming) return;
    const text = els.input.value.trim();
    if (!text) return;

    if (synth) synth.cancel();

    let c = currentChat();
    if (!c) {
      const chat = { id: null, title: 'New chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
      const row = await dbInsert(chat);
      if (!row) return;
      chats.unshift(chat);
      activeId = chat.id;
      c = chat;
      renderSidebar();
    }

    c.messages.push({ role: 'user', content: text });
    if (c.title === 'New chat') c.title = text.length > 40 ? text.slice(0, 40) + '...' : text;
    c.updatedAt = Date.now();
    if (isGuest) saveGuestChats();

    els.input.value = '';
    els.input.style.height = 'auto';
    renderSidebar();
    els.headerTitle.textContent = c.title;
    els.emptyState.classList.add('hidden');
    els.messages.classList.remove('hidden');
    els.messages.appendChild(makeUserMsg(text));

    isStreaming = true;
    updateSendState();

    const detected = await detectIntent(text);

    if (detected.intent === 'image' || detected.intent === 'video') {
      await handleMediaGeneration(c, detected.prompt, detected.intent);
      await dbUpdate(c);
      if (isGuest) saveGuestChats();
      renderSidebar();
      isStreaming = false;
      updateSendState();
      els.input.focus();
      return;
    }

    const aiW = makeAiMsg();
    els.messages.appendChild(aiW);
    const aiT = aiW.querySelector('.msg-md');
    els.chat.scrollTop = els.chat.scrollHeight;

    let aiText = '';
    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ messages: c.messages }),
      });
      if (!res.ok) throw new Error('Server ' + res.status);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        aiText += decoder.decode(value, { stream: true });
        aiT.textContent = aiText;
        els.chat.scrollTop = els.chat.scrollHeight;
      }
    } catch (err) {
      aiText = 'Error: ' + err.message;
      aiT.textContent = aiText;
    }

    aiT.classList.remove('typing-cursor');
    aiT.innerHTML = marked.parse(aiText);
    enhanceCodeBlocks(aiT);
    aiT.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));

    if (speakReplies) speakText(aiText);

    c.messages.push({ role: 'assistant', content: aiText });
    c.updatedAt = Date.now();
    await dbUpdate(c);
    if (isGuest) saveGuestChats();
    renderSidebar();
    els.chat.scrollTop = els.chat.scrollHeight;

    isStreaming = false;
    updateSendState();
    els.input.focus();
  });

  setAuthMode('login');
  console.log('App ready');
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


def build_messages(req_messages, history_size):
    recent = [m.model_dump() for m in req_messages][-history_size:]
    return [
        {"role": "system", "content": "You are DamperBot, a friendly and helpful AI assistant. Use markdown when useful. Be thorough but clear."},
    ] + recent


def stream_model(model, messages, max_tokens):
    completion = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=max_tokens,
    )
    for chunk in completion:
        token = chunk.choices[0].delta.content
        if token:
            yield token


@app.post("/api/chat")
async def chat(req: ChatRequest):
    def stream():
        attempts = [
            (MODEL_PRIMARY, 4096, 10),
            (MODEL_PRIMARY, 2048, 8),
            (MODEL_FALLBACK, 4096, 10),
            (MODEL_FALLBACK, 2048, 8),
            (MODEL_FALLBACK, 1024, 4),
        ]
        for model, max_tok, hist in attempts:
            try:
                messages = build_messages(req.messages, hist)
                yield from stream_model(model, messages, max_tok)
                return
            except Exception as e:
                err = str(e)
                if "413" in err or "rate" in err.lower() or "tokens" in err.lower():
                    continue
                yield "\n\n**[Error]** " + err
                return
        yield "\n\n**[Heads up]** Conversation too long. Please start a new chat."

    return StreamingResponse(stream(), media_type="text/plain")