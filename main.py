import os
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

app = FastAPI(title="DamperBot")

print("=" * 60)
print("DamperBot starting")
print("  SUPABASE_URL     :", SUPABASE_URL or "MISSING")
print("  SUPABASE_ANON_KEY:", (SUPABASE_ANON_KEY[:40] + "...") if SUPABASE_ANON_KEY else "MISSING")
print("  GROQ_API_KEY     :", "set" if os.getenv("GROQ_API_KEY") else "MISSING")
print("=" * 60)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/config")
def config():
    return {"supabase_url": SUPABASE_URL, "supabase_anon_key": SUPABASE_ANON_KEY}


HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
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
  html, body { height: 100%; margin: 0; }
  body { font-family: 'Inter', system-ui, -apple-system, sans-serif; -webkit-font-smoothing: antialiased; }
  .scroll-thin::-webkit-scrollbar { width: 6px; height: 6px; }
  .scroll-thin::-webkit-scrollbar-thumb { background: rgba(120,120,120,0.25); border-radius: 3px; }
  .msg-md { line-height: 1.7; font-size: 15px; word-wrap: break-word; }
  .msg-md p { margin: 0 0 0.8em 0; }
  .msg-md ul, .msg-md ol { margin: 0.5em 0 0.8em 1.5em; }
  .msg-md code { font-family: monospace; font-size: 0.88em; padding: 0.15em 0.4em; border-radius: 4px; background: rgba(249,115,22,0.12); color: #c2410c; }
  .dark .msg-md code { background: rgba(251,146,60,0.15); color: #fdba74; }
  .msg-md pre { margin: 0.8em 0; padding: 1em; border-radius: 10px; overflow-x: auto; background: #0d1117; border: 1px solid rgba(255,255,255,0.06); }
  .msg-md pre code { padding: 0; background: transparent; color: #e6edf3; font-size: 13px; }
  .msg-md a { color: #ea580c; text-decoration: underline; }
  .dark .msg-md a { color: #fb923c; }
  .msg-md img { max-width: 100%; border-radius: 12px; margin: 0.5em 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .typing-cursor::after { content: "\258D"; color: #f97316; animation: blink 1s step-start infinite; margin-left: 1px; }
  @keyframes blink { 50% { opacity: 0; } }
  .grad-text { background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
  .logo-grad { background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%); }
  textarea { resize: none; }
  .fade-in { animation: fadeIn 0.3s ease-out; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; animation: spin 0.6s linear infinite; display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #fatal { display: none; padding: 40px; text-align: center; font-family: Inter, sans-serif; color: #ef4444; }
  .mic-active { color: #f97316 !important; }
  #micBtn { position: relative; }

  /* Image generation loading shimmer */
  .img-loading {
    width: 100%; max-width: 512px; aspect-ratio: 1/1; border-radius: 12px;
    background: linear-gradient(90deg, #1a1a1a 0%, #2a2a2a 50%, #1a1a1a 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    display: flex; align-items: center; justify-content: center;
    color: #666; font-size: 13px;
  }
  @keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
</style>
</head>
<body class="bg-gray-50 dark:bg-[#0a0a0a] text-gray-900 dark:text-gray-100">

<div id="fatal"></div>

<!-- AUTH -->
<div id="authScreen" class="hidden fixed inset-0 z-50 bg-gray-50 dark:bg-[#0a0a0a] flex items-center justify-center p-5">
  <div class="w-full max-w-[420px] fade-in">
    <div class="flex flex-col items-center mb-6">
      <div class="w-16 h-16 rounded-2xl logo-grad flex items-center justify-center text-white text-3xl mb-3">&#129302;</div>
      <h1 class="text-2xl font-bold tracking-tight">Welcome to <span class="grad-text">DamperBot</span></h1>
      <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">Sign in to save your chats</p>
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
    </div>
  </div>
</div>

<!-- APP -->
<div id="app" class="hidden flex h-screen overflow-hidden">
  <aside class="w-[280px] h-full flex flex-col bg-white dark:bg-[#111111] border-r border-gray-200 dark:border-[#1a1a1a] shrink-0">
    <div class="p-4 flex items-center gap-2.5">
      <div class="w-9 h-9 rounded-xl logo-grad flex items-center justify-center text-white text-lg">&#129302;</div>
      <div class="min-w-0 flex-1">
        <div class="font-bold text-[15px] leading-tight">DamperBot</div>
        <div class="text-[11px] text-gray-500">AI Assistant</div>
      </div>
    </div>
    <div class="px-3 pb-2">
      <button id="newChatBtn" class="w-full flex items-center justify-center gap-2 px-3.5 py-2.5 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-semibold text-sm transition">
        + New chat
      </button>
    </div>
    <div id="chatList" class="flex-1 overflow-y-auto scroll-thin px-2 pb-3 space-y-0.5"></div>
    <div class="p-3 border-t border-gray-200 dark:border-[#1a1a1a] flex items-center gap-2">
      <div id="userAvatar" class="w-8 h-8 rounded-full logo-grad flex items-center justify-center text-white text-xs font-bold">?</div>
      <div class="min-w-0 flex-1">
        <div id="userEmail" class="text-[12px] font-medium truncate">-</div>
      </div>
      <button id="logoutBtn" class="p-2 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/30 text-gray-500 hover:text-red-500 transition" title="Log out">
        <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/></svg>
      </button>
    </div>
  </aside>

  <main class="flex-1 flex flex-col min-w-0 bg-gray-50 dark:bg-[#0a0a0a]">
    <header class="h-14 flex items-center gap-2 px-4 border-b border-gray-200 dark:border-[#1a1a1a]">
      <span class="font-semibold text-[15px] truncate" id="headerTitle">DamperBot</span>
      <span id="connStatus" class="ml-auto text-[11px] px-2 py-0.5 rounded-full bg-green-100 dark:bg-green-950/40 text-green-700 dark:text-green-400">ready</span>
    </header>

    <div id="chat" class="flex-1 overflow-y-auto scroll-thin">
      <div id="emptyState" class="min-h-full flex flex-col items-center justify-center text-center px-6 py-10">
        <div class="w-20 h-20 rounded-3xl logo-grad flex items-center justify-center text-white text-4xl mb-5 shadow-2xl shadow-orange-500/30">&#129302;</div>
        <h1 class="text-3xl font-extrabold mb-2">Hi, I'm <span class="grad-text">DamperBot</span></h1>
        <p class="text-gray-500 dark:text-gray-400 max-w-md text-[15px] mb-8">Chat, speak, or generate images.</p>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 max-w-xl w-full px-2">
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all" data-prompt="/image a cute robot reading a book">
            <div class="text-lg mb-1.5">&#127912;</div>
            <div class="font-semibold mb-0.5">Generate an image</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">/image a cute robot reading a book</div>
          </button>
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all" data-prompt="/image a sunset over mountains, oil painting">
            <div class="text-lg mb-1.5">&#127749;</div>
            <div class="font-semibold mb-0.5">Artistic image</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">/image a sunset over mountains, oil painting</div>
          </button>
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all" data-prompt="Write a Python function to reverse a string">
            <div class="text-lg mb-1.5">&#128187;</div>
            <div class="font-semibold mb-0.5">Write code</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">Python function to reverse a string</div>
          </button>
          <button class="suggestion p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all" data-prompt="Write a haiku about the ocean at sunrise">
            <div class="text-lg mb-1.5">&#9997;&#65039;</div>
            <div class="font-semibold mb-0.5">Create</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">A haiku about the ocean</div>
          </button>
        </div>
      </div>
      <div id="messages" class="max-w-3xl mx-auto px-4 py-6 space-y-6 hidden"></div>
    </div>

    <div class="pt-2 pb-4 px-4">
      <div class="max-w-3xl mx-auto">
        <form id="form" class="flex items-end gap-2 rounded-2xl border border-gray-300 dark:border-[#262626] bg-white dark:bg-[#141414] focus-within:border-orange-500 transition px-3 py-2 shadow-lg">
          <textarea id="input" rows="1" placeholder="Message DamperBot... (try /image a cat)" class="flex-1 bg-transparent outline-none py-2 px-2 max-h-52 scroll-thin text-[15px]" autocomplete="off"></textarea>
          <button id="micBtn" type="button" title="Voice input" class="shrink-0 w-9 h-9 rounded-xl text-gray-500 hover:bg-gray-200 dark:hover:bg-[#2a2a2a] flex items-center justify-center transition">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path><path d="M19 10v2a7 7 0 0 1-14 0v-2"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
          </button>
          <button id="speakBtn" type="button" title="Toggle voice reply" class="shrink-0 w-9 h-9 rounded-xl text-gray-500 hover:bg-gray-200 dark:hover:bg-[#2a2a2a] flex items-center justify-center transition">
            <svg id="speakerOff" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><line x1="23" y1="9" x2="17" y2="15"></line><line x1="17" y1="9" x2="23" y2="15"></line></svg>
            <svg id="speakerOn" class="hidden" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"></path></svg>
          </button>
          <button id="send" type="submit" class="shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 disabled:opacity-30 text-white flex items-center justify-center transition">
            <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </form>
        <p class="text-[11px] text-center text-gray-400 dark:text-gray-600 mt-2">
          Type <code class="bg-orange-100 dark:bg-orange-950/40 text-orange-600 dark:text-orange-400 px-1.5 py-0.5 rounded">/image your prompt</code> to generate an image
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

  // ---------- State ----------
  let chats = [];
  let activeId = null;
  let isStreaming = false;
  let currentUser = null;
  let speakReplies = false;

  const $ = (id) => document.getElementById(id);
  const els = {
    authScreen: $('authScreen'), app: $('app'),
    tabLogin: $('tabLogin'), tabSignup: $('tabSignup'),
    authForm: $('authForm'), emailInput: $('emailInput'),
    passwordInput: $('passwordInput'), authError: $('authError'),
    authSubmit: $('authSubmit'), authSubmitText: $('authSubmitText'),
    userEmail: $('userEmail'), userAvatar: $('userAvatar'), logoutBtn: $('logoutBtn'),
    newChatBtn: $('newChatBtn'), chatList: $('chatList'),
    messages: $('messages'), emptyState: $('emptyState'), chat: $('chat'),
    form: $('form'), input: $('input'), send: $('send'),
    headerTitle: $('headerTitle'), connStatus: $('connStatus'),
    micBtn: $('micBtn'), speakBtn: $('speakBtn'),
    speakerOff: $('speakerOff'), speakerOn: $('speakerOn'),
  };

  function setStatus(text, color) {
    els.connStatus.textContent = text;
    els.connStatus.className = 'ml-auto text-[11px] px-2 py-0.5 rounded-full ' +
      (color === 'green' ? 'bg-green-100 dark:bg-green-950/40 text-green-700 dark:text-green-400' :
       color === 'red' ? 'bg-red-100 dark:bg-red-950/40 text-red-700 dark:text-red-400' :
       'bg-yellow-100 dark:bg-yellow-950/40 text-yellow-700 dark:text-yellow-400');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  }

  // ---------- Voice: Speech Recognition ----------
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  let recognition = null;
  let isListening = false;

  if (SR) {
    recognition = new SR();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isListening = true;
      els.micBtn.classList.add('mic-active');
      els.input.placeholder = 'Listening...';
      setStatus('listening', 'yellow');
    };
    recognition.onend = () => {
      isListening = false;
      els.micBtn.classList.remove('mic-active');
      els.input.placeholder = 'Message DamperBot... (try /image a cat)';
      setStatus('ready', 'green');
    };
    recognition.onresult = (event) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += t; else interim += t;
      }
      els.input.value = final || interim;
      els.input.dispatchEvent(new Event('input'));
      if (final) setTimeout(() => { if (els.input.value.trim()) els.form.requestSubmit(); }, 300);
    };
    recognition.onerror = (event) => {
      console.error('Speech error:', event.error);
      isListening = false;
      els.micBtn.classList.remove('mic-active');
      els.input.placeholder = 'Message DamperBot... (try /image a cat)';
      setStatus('ready', 'green');
      if (event.error === 'not-allowed') {
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

  // ---------- Voice: Speech Synthesis ----------
  const synth = window.speechSynthesis;

  els.speakBtn.addEventListener('click', () => {
    speakReplies = !speakReplies;
    els.speakerOn.classList.toggle('hidden', !speakReplies);
    els.speakerOff.classList.toggle('hidden', speakReplies);
    els.speakBtn.classList.toggle('text-orange-500', speakReplies);
    if (!speakReplies && synth) synth.cancel();
  });

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

  function speakText(text) {
    if (!speakReplies || !synth) return;
    synth.cancel();
    const clean = stripMarkdown(text).slice(0, 3000);
    const u = new SpeechSynthesisUtterance(clean);
    u.rate = 1.0; u.pitch = 1.0; u.volume = 1.0;
    const voices = synth.getVoices();
    const preferred = voices.find(v => v.name.includes('Google US English') || v.name.includes('Samantha') || v.name.includes('Microsoft Aria'));
    if (preferred) u.voice = preferred;
    synth.speak(u);
  }

  // ---------- Image Generation ----------
  function buildImageUrl(prompt) {
    const seed = Math.floor(Math.random() * 1000000);
    const encoded = encodeURIComponent(prompt);
    return `https://image.pollinations.ai/prompt/${encoded}?width=1024&height=1024&nologo=true&seed=${seed}&model=flux&enhance=true&quality=hd&private=true`;
  }

  function isImageRequest(text) {
    return /^\/(image|img|draw)\s+/i.test(text.trim());
  }

  function extractImagePrompt(text) {
    return text.trim().replace(/^\/(image|img|draw)\s+/i, '').trim();
  }

  // ---------- Auth ----------
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

  els.logoutBtn.addEventListener('click', async () => {
    if (!confirm('Log out?')) return;
    if (synth) synth.cancel();
    await sb.auth.signOut();
  });

  sb.auth.onAuthStateChange(async (event, session) => {
    if (session?.user) {
      currentUser = session.user;
      els.authScreen.classList.add('hidden');
      els.app.classList.remove('hidden');
      els.userEmail.textContent = currentUser.email;
      els.userAvatar.textContent = (currentUser.email[0] || '?').toUpperCase();
      setStatus('ready', 'green');
      await loadChats();
    } else {
      currentUser = null;
      chats = [];
      activeId = null;
      els.authScreen.classList.remove('hidden');
      els.app.classList.add('hidden');
      setAuthMode('login');
    }
  });

  // ---------- Database ----------
  async function loadChats() {
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
    const { data, error } = await sb.from('chats').insert({
      user_id: currentUser.id, title: chat.title, messages: chat.messages,
    }).select().single();
    if (error) {
      console.error(error);
      setStatus('save error', 'red');
      alert('Could not save chat.\n\nError: ' + error.message);
      return null;
    }
    chat.id = data.id;
    setStatus('ready', 'green');
    return data;
  }

  async function dbUpdate(chat) {
    const { error } = await sb.from('chats').update({ title: chat.title, messages: chat.messages }).eq('id', chat.id);
    if (error) { console.error(error); setStatus('save error', 'red'); }
  }

  async function dbDelete(id) {
    const { error } = await sb.from('chats').delete().eq('id', id);
    if (error) console.error(error);
  }

  // ---------- Chats ----------
  async function newChat() {
    if (!currentUser) return;
    const cur = currentChat();
    if (cur && cur.messages.length === 0) { els.input.focus(); return; }
    const chat = { id: null, title: 'New chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
    const row = await dbInsert(chat);
    if (!row) return;
    chats.unshift(chat);
    activeId = chat.id;
    renderSidebar();
    renderMessages();
    els.input.focus();
  }

  async function deleteChat(id) {
    await dbDelete(id);
    chats = chats.filter(c => c.id !== id);
    if (activeId === id) activeId = chats[0]?.id || null;
    renderSidebar();
    renderMessages();
  }

  function selectChat(id) {
    if (isStreaming) return;
    if (synth) synth.cancel();
    activeId = id;
    renderSidebar();
    renderMessages();
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
      row.innerHTML = '<span class="flex-1 truncate">' + escapeHtml(c.title) + '</span>' +
        '<button class="del opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-200 dark:hover:bg-[#2a2a2a] text-xs">x</button>';
      row.addEventListener('click', (e) => { if (!e.target.closest('.del')) selectChat(c.id); });
      row.querySelector('.del').addEventListener('click', (e) => { e.stopPropagation(); deleteChat(c.id); });
      els.chatList.appendChild(row);
    });
  }

  // ---------- Messages ----------
  function makeUserMsg(content) {
    const w = document.createElement('div');
    w.className = 'flex gap-3 justify-end fade-in';
    w.innerHTML = '<div class="max-w-[85%] rounded-2xl rounded-tr-md px-4 py-2.5 bg-gradient-to-br from-orange-500 to-amber-500 text-white whitespace-pre-wrap break-words text-[15px] leading-relaxed">' + escapeHtml(content) + '</div>' +
      '<div class="shrink-0 w-8 h-8 rounded-full bg-gray-200 dark:bg-[#262626] flex items-center justify-center text-[11px] font-semibold text-gray-600 dark:text-gray-300">You</div>';
    return w;
  }

  function makeAiMsg() {
    const w = document.createElement('div');
    w.className = 'flex gap-3 fade-in';
    w.innerHTML = '<div class="shrink-0 w-8 h-8 rounded-full logo-grad flex items-center justify-center text-white text-[14px]">&#129302;</div>' +
      '<div class="flex-1 min-w-0 pt-0.5"><div class="msg-md typing-cursor text-gray-800 dark:text-gray-100"></div></div>';
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
        t.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
        els.messages.appendChild(w);
      }
    });
    els.chat.scrollTop = els.chat.scrollHeight;
  }

  function updateSendState() {
    els.send.disabled = isStreaming || els.input.value.trim().length === 0;
  }
  els.input.addEventListener('input', () => {
    els.input.style.height = 'auto';
    els.input.style.height = Math.min(els.input.scrollHeight, 200) + 'px';
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

  // ---------- Image generation flow ----------
  async function handleImageGeneration(c, prompt, userText) {
    // Add AI message shell
    const aiW = makeAiMsg();
    els.messages.appendChild(aiW);
    const aiT = aiW.querySelector('.msg-md');
    aiT.classList.remove('typing-cursor');

    // Show loading state
    aiT.innerHTML = '<p>Generating your image...</p><div class="img-loading">&#127912; painting pixels...</div>';
    els.chat.scrollTop = els.chat.scrollHeight;

    const imgUrl = buildImageUrl(prompt);

    // Preload image and swap in
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        const markdown = 'Here is your image:\n\n![' + prompt.replace(/[\[\]]/g, '') + '](' + imgUrl + ')\n\n*Prompt: ' + prompt + '*';
        aiT.innerHTML = marked.parse(markdown);
        c.messages.push({ role: 'assistant', content: markdown });
        resolve(markdown);
      };
      img.onerror = () => {
        const err = 'Could not generate image. Please try again with a different prompt.';
        aiT.innerHTML = '<p>' + err + '</p>';
        c.messages.push({ role: 'assistant', content: err });
        resolve(err);
      };
      img.src = imgUrl;
    });
  }

  // ---------- Send ----------
  els.form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (isStreaming) return;
    const text = els.input.value.trim();
    if (!text) return;
    if (!currentUser) return;

    if (synth) synth.cancel();

    // Ensure there's a chat
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

    // Add user message
    c.messages.push({ role: 'user', content: text });
    if (c.title === 'New chat') c.title = text.length > 40 ? text.slice(0, 40) + '...' : text;
    c.updatedAt = Date.now();

    els.input.value = '';
    els.input.style.height = 'auto';
    renderSidebar();
    els.headerTitle.textContent = c.title;
    els.emptyState.classList.add('hidden');
    els.messages.classList.remove('hidden');
    els.messages.appendChild(makeUserMsg(text));

    // ---------- IMAGE PATH ----------
    if (isImageRequest(text)) {
      isStreaming = true;
      updateSendState();
      setStatus('generating image...', 'yellow');

      const prompt = extractImagePrompt(text);
      await handleImageGeneration(c, prompt, text);

      await dbUpdate(c);
      renderSidebar();
      isStreaming = false;
      updateSendState();
      setStatus('ready', 'green');
      els.input.focus();
      return;
    }

    // ---------- CHAT PATH ----------
    const aiW = makeAiMsg();
    els.messages.appendChild(aiW);
    const aiT = aiW.querySelector('.msg-md');
    els.chat.scrollTop = els.chat.scrollHeight;

    isStreaming = true;
    updateSendState();
    setStatus('thinking...', 'yellow');

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
    aiT.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));

    speakText(aiText);

    c.messages.push({ role: 'assistant', content: aiText });
    c.updatedAt = Date.now();
    await dbUpdate(c);
    renderSidebar();
    els.chat.scrollTop = els.chat.scrollHeight;

    isStreaming = false;
    updateSendState();
    setStatus('ready', 'green');
    els.input.focus();
  });

  setAuthMode('login');
  console.log('App ready with image generation');
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": "You are DamperBot, a friendly and helpful AI assistant. Use markdown formatting when useful."},
    ] + [m.model_dump() for m in req.messages]

    def stream():
        try:
            completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    yield token
        except Exception as e:
            yield "\n[Error: " + str(e) + "]"

    return StreamingResponse(stream(), media_type="text/plain")