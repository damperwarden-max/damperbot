import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI(title="DamperBot")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


HTML = r"""
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0" />
<meta name="theme-color" content="#0a0a0a" />
<title>DamperBot · AI Assistant</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🤖</text></svg>">
<script src="https://cdn.tailwindcss.com"></script>
<script>
  tailwind.config = {
    darkMode: 'class',
    theme: {
      extend: {
        colors: {
          brand: {
            50:'#fff7ed',100:'#ffedd5',200:'#fed7aa',300:'#fdba74',400:'#fb923c',
            500:'#f97316',600:'#ea580c',700:'#c2410c',800:'#9a3412',900:'#7c2d12',
          }
        },
        fontFamily: { sans: ['Inter','system-ui','-apple-system','Segoe UI','sans-serif'] },
        boxShadow: {
          'glow': '0 0 24px rgba(249,115,22,0.35)',
          'soft': '0 4px 24px rgba(0,0,0,0.06)',
        }
      }
    }
  }
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/atom-one-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>

<style>
  :root { --brand: 249 115 22; }
  html, body { height: 100%; margin: 0; overscroll-behavior: none; }
  body {
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }

  .scroll-thin::-webkit-scrollbar { width: 6px; height: 6px; }
  .scroll-thin::-webkit-scrollbar-thumb { background: rgba(120,120,120,0.22); border-radius: 3px; transition: background 0.15s; }
  .scroll-thin::-webkit-scrollbar-thumb:hover { background: rgba(120,120,120,0.45); }
  .scroll-thin::-webkit-scrollbar-track { background: transparent; }
  .no-scrollbar::-webkit-scrollbar { display: none; }

  /* Markdown */
  .msg-md { line-height: 1.72; font-size: 15px; word-wrap: break-word; overflow-wrap: break-word; }
  .msg-md > *:first-child { margin-top: 0 !important; }
  .msg-md > *:last-child { margin-bottom: 0 !important; }
  .msg-md p { margin: 0 0 0.9em 0; }
  .msg-md ul, .msg-md ol { margin: 0.5em 0 0.9em 1.5em; }
  .msg-md li { margin: 0.25em 0; }
  .msg-md li > p { margin: 0; }
  .msg-md h1, .msg-md h2, .msg-md h3, .msg-md h4 { margin: 1.3em 0 0.6em; font-weight: 700; line-height: 1.3; letter-spacing: -0.01em; }
  .msg-md h1 { font-size: 1.55em; }
  .msg-md h2 { font-size: 1.3em; }
  .msg-md h3 { font-size: 1.13em; }
  .msg-md h4 { font-size: 1em; }
  .msg-md strong { font-weight: 700; }
  .msg-md em { font-style: italic; }
  .msg-md code {
    font-family: 'SF Mono', ui-monospace, Menlo, Consolas, monospace;
    font-size: 0.87em;
    padding: 0.16em 0.42em;
    border-radius: 5px;
    background: rgba(249,115,22,0.12);
    color: #c2410c;
    font-weight: 500;
  }
  .dark .msg-md code { background: rgba(251,146,60,0.14); color: #fdba74; }
  .msg-md pre {
    margin: 0.9em 0;
    border-radius: 12px;
    overflow: hidden;
    background: #0d1117;
    border: 1px solid rgba(255,255,255,0.06);
    position: relative;
  }
  .msg-md pre code {
    display: block;
    padding: 1em 1.15em;
    background: transparent;
    color: #e6edf3;
    font-size: 13.2px;
    line-height: 1.65;
    overflow-x: auto;
    font-weight: 400;
  }
  .msg-md blockquote {
    border-left: 3px solid #f97316;
    padding: 0.2em 0 0.2em 1em;
    margin: 0.9em 0;
    opacity: 0.9;
    font-style: italic;
    color: inherit;
  }
  .msg-md a { color: #ea580c; text-decoration: underline; text-underline-offset: 2px; font-weight: 500; }
  .dark .msg-md a { color: #fb923c; }
  .msg-md a:hover { opacity: 0.8; }
  .msg-md hr { border: none; border-top: 1px solid rgba(127,127,127,0.18); margin: 1.4em 0; }
  .msg-md table { border-collapse: collapse; margin: 0.9em 0; display: block; overflow-x: auto; font-size: 14px; }
  .msg-md th, .msg-md td { border: 1px solid rgba(127,127,127,0.22); padding: 0.55em 0.95em; text-align: left; }
  .msg-md th { background: rgba(249,115,22,0.08); font-weight: 600; }
  .msg-md img { max-width: 100%; border-radius: 8px; margin: 0.6em 0; }
  .msg-md :not(pre) > code:before, .msg-md :not(pre) > code:after { content: ''; }

  /* Code block header */
  .code-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.5em 0.9em;
    background: rgba(255,255,255,0.03);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 11.5px;
    color: #8b949e;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
  }
  .code-copy {
    display: inline-flex; align-items: center; gap: 4px;
    color: #8b949e; background: transparent; border: none; cursor: pointer;
    padding: 4px 8px; border-radius: 6px;
    transition: all 0.15s; font-size: 11.5px; font-weight: 500;
  }
  .code-copy:hover { background: rgba(255,255,255,0.08); color: #e6edf3; }
  .code-copy.copied { color: #4ade80; }

  /* Message actions */
  .msg-actions {
    opacity: 0; transition: opacity 0.15s;
    display: flex; gap: 4px; margin-top: 8px;
  }
  .msg-row:hover .msg-actions { opacity: 1; }
  .msg-btn {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 8px; border-radius: 6px; font-size: 11.5px;
    color: #6b7280; background: transparent;
    border: 1px solid transparent; cursor: pointer;
    transition: all 0.15s; font-weight: 500;
  }
  .dark .msg-btn { color: #9ca3af; }
  .msg-btn:hover {
    background: rgba(249,115,22,0.08);
    color: #ea580c;
    border-color: rgba(249,115,22,0.2);
  }
  .dark .msg-btn:hover { color: #fb923c; }

  /* Typing cursor */
  .typing-cursor::after {
    content: "▍";
    display: inline-block;
    margin-left: 1px;
    color: #f97316;
    animation: blink 1s step-start infinite;
    font-weight: 400;
  }
  @keyframes blink { 50% { opacity: 0; } }

  /* Fade-in animations */
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  @keyframes slideDown {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  @keyframes shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
  }
  .fade-in { animation: fadeInUp 0.32s cubic-bezier(0.16, 1, 0.3, 1); }
  .fade-in-fast { animation: fadeIn 0.2s ease-out; }
  .slide-down { animation: slideDown 0.24s cubic-bezier(0.16, 1, 0.3, 1); }

  textarea { resize: none; }

  .grad-text {
    background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .logo-grad { background: linear-gradient(135deg, #f97316 0%, #fbbf24 100%); }
  .glow-border:focus-within {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 4px rgba(249,115,22,0.12);
  }

  /* Sidebar chat item */
  .chat-item { transition: all 0.15s cubic-bezier(0.16, 1, 0.3, 1); }

  /* Toast */
  .toast {
    position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%);
    background: rgba(20,20,20,0.96); backdrop-filter: blur(12px);
    color: white; padding: 12px 18px; border-radius: 12px;
    font-size: 13.5px; font-weight: 500;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.08);
    z-index: 100; opacity: 0; pointer-events: none;
    transition: opacity 0.25s, transform 0.25s;
    display: flex; align-items: center; gap: 8px;
    max-width: 90vw;
  }
  .toast.show { opacity: 1; transform: translateX(-50%) translateY(-4px); }
  .toast.success { border-left: 3px solid #4ade80; }
  .toast.error { border-left: 3px solid #ef4444; }

  /* Skip transition on first paint */
  .no-transition, .no-transition * { transition: none !important; }

  /* Scroll indicator */
  .scroll-btn {
    position: absolute; bottom: 20px; right: 24px;
    width: 36px; height: 36px; border-radius: 50%;
    background: white; border: 1px solid rgba(0,0,0,0.08);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    display: flex; align-items: center; justify-content: center;
    cursor: pointer; transition: all 0.2s;
    opacity: 0; pointer-events: none;
  }
  .dark .scroll-btn { background: #1f1f1f; border-color: rgba(255,255,255,0.08); }
  .scroll-btn.visible { opacity: 1; pointer-events: auto; }
  .scroll-btn:hover { transform: translateY(-2px); }

  /* Shimmer skeleton */
  .shimmer {
    background: linear-gradient(90deg, rgba(127,127,127,0.1) 0%, rgba(127,127,127,0.2) 50%, rgba(127,127,127,0.1) 100%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
    border-radius: 8px;
  }

  /* Sidebar animation */
  @media (max-width: 767px) {
    #sidebar { will-change: transform; }
  }

  /* Hide scroll-btn transition on init */
  .scroll-btn.init { transition: none; }
</style>
</head>
<body class="bg-gray-50 dark:bg-[#0a0a0a] text-gray-900 dark:text-gray-100">

<div class="flex h-screen overflow-hidden">

  <!-- ==================== SIDEBAR ==================== -->
  <aside id="sidebar" class="fixed md:relative z-40 w-[280px] h-full flex flex-col bg-white dark:bg-[#111111] border-r border-gray-200 dark:border-[#1a1a1a] transition-transform duration-300 ease-out -translate-x-full md:translate-x-0">

    <!-- Brand -->
    <div class="p-4 flex items-center gap-2.5">
      <div class="w-9 h-9 rounded-xl logo-grad flex items-center justify-center text-white text-lg shadow-lg shadow-orange-500/25">🤖</div>
      <div class="min-w-0 flex-1">
        <div class="font-bold text-[15px] leading-tight tracking-tight">DamperBot</div>
        <div class="text-[11px] text-gray-500 dark:text-gray-500 leading-tight">AI Assistant</div>
      </div>
      <button id="closeSidebarBtn" class="md:hidden p-1.5 rounded-lg hover:bg-gray-100 dark:hover:bg-[#1f1f1f] transition text-gray-500">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>

    <!-- New chat -->
    <div class="px-3 pt-1 pb-2">
      <button id="newChatBtn" class="w-full flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 text-white font-semibold text-sm transition shadow-lg shadow-orange-500/20 hover:shadow-orange-500/30 active:scale-[0.98]">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New chat
        <kbd class="ml-auto text-[10px] font-mono bg-white/20 px-1.5 py-0.5 rounded">Ctrl K</kbd>
      </button>
    </div>

    <!-- Search -->
    <div class="px-3 pb-2">
      <div class="relative">
        <svg class="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input id="searchInput" type="text" placeholder="Search chats..." class="w-full pl-8 pr-3 py-2 text-[13px] rounded-lg bg-gray-100 dark:bg-[#1a1a1a] border border-transparent focus:border-orange-400 dark:focus:border-orange-500/50 outline-none transition">
      </div>
    </div>

    <!-- Chat list -->
    <div id="chatList" class="flex-1 overflow-y-auto scroll-thin px-2 pb-3 space-y-0.5"></div>

    <!-- Bottom bar -->
    <div class="p-3 border-t border-gray-200 dark:border-[#1a1a1a] flex items-center justify-between gap-2">
      <div class="flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-gray-100 dark:bg-[#1a1a1a] flex-1 min-w-0">
        <div class="w-6 h-6 rounded-full logo-grad flex items-center justify-center text-white text-[10px] font-bold shrink-0 shadow-sm">D</div>
        <span class="text-xs font-medium truncate">You</span>
      </div>
      <button id="themeToggle" class="shrink-0 w-9 h-9 rounded-lg hover:bg-gray-100 dark:hover:bg-[#1a1a1a] flex items-center justify-center transition text-gray-600 dark:text-gray-400" title="Toggle theme">
        <svg id="sunIcon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/></svg>
        <svg id="moonIcon" xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="hidden"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      </button>
    </div>
  </aside>

  <div id="overlay" class="fixed inset-0 bg-black/60 backdrop-blur-sm z-30 hidden md:hidden"></div>

  <!-- ==================== MAIN ==================== -->
  <main class="flex-1 flex flex-col min-w-0 bg-gray-50 dark:bg-[#0a0a0a] relative">

    <!-- Header -->
    <header class="h-14 flex items-center gap-2 px-4 border-b border-gray-200 dark:border-[#1a1a1a] bg-white/80 dark:bg-[#0a0a0a]/80 backdrop-blur-md z-20 shrink-0">
      <button id="menuBtn" class="md:hidden p-2 -ml-1 rounded-lg hover:bg-gray-100 dark:hover:bg-[#1a1a1a]">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <div class="flex items-center gap-2.5 min-w-0 flex-1">
        <span class="font-semibold text-[15px] truncate" id="headerTitle">DamperBot</span>
        <span class="hidden sm:inline-flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-full bg-orange-50 dark:bg-orange-950/40 text-orange-600 dark:text-orange-400 font-medium border border-orange-200 dark:border-orange-900/50">
          <span class="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></span>
          online
        </span>
      </div>
      <button id="exportBtn" class="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-[#1a1a1a] text-gray-500 transition" title="Export chat">
        <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      </button>
    </header>

    <!-- Chat scroll area -->
    <div id="chat" class="flex-1 overflow-y-auto scroll-thin relative">
      <!-- Empty state -->
      <div id="emptyState" class="min-h-full flex flex-col items-center justify-center text-center px-6 py-10">
        <div class="w-20 h-20 rounded-3xl logo-grad flex items-center justify-center text-white text-4xl mb-5 shadow-2xl shadow-orange-500/30 fade-in">🤖</div>
        <h1 class="text-3xl sm:text-4xl font-extrabold mb-2 tracking-tight fade-in">Hi, I'm <span class="grad-text">DamperBot</span></h1>
        <p class="text-gray-500 dark:text-gray-400 max-w-md text-[15px] fade-in">Your AI assistant. Ask me anything — code, writing, ideas, and more.</p>

        <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 mt-10 max-w-xl w-full px-2">
          <button class="suggestion group p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all fade-in active:scale-[0.98]" data-prompt="Write a Python function to reverse a string">
            <div class="text-lg mb-1.5">💻</div>
            <div class="font-semibold mb-0.5">Write code</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">Python function to reverse a string</div>
          </button>
          <button class="suggestion group p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all fade-in active:scale-[0.98]" data-prompt="Explain quantum computing like I'm 10 years old">
            <div class="text-lg mb-1.5">🧠</div>
            <div class="font-semibold mb-0.5">Explain</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">Quantum computing like I'm 10</div>
          </button>
          <button class="suggestion group p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all fade-in active:scale-[0.98]" data-prompt="Give me a 7-day home workout plan with no equipment">
            <div class="text-lg mb-1.5">💪</div>
            <div class="font-semibold mb-0.5">Plan</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">7-day home workout plan</div>
          </button>
          <button class="suggestion group p-4 text-left text-sm rounded-2xl border border-gray-200 dark:border-[#1f1f1f] hover:border-orange-400 dark:hover:border-orange-500/50 hover:bg-orange-50/50 dark:hover:bg-orange-950/10 transition-all fade-in active:scale-[0.98]" data-prompt="Write a haiku about the ocean at sunrise">
            <div class="text-lg mb-1.5">✍️</div>
            <div class="font-semibold mb-0.5">Create</div>
            <div class="text-xs text-gray-500 dark:text-gray-500">A haiku about the ocean</div>
          </button>
        </div>
      </div>

      <div id="messages" class="max-w-3xl mx-auto px-4 py-6 space-y-7 hidden"></div>

      <!-- Scroll to bottom button -->
      <button id="scrollBtn" class="scroll-btn init" title="Scroll to bottom">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
      </button>
    </div>

    <!-- Input -->
    <div class="bg-gradient-to-t from-gray-50 dark:from-[#0a0a0a] via-gray-50 dark:via-[#0a0a0a] to-transparent pt-2 pb-4 px-4 shrink-0">
      <div class="max-w-3xl mx-auto">
        <form id="form" class="flex items-end gap-2 rounded-2xl border border-gray-300 dark:border-[#262626] bg-white dark:bg-[#141414] glow-border transition px-3 py-2 shadow-lg shadow-black/[0.03] dark:shadow-black/30">
          <textarea id="input" rows="1" placeholder="Message DamperBot..." class="flex-1 bg-transparent outline-none py-2 px-2 max-h-52 scroll-thin text-[15px]" autocomplete="off" spellcheck="true"></textarea>
          <button id="send" type="submit" class="shrink-0 w-9 h-9 rounded-xl bg-gradient-to-br from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 disabled:opacity-30 disabled:cursor-not-allowed text-white flex items-center justify-center transition shadow-md shadow-orange-500/25 disabled:shadow-none active:scale-95">
            <svg xmlns="http://www.w3.org/2000/svg" width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </form>
        <p class="text-[11px] text-center text-gray-400 dark:text-gray-600 mt-2.5">DamperBot can make mistakes. Verify important information.</p>
      </div>
    </div>

  </main>
</div>

<div id="toast" class="toast"></div>

<script>
// ============================================================
// Config & state
// ============================================================
const STORAGE_KEY = 'damperbot_chats_v3';
const ACTIVE_KEY  = 'damperbot_active_v3';
const THEME_KEY   = 'damperbot_theme_v3';

let chats = [];
let activeId = null;
let isStreaming = false;
let searchQuery = '';
let userAtBottom = true;

try {
  chats = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  activeId = localStorage.getItem(ACTIVE_KEY) || null;
} catch { chats = []; }

const $ = id => document.getElementById(id);
const els = {
  sidebar: $('sidebar'), overlay: $('overlay'), menuBtn: $('menuBtn'),
  closeSidebarBtn: $('closeSidebarBtn'),
  newChatBtn: $('newChatBtn'), chatList: $('chatList'), searchInput: $('searchInput'),
  messages: $('messages'), emptyState: $('emptyState'), chat: $('chat'),
  form: $('form'), input: $('input'), send: $('send'),
  headerTitle: $('headerTitle'), themeToggle: $('themeToggle'),
  sunIcon: $('sunIcon'), moonIcon: $('moonIcon'),
  toast: $('toast'), scrollBtn: $('scrollBtn'), exportBtn: $('exportBtn'),
};

// ============================================================
// Toast
// ============================================================
let toastTimer = null;
function toast(msg, type = '') {
  els.toast.textContent = msg;
  els.toast.className = 'toast show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2200);
}

// ============================================================
// Theme
// ============================================================
function applyTheme() {
  const t = localStorage.getItem(THEME_KEY) || 'dark';
  document.documentElement.classList.toggle('dark', t === 'dark');
  els.sunIcon.classList.toggle('hidden', t !== 'dark');
  els.moonIcon.classList.toggle('hidden', t === 'dark');
}
els.themeToggle.addEventListener('click', () => {
  const cur = localStorage.getItem(THEME_KEY) || 'dark';
  localStorage.setItem(THEME_KEY, cur === 'dark' ? 'light' : 'dark');
  applyTheme();
});
applyTheme();

// ============================================================
// Sidebar toggle
// ============================================================
function openSidebar() {
  els.sidebar.classList.remove('-translate-x-full');
  els.overlay.classList.remove('hidden');
}
function closeSidebar() {
  els.sidebar.classList.add('-translate-x-full');
  els.overlay.classList.add('hidden');
}
els.menuBtn.addEventListener('click', openSidebar);
els.closeSidebarBtn.addEventListener('click', closeSidebar);
els.overlay.addEventListener('click', closeSidebar);

// ============================================================
// Storage
// ============================================================
function saveChats() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch (e) {
    toast('⚠️ Storage full — delete some chats', 'error');
  }
}
function saveActive() { localStorage.setItem(ACTIVE_KEY, activeId || ''); }
function currentChat() { return chats.find(c => c.id === activeId); }
function uid() { return 'c_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8); }

// ============================================================
// Chat management
// ============================================================
function newChat() {
  // If current chat is empty, just keep it
  const cur = currentChat();
  if (cur && cur.messages.length === 0) {
    els.input.focus();
    return;
  }
  const chat = { id: uid(), title: 'New chat', messages: [], createdAt: Date.now(), updatedAt: Date.now() };
  chats.unshift(chat);
  activeId = chat.id;
  saveChats(); saveActive();
  renderSidebar();
  renderMessages();
  els.input.focus();
  if (window.innerWidth < 768) closeSidebar();
}

function deleteChat(id) {
  chats = chats.filter(c => c.id !== id);
  if (activeId === id) activeId = chats[0]?.id || null;
  saveChats(); saveActive();
  renderSidebar();
  renderMessages();
  toast('Chat deleted', 'success');
}

function selectChat(id) {
  if (isStreaming) { toast('Wait for reply to finish'); return; }
  activeId = id;
  saveActive();
  renderSidebar();
  renderMessages();
  if (window.innerWidth < 768) closeSidebar();
}

function renameChat(id) {
  const c = chats.find(x => x.id === id);
  if (!c) return;
  const name = prompt('Rename chat:', c.title);
  if (name && name.trim()) {
    c.title = name.trim();
    saveChats();
    renderSidebar();
    if (activeId === id) els.headerTitle.textContent = c.title;
  }
}

function exportChat() {
  const c = currentChat();
  if (!c || c.messages.length === 0) { toast('Nothing to export'); return; }
  let md = `# ${c.title}\n\n`;
  c.messages.forEach(m => {
    md += `**${m.role === 'user' ? 'You' : 'DamperBot'}:**\n\n${m.content}\n\n---\n\n`;
  });
  const blob = new Blob([md], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${c.title.replace(/[^a-z0-9]/gi, '_')}.md`;
  a.click();
  URL.revokeObjectURL(url);
  toast('Exported as Markdown', 'success');
}

// ============================================================
// Sidebar render
// ============================================================
function renderSidebar() {
  const q = searchQuery.toLowerCase().trim();
  const filtered = q ? chats.filter(c =>
    c.title.toLowerCase().includes(q) ||
    c.messages.some(m => m.content.toLowerCase().includes(q))
  ) : chats;

  els.chatList.innerHTML = '';

  if (filtered.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'text-xs text-gray-400 dark:text-gray-600 px-3 py-6 text-center';
    empty.textContent = q ? 'No matches' : 'No chats yet. Start one!';
    els.chatList.appendChild(empty);
    return;
  }

  const now = Date.now();
  const day = 86400000;
  const groups = { today: [], yesterday: [], week: [], older: [] };
  filtered.forEach(c => {
    const age = now - (c.updatedAt || c.createdAt);
    if (age < day) groups.today.push(c);
    else if (age < 2 * day) groups.yesterday.push(c);
    else if (age < 7 * day) groups.week.push(c);
    else groups.older.push(c);
  });

  const labels = { today: 'Today', yesterday: 'Yesterday', week: 'Previous 7 days', older: 'Older' };
  Object.keys(groups).forEach(key => {
    if (!groups[key].length) return;
    const label = document.createElement('div');
    label.className = 'text-[10px] font-bold uppercase tracking-wider text-gray-400 dark:text-gray-600 px-3 pt-3 pb-1';
    label.textContent = labels[key];
    els.chatList.appendChild(label);

    groups[key].forEach(c => {
      const row = document.createElement('div');
      const active = c.id === activeId;
      row.className = 'chat-item group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer text-[13px] ' +
        (active
          ? 'bg-orange-50 dark:bg-orange-950/30 text-orange-700 dark:text-orange-300 font-medium'
          : 'hover:bg-gray-100 dark:hover:bg-[#1a1a1a] text-gray-700 dark:text-gray-300');
      row.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="shrink-0 ${active ? 'text-orange-500' : 'opacity-60'}"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="flex-1 truncate">${escapeHtml(c.title || 'New chat')}</span>
        <button class="del opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-gray-200 dark:hover:bg-[#2a2a2a] transition shrink-0" title="Delete">
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>
        </button>`;
      row.addEventListener('click', (e) => {
        if (e.target.closest('.del')) return;
        selectChat(c.id);
      });
      row.addEventListener('dblclick', () => renameChat(c.id));
      row.querySelector('.del').addEventListener('click', (e) => {
        e.stopPropagation();
        deleteChat(c.id);
      });
      els.chatList.appendChild(row);
    });
  });
}

// ============================================================
// Helpers
// ============================================================
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, m => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[m]));
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => toast('Copied!', 'success'))
    .catch(() => toast('Copy failed', 'error'));
}

// Add copy buttons to code blocks after rendering
function enhanceCodeBlocks(container) {
  container.querySelectorAll('pre').forEach(pre => {
    if (pre.dataset.enhanced) return;
    pre.dataset.enhanced = '1';
    const code = pre.querySelector('code');
    if (!code) return;
    const lang = (code.className.match(/language-(\w+)/) || [])[1] || 'code';
    const header = document.createElement('div');
    header.className = 'code-header';
    header.innerHTML = `<span>${lang}</span>`;
    const btn = document.createElement('button');
    btn.className = 'code-copy';
    btn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span>Copy</span>`;
    btn.addEventListener('click', () => {
      copyToClipboard(code.innerText);
      btn.classList.add('copied');
      btn.querySelector('span').textContent = 'Copied';
      setTimeout(() => {
        btn.classList.remove('copied');
        btn.querySelector('span').textContent = 'Copy';
      }, 1500);
    });
    header.appendChild(btn);
    pre.insertBefore(header, code);
  });
}

// ============================================================
// Message rendering
// ============================================================
function makeUserMsg(content) {
  const wrap = document.createElement('div');
  wrap.className = 'flex gap-3 justify-end fade-in msg-row';
  wrap.innerHTML = `
    <div class="max-w-[85%] rounded-2xl rounded-tr-md px-4 py-2.5 bg-gradient-to-br from-orange-500 to-amber-500 text-white whitespace-pre-wrap break-words shadow-md shadow-orange-500/15 text-[15px] leading-relaxed">${escapeHtml(content)}</div>
    <div class="shrink-0 w-8 h-8 rounded-full bg-gray-200 dark:bg-[#262626] flex items-center justify-center text-[11px] font-semibold text-gray-600 dark:text-gray-300">You</div>`;
  return wrap;
}

function makeAiMsg() {
  const wrap = document.createElement('div');
  wrap.className = 'flex gap-3 fade-in msg-row';
  wrap.innerHTML = `
    <div class="shrink-0 w-8 h-8 rounded-full logo-grad flex items-center justify-center text-white text-[14px] shadow-md shadow-orange-500/25">🤖</div>
    <div class="flex-1 min-w-0 pt-0.5">
      <div class="msg-md typing-cursor text-gray-800 dark:text-gray-100"></div>
      <div class="msg-actions"></div>
    </div>`;
  return wrap;
}

function attachAiActions(wrap, content) {
  const actions = wrap.querySelector('.msg-actions');
  actions.innerHTML = '';

  const copyBtn = document.createElement('button');
  copyBtn.className = 'msg-btn';
  copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy`;
  copyBtn.addEventListener('click', () => copyToClipboard(content));
  actions.appendChild(copyBtn);

  const regenBtn = document.createElement('button');
  regenBtn.className = 'msg-btn';
  regenBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>Regenerate`;
  regenBtn.addEventListener('click', () => regenerate());
  actions.appendChild(regenBtn);
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
  els.headerTitle.textContent = c.title || 'DamperBot';

  c.messages.forEach((m, i) => {
    if (m.role === 'user') {
      els.messages.appendChild(makeUserMsg(m.content));
    } else {
      const wrap = makeAiMsg();
      const target = wrap.querySelector('.msg-md');
      target.classList.remove('typing-cursor');
      target.innerHTML = marked.parse(m.content || '');
      enhanceCodeBlocks(target);
      target.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
      if (i === c.messages.length - 1) attachAiActions(wrap, m.content);
      els.messages.appendChild(wrap);
    }
  });
  scrollToBottom(false);
}

// ============================================================
// Scroll handling
// ============================================================
function scrollToBottom(smooth = true) {
  requestAnimationFrame(() => {
    if (smooth) els.chat.scrollTo({ top: els.chat.scrollHeight, behavior: 'smooth' });
    else els.chat.scrollTop = els.chat.scrollHeight;
  });
}
els.chat.addEventListener('scroll', () => {
  const distanceFromBottom = els.chat.scrollHeight - els.chat.scrollTop - els.chat.clientHeight;
  userAtBottom = distanceFromBottom < 80;
  els.scrollBtn.classList.toggle('visible', !userAtBottom);
  if (els.scrollBtn.classList.contains('init')) els.scrollBtn.classList.remove('init');
});
els.scrollBtn.addEventListener('click', () => scrollToBottom());

// ============================================================
// Input
// ============================================================
function updateSendState() {
  els.send.disabled = isStreaming || els.input.value.trim().length === 0;
}
els.input.addEventListener('input', () => {
  els.input.style.height = 'auto';
  els.input.style.height = Math.min(els.input.scrollHeight, 208) + 'px';
  updateSendState();
});
els.input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    els.form.requestSubmit();
  }
});
document.querySelectorAll('.suggestion').forEach(b => {
  b.addEventListener('click', () => {
    els.input.value = b.dataset.prompt;
    els.input.dispatchEvent(new Event('input'));
    els.form.requestSubmit();
  });
});

// ============================================================
// Keyboard shortcuts
// ============================================================
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    newChat();
  }
  if (e.key === 'Escape') {
    if (window.innerWidth < 768) closeSidebar();
  }
  if ((e.ctrlKey || e.metaKey) && e.key === '/') {
    e.preventDefault();
    els.input.focus();
  }
});

// ============================================================
// Search
// ============================================================
els.searchInput.addEventListener('input', (e) => {
  searchQuery = e.target.value;
  renderSidebar();
});

// ============================================================
// Send message
// ============================================================
async function sendMessage(text, opts = {}) {
  if (isStreaming) return;
  if (!currentChat()) newChat();
  const c = currentChat();

  if (!opts.isRegen) {
    c.messages.push({ role: 'user', content: text });
    if (c.title === 'New chat' || !c.title) {
      c.title = text.length > 42 ? text.slice(0, 42) + '…' : text;
    }
  }
  c.updatedAt = Date.now();
  saveChats();

  els.input.value = '';
  els.input.style.height = 'auto';
  renderSidebar();
  els.headerTitle.textContent = c.title;

  els.emptyState.classList.add('hidden');
  els.messages.classList.remove('hidden');
  if (!opts.isRegen) els.messages.appendChild(makeUserMsg(text));

  const aiWrap = makeAiMsg();
  els.messages.appendChild(aiWrap);
  const aiTarget = aiWrap.querySelector('.msg-md');
  scrollToBottom();

  isStreaming = true;
  updateSendState();

  let aiText = '';
  let pending = '';
  let rafScheduled = false;

  function flush() {
    aiTarget.textContent = aiText;
    if (userAtBottom) scrollToBottom(false);
    rafScheduled = false;
  }

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: c.messages }),
    });
    if (!res.ok) throw new Error('Server error ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      aiText += decoder.decode(value, { stream: true });
      if (!rafScheduled) {
        rafScheduled = true;
        requestAnimationFrame(flush);
      }
    }
  } catch (err) {
    aiText = '⚠️ Error: ' + err.message;
    aiTarget.textContent = aiText;
  }

  // Finalize with markdown
  aiTarget.classList.remove('typing-cursor');
  aiTarget.innerHTML = marked.parse(aiText);
  enhanceCodeBlocks(aiTarget);
  aiTarget.querySelectorAll('pre code').forEach(el => hljs.highlightElement(el));
  attachAiActions(aiWrap, aiText);

  c.messages.push({ role: 'assistant', content: aiText });
  c.updatedAt = Date.now();
  saveChats();
  renderSidebar();
  if (userAtBottom) scrollToBottom();

  isStreaming = false;
  updateSendState();
  els.input.focus();
}

async function regenerate() {
  const c = currentChat();
  if (!c) return;
  // Remove last assistant message
  while (c.messages.length && c.messages[c.messages.length - 1].role === 'assistant') {
    c.messages.pop();
  }
  saveChats();
  renderMessages();
  // Find last user message to resend
  const lastUser = [...c.messages].reverse().find(m => m.role === 'user');
  if (!lastUser) return;
  await sendMessage(lastUser.content, { isRegen: true });
}

els.form.addEventListener('submit', (e) => {
  e.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;
  sendMessage(text);
});

els.newChatBtn.addEventListener('click', newChat);
els.exportBtn.addEventListener('click', exportChat);

// ============================================================
// Init
// ============================================================
if (!activeId && chats.length > 0) activeId = chats[0].id;
renderSidebar();
renderMessages();
updateSendState();
els.input.focus();
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
        {"role": "system", "content": "You are DamperBot, a friendly, sharp, and helpful AI assistant. Use markdown formatting (headings, lists, code blocks, bold) when it improves clarity. Keep answers focused and useful."},
    ] + [m.model_dump() for m in req.messages]

    def stream():
        try:
            completion = client.chat.completions.create(
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