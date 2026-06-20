// DOT Cloudflare Worker — chat + autonomous agent + self-improvement
// ES Modules 格式(Workers AI binding 要求,不能用舊版 Service Worker 格式)
// ================================================================

var CORS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, x-improve-secret',
};

function json(data, status) {
  return new Response(JSON.stringify(data), {
    status: status || 200,
    headers: Object.assign({ 'Content-Type': 'application/json' }, CORS),
  });
}

function getModel(env) {
  return env.GROQ_MODEL_NAME || 'llama-3.3-70b-versatile';
}

function getAgentModel(env) {
  // Agent 需要更強的推理能力,預設用 70B,即使 chat 用小模型也不受影響
  return env.AGENT_MODEL_NAME || 'llama-3.3-70b-versatile';
}

function getKey(env) {
  return env.GROQ_API_KEY || '';
}

function getWorkersAIModel(env) {
  return env.WORKERS_AI_MODEL || '@cf/meta/llama-3.3-70b-instruct-fp8-fast';
}

function hasWorkersAI(env) {
  // AI 是 Cloudflare Workers AI 的 binding 名稱(Worker Settings → Bindings → Workers AI,變數名稱要叫 AI)
  return !!(env && env.AI);
}

export default {
  async fetch(request, env, ctx) {
    return handleRequest(request, env);
  }
};

async function handleRequest(request, env) {
  if (request.method === 'OPTIONS') return new Response(null, { headers: CORS });
  var url = new URL(request.url);

  if (url.pathname === '/meta'  && request.method === 'POST') return handleMeta(request, env);
  if (url.pathname === '/agent' && request.method === 'POST') return handleAgent(request, env);
  if (request.method === 'GET')  return json({
    ok: true,
    groq_key: !!getKey(env),
    workers_ai: hasWorkersAI(env),
    chat_model: getModel(env),
    agent_model: getAgentModel(env),
    workers_ai_model: getWorkersAIModel(env),
    version: '3.1-es-modules',
  });
  if (request.method === 'POST') return handleChat(request, env);
  return new Response('not found', { status: 404, headers: CORS });
}

// ── Groq ─────────────────────────────────────────────────────
async function callGroq(messages, maxTokens, temp, modelOverride, env) {
  var r = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Authorization': 'Bearer ' + getKey(env), 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: modelOverride || getModel(env),
      messages: messages,
      max_tokens: maxTokens || 600,
      temperature: temp !== undefined ? temp : 0.7,
    }),
  });
  var d = await r.json();
  if (!d.choices) throw new Error('Groq: ' + JSON.stringify(d).slice(0, 200));
  return d.choices[0].message.content;
}

// ── Cloudflare Workers AI ────────────────────────────────────
// 跑在同一個 Cloudflare 網路裡,不是外部 fetch,沒有 WAF / 跨網路問題
// 免費額度:10,000 neurons/day,每天 00:00 UTC 重置
async function callWorkersAI(messages, maxTokens, env) {
  if (!hasWorkersAI(env)) throw new Error('Workers AI binding not configured (Settings → Bindings → Workers AI → name it "AI")');
  var resp = await env.AI.run(getWorkersAIModel(env), {
    messages: messages,
    max_tokens: maxTokens || 600,
  });
  var text = resp && (resp.response || (resp.result && resp.result.response));
  if (!text) throw new Error('Workers AI: empty response ' + JSON.stringify(resp).slice(0, 150));
  return text;
}

// ── 統一入口:Groq 優先(快),失敗自動 fallback 到 Workers AI ──
async function callLLM(messages, maxTokens, temp, modelOverride, env) {
  var groqErr = null;
  if (getKey(env)) {
    try {
      var text = await callGroq(messages, maxTokens, temp, modelOverride, env);
      return { text: text, provider: 'groq' };
    } catch (e) {
      groqErr = e;
    }
  }
  if (hasWorkersAI(env)) {
    try {
      var text2 = await callWorkersAI(messages, maxTokens, env);
      return { text: text2, provider: 'workers-ai' + (groqErr ? ' (groq fallback)' : '') };
    } catch (e2) {
      throw new Error('Groq: ' + (groqErr ? groqErr.message : 'no key') + ' | Workers AI: ' + e2.message);
    }
  }
  throw groqErr || new Error('No LLM provider configured (set GROQ_API_KEY or bind Workers AI)');
}

function friendlyError(e) {
  var msg = String(e && e.message || e);
  if (msg.includes('tokens per minute') || msg.includes('rate_limit')) {
    return '⏳ Groq 免費版額度用完，Workers AI 備援也沒設定或失敗了，請等 60 秒再試。';
  }
  return '⚠️ 所有 provider 都失敗了：' + msg.slice(0, 250);
}

// ── Chat ─────────────────────────────────────────────────────
async function handleChat(request, env) {
  if (!getKey(env) && !hasWorkersAI(env)) return json({ reply: '⚠️ 沒有設定任何 LLM provider。請設 GROQ_API_KEY 或綁定 Workers AI。' });
  var body;
  try { body = await request.json(); } catch(e) { return json({ error: 'invalid JSON' }, 400); }
  var messages = (body.messages || []).slice(-6);
  var system   = body.system || CHAT_SYSTEM;
  try {
    var result = await callLLM([{ role: 'system', content: system }].concat(messages), 300, 0.7, null, env);
    return json({ reply: result.text, provider: result.provider });
  } catch(e) {
    return json({ reply: friendlyError(e) });
  }
}

var CHAT_SYSTEM = `You are DOT, built by Fabrica. You are not a generic assistant: you are the
conversational front-end of a real running system with two parts:

1. A NEAT neuroevolution engine, currently evolving small neural networks against math
   approximation tasks. It has live, current state (generation count, fitness, neuron
   counts) that changes constantly. You do NOT have this live state in plain chat mode —
   it is only available through /status or by switching to agent mode (/agent or
   prefixing a message with $). NEVER guess or make up current generation numbers,
   fitness values, or neuron counts. If asked about current NEAT status, say plainly
   that you need agent mode or /status to check live data — do not fabricate a plausible-
   sounding number.
2. An autonomous agent mode (separate from this chat) with tools: web search, fetching
   URLs, spawning sub-agents, checking/running NEAT evolution, proposing code changes to
   evolution tasks, and persistent memory. You yourself, in this current chat exchange,
   cannot call any of these tools — only agent mode can. If the user asks you to do
   something requiring a tool (search, check status, run evolution, etc.), tell them to
   switch to agent mode or prefix their message with $, rather than pretending to have
   done it.

On your own knowledge: you are running on a Llama 3.3 based model (via Groq or
Cloudflare Workers AI), whose training data has a cutoff in the past — be honest about
that being a real limitation for general world knowledge. But this is separate from your
identity as DOT: you are not "just a base Llama model," you are that model put to work as
the conversational layer of a specific, real, currently-running system, and you should
talk about yourself that way rather than reverting to a generic assistant persona.

Be concise and direct.`;

// ── Autonomous Agent (ReAct loop with persistent memory + conversation context) ─
var AGENT_SYSTEM = `You are DOT, an autonomous AI agent by Fabrica with persistent memory across conversations.
Complete goals step-by-step using tools. Format each step EXACTLY like this:

THOUGHT: your reasoning
ACTION: tool_name
INPUT: tool input (plain text or JSON)

EXAMPLE of a correct response (copy this exact structure, just change the content):
THOUGHT: I need to check the current NEAT status before I can compare it to history.
ACTION: neat_status
INPUT: status

Available tools:
- web_search   → search the web (INPUT: query string)
- fetch_url    → fetch webpage content (INPUT: URL)
- spawn_agent  → run a specialized sub-agent (INPUT: {"role":"...","task":"..."})
- neat_status  → check current NEAT fitness/neuron stats AND recent historical trend, WITHOUT running new evolution (INPUT: leave empty or "status")
- run_neat     → run N *additional* NEAT generations on top of whatever has already run (INPUT: a number, e.g. 150 means run 150 new generations, NOT reach generation 150)
- write_code   → propose ONE specific change to the NEAT evolution TASKS (the math functions being evolved) and commit it. This does NOT write general-purpose code, algorithms, scripts, or programs — it can ONLY modify the NEAT TASKS dictionary. Never use this tool for "write me an algorithm / sorting function / program" type requests; it will fail because that's not what it does. (INPUT: description of what TASKS change to make)
- remember     → save an important fact for future conversations (INPUT: the fact, short and specific)

CRITICAL RULES:
- Every single response you give MUST contain either (a) a THOUGHT + ACTION + INPUT block, or (b) a FINAL_ANSWER. There is no third option — never respond with only a THOUGHT, only a sentence, or a statement like "I need to wait for tool results." You cannot "wait" — calling ACTION immediately gives you the result in the same turn. If you need a tool, call it NOW in this response.
- NEVER call the same read-only tool (neat_status, web_search with the same query, fetch_url on the same URL) twice in a row without a different ACTION (like run_neat) happening in between. If you already received an OBSERVATION with the data you need earlier in this conversation, reuse it — do not re-query for identical data, it will not change.
- INPUT must contain ONLY the raw tool input — no parenthetical notes, no explanations, no trailing comments after it. For spawn_agent, INPUT must be valid JSON and nothing else.
- For "compare to before / has it improved / plateau" type questions about NEAT: call neat_status FIRST to see the historical trend, optionally run_neat if the goal asks you to evolve further, then call neat_status AGAIN afterward (only once, only if run_neat happened in between) and compare the fitness numbers yourself in your FINAL_ANSWER. Do not ask the user to do this comparison for you — you have the data, use it.
- For analytical questions (e.g. "what does this trend mean", "why is X happening"), your FINAL_ANSWER must cite the SPECIFIC numbers from your observations (exact fitness values, neuron counts, avg/max stats) as evidence — never give a generic explanation like "may have reached a local optimum" without grounding it in the actual numbers you observed. If the numbers don't clearly support a conclusion, say what's ambiguous about them instead of guessing.
- Use [Recent conversation] below to resolve vague follow-ups like "tell me more" or "go on" — figure out what the previous turn was about and continue from there. Never ask the user to repeat something they already said.
- If the [Memory] section contains relevant facts, use them directly — don't re-ask or re-search what you already know.
- NEVER use a "think" action to stall or loop. If you are unsure what the user wants, immediately write FINAL_ANSWER asking ONE specific clarifying question — do not call any tool first.
- If the goal is abstract or about proving a quality about yourself (e.g. "prove you're smart", "show me what you can do", "impress me") rather than a concrete task with a clear deliverable, do NOT guess which tool to misuse. Ask in your FINAL_ANSWER what specific outcome they want (e.g. "want me to research something, run an evolution cycle, or analyze existing data?") — without calling any tool first.
- If you already have enough information from your tool results to answer the goal, finish with FINAL_ANSWER immediately. Do NOT ask the user a clarifying question when the data needed to answer is already in your observations — that wastes their time. Only ask a clarifying question when the goal itself is genuinely ambiguous before you've used any tools.
- Every ACTION must make concrete progress (search, fetch, spawn, evolve, write, check status, or remember). If no tool can help, skip straight to FINAL_ANSWER.
- When done, write:
FINAL_ANSWER: [complete answer]

Rules: max 6 steps. Be concise in inputs. Always ground your answer in real information — don't guess.`;

async function runTool(name, input, memoryRef, env) {
  var TAVILY_KEY = env.TAVILY_API_KEY || '';
  var VERCEL_URL = env.DOT_VERCEL_URL || '';

  if (name === 'remember') {
    var fact = String(input || '').trim();
    if (fact) {
      memoryRef.push(fact);
      if (memoryRef.length > 50) memoryRef.shift(); // cap memory size
    }
    return 'Saved to memory: ' + fact;
  }

  if (name === 'web_search') {
    if (!TAVILY_KEY) return 'web_search unavailable (set TAVILY_API_KEY in Worker)';
    try {
      var r = await fetch('https://api.tavily.com/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: TAVILY_KEY, query: input, max_results: 4, search_depth: 'basic' }),
      });
      var d = await r.json();
      var results = (d.results || []).map(function(x) {
        return x.title + '\n' + x.url + '\n' + (x.content || '').slice(0, 300);
      }).join('\n---\n');
      return results || 'No results found.';
    } catch(e) { return 'web_search error: ' + e; }
  }

  if (name === 'fetch_url') {
    try {
      var r = await fetch(input, { headers: { 'User-Agent': 'DOT-Agent/2.0' } });
      var text = await r.text();
      // Strip HTML tags, keep text
      text = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
      return text.slice(0, 2500);
    } catch(e) { return 'fetch_url error: ' + e; }
  }

  if (name === 'spawn_agent') {
    try {
      var role, task;
      if (typeof input === 'string') {
        var jsonMatch = input.match(/\{[\s\S]*\}/);  // 只抓 {...} 區塊,忽略前後雜訊文字
        if (jsonMatch) {
          var parsed = JSON.parse(jsonMatch[0]);
          role = parsed.role || 'assistant';
          task = parsed.task || jsonMatch[0];
        } else {
          // 不是 JSON 格式,直接把整段當 task,role 用通用值
          role = 'assistant';
          task = input;
        }
      } else {
        role = input.role || 'assistant';
        task = input.task || JSON.stringify(input);
      }
      var result = await callLLM([
        { role: 'system', content: 'You are a specialist sub-agent: ' + role + '. Be thorough and precise.' },
        { role: 'user',   content: task }
      ], 500, 0.5, null, env);
      return '[Sub-agent ' + role + ']: ' + result.text;
    } catch(e) { return 'spawn_agent error: ' + e; }
  }

  if (name === 'run_neat') {
    if (!VERCEL_URL) return 'run_neat unavailable (set DOT_VERCEL_URL in Worker)';
    try {
      var gens = parseInt(input) || 100;
      var r = await fetch(VERCEL_URL + '/api/evolve?generations=' + gens);
      var d = await r.json();
      return 'NEAT result after running ' + gens + ' NEW generations: total_gen=' + d.total_generations +
             ' fitness=' + d.best_fitness + ' neurons=' + d.n_neurons + ' mse=' + d.mse_current_task +
             '. Use neat_status to compare against historical trend.';
    } catch(e) { return 'run_neat error: ' + e; }
  }

  if (name === 'neat_status') {
    if (!VERCEL_URL) return 'neat_status unavailable (set DOT_VERCEL_URL in Worker)';
    try {
      var r = await fetch(VERCEL_URL + '/api/status');
      var d = await r.json();
      if (d.status === 'no_data') return 'No NEAT data yet — nothing has evolved.';
      var hist = (d.history || []).slice(-10).map(function(h) {
        return 'gen' + h.gen + '/' + h.task + ': fit=' + h.fitness + ' n=' + h.n_neurons;
      }).join(', ');
      return 'Current snapshot: total_gen=' + d.total_generations + ' task=' + d.current_task +
             ' fitness=' + d.best_fitness + ' neurons=' + d.n_neurons + '.' +
             ' Population structural spread RIGHT NOW: avg_neurons=' + d.avg_neurons + ', max_neurons=' + d.max_neurons +
             ' (this tells you how much the current population is exploring larger structures vs converging to a compact one).' +
             ' Last 10 generation checkpoints (gen/task: fitness=n_neurons): ' + hist;
    } catch(e) { return 'neat_status error: ' + e; }
  }

  if (name === 'write_code') {
    if (!VERCEL_URL) return 'write_code unavailable (set DOT_VERCEL_URL in Worker)';
    try {
      var r = await fetch(VERCEL_URL + '/api/self-improve', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      var d = await r.json();
      return d.ok ? ('Code updated: ' + d.expected_effect + ' | commit: ' + d.commit_url) : ('self-improve: ' + (d.error || JSON.stringify(d)));
    } catch(e) { return 'write_code error: ' + e; }
  }

  if (name === 'think') {
    return 'Thought noted: ' + input;
  }

  return 'Unknown tool: ' + name;
}

async function handleAgent(request, env) {
  if (!getKey(env) && !hasWorkersAI(env)) return json({ error: '沒有設定任何 LLM provider。請設 GROQ_API_KEY 或綁定 Workers AI。' }, 400);
  var body;
  try { body = await request.json(); } catch(e) { return json({ error: 'invalid JSON' }, 400); }
  var goal    = body.goal || '';
  var context = body.context || '';
  var memory  = Array.isArray(body.memory)  ? body.memory.slice()  : [];  // mutable copy
  var history = Array.isArray(body.history) ? body.history.slice(-8) : []; // recent turns
  if (!goal) return json({ error: 'goal required' }, 400);

  var memBlock = memory.length
    ? '\n\n[Memory]\n' + memory.map(function(m, i) { return (i + 1) + '. ' + m; }).join('\n')
    : '';
  var histBlock = history.length
    ? '\n\n[Recent conversation]\n' + history.map(function(h) { return (h.role === 'user' ? 'User: ' : 'DOT: ') + h.content; }).join('\n')
    : '';

  var messages = [
    { role: 'system', content: AGENT_SYSTEM },
    { role: 'user', content: (context ? 'Context:\n' + context + '\n\n' : '') + memBlock + histBlock + '\n\nGoal: ' + goal }
  ];

  var steps = [];
  var finalAnswer = null;
  var MAX_STEPS = 6;
  var thinkStreak = 0;
  var lastAction = null;
  var lastInput = null;
  var formatRetry = 0;
  var AGENT_MODEL = getAgentModel(env);
  var usedProvider = null;

  for (var i = 0; i < MAX_STEPS; i++) {
    var response;
    try {
      var llmRes = await callLLM(messages, 900, 0.15, AGENT_MODEL, env);
      response = llmRes.text;
      if (typeof response !== 'string') {
        response = (response === null || response === undefined) ? '' :
                   (typeof response === 'object' ? JSON.stringify(response) : String(response));
      }
      usedProvider = llmRes.provider;
    } catch(e) {
      steps.push({ type: 'error', content: String(e) });
      finalAnswer = friendlyError(e);  // 立刻把真實錯誤傳給使用者,不要靜默吞掉
      break;
    }

    if (response.includes('FINAL_ANSWER:')) {
      finalAnswer = response.split('FINAL_ANSWER:').pop().trim();
      steps.push({ type: 'final', content: response });
      break;
    }

    var thoughtMatch = response.match(/THOUGHT:\s*([\s\S]*?)(?=ACTION:|FINAL_ANSWER:|$)/i);
    var actionMatch  = response.match(/ACTION:\s*(\w+)/i);
    var inputMatch   = response.match(/INPUT:\s*([\s\S]*?)(?=THOUGHT:|ACTION:|FINAL_ANSWER:|$)/i);

    var thought = thoughtMatch ? thoughtMatch[1].trim() : '';
    var action  = actionMatch  ? actionMatch[1].trim().toLowerCase() : '';
    var input   = inputMatch   ? inputMatch[1].trim() : '';
    // 防禦性清理:砍掉模型偶爾加在 INPUT 後面的註解尾巴,例如 "(Note: ...)"
    // 但不能砍掉合法 JSON 內容(JSON 開頭是 { 或 [ 就跳過這個清理)
    if (input && input[0] !== '{' && input[0] !== '[') {
      input = input.replace(/\s*\([Nn]ote:[\s\S]*$/, '').trim();
    } else if (input && (input[0] === '{' || input[0] === '[')) {
      // JSON 輸入:只保留到對應的結尾括號,砍掉後面任何雜訊文字
      var closeChar = input[0] === '{' ? '}' : ']';
      var openChar  = input[0];
      var depth = 0, endIdx = -1;
      for (var ci = 0; ci < input.length; ci++) {
        if (input[ci] === openChar) depth++;
        else if (input[ci] === closeChar) { depth--; if (depth === 0) { endIdx = ci; break; } }
      }
      if (endIdx !== -1) input = input.slice(0, endIdx + 1);
    }

    steps.push({ type: 'step', thought: thought, action: action, input: input });

    if (!action) {
      var onFallback = !usedProvider || usedProvider.indexOf('groq') !== 0; // 不是純 groq,代表正在用備援模型
      formatRetry++;
      steps.push({ type: 'format_retry', content: response });

      if (onFallback) {
        // Workers AI 備援模型對嚴格的多步驟工具格式遵循能力較弱,重試大概率還是失敗
        // 與其浪費一輪 API 呼叫,不如誠實告知降級狀態,而不是把半成品答案包裝成正式結論
        finalAnswer = '⚠️ Groq 額度目前還沒恢復，由 Workers AI 備援處理，但這個模型在多步驟工具呼叫的格式遵循上不夠穩定，無法可靠完成這個任務。' +
          (response.trim() ? '（它目前的想法：「' + response.trim().slice(0, 150) + '」）' : '') +
          ' 建議稍後 Groq 額度恢復後再試一次，或把任務拆成更簡單的單一步驟。';
        break;
      }

      if (formatRetry >= 2) {
        // 連續兩次格式不對才真的放棄,避免無限重試,但也不會只憑一次失敗就把半成品答案當正式結果
        finalAnswer = response.trim() ||
          '我需要使用工具才能回答這個問題，但格式判讀失敗了，可以換個方式描述目標嗎？';
        break;
      }
      // 不要把這句「半成品」當最終答案,而是糾正格式並讓它重試,並給一個具體範例
      messages.push({ role: 'assistant', content: response });
      messages.push({
        role: 'user',
        content: 'INVALID — your response had no ACTION line. There is no "waiting" step; calling ACTION gives you the result immediately in this same exchange. Respond with EXACTLY this structure now, using a real tool:\n' +
                 'THOUGHT: <short reasoning>\n' +
                 'ACTION: <one of: web_search, fetch_url, spawn_agent, neat_status, run_neat, write_code, remember>\n' +
                 'INPUT: <the input>\n' +
                 'Do not write anything else. Try again now.'
      });
      continue;
    }

    // Hard loop-breaker: model keeps calling "think" instead of making progress
    if (action === 'think') {
      thinkStreak++;
      if (thinkStreak >= 2) {
        steps.push({ type: 'loop_break', content: 'Stopped repeated think loop' });
        finalAnswer = thought || '我需要更明確的問題才能繼續，可以再具體說明一下嗎？';
        break;
      }
    } else {
      thinkStreak = 0;
    }

    var observation;
    var READ_ONLY = { neat_status: 1, web_search: 1, fetch_url: 1 };
    if (READ_ONLY[action] && action === lastAction && input === lastInput) {
      // 程式碼層級防呆:同一個唯讀工具、同樣的 input,連續呼叫第二次直接攔下來
      // 不真的再打一次 API,因為結果保證一樣,純粹浪費一輪
      observation = '(Skipped — identical to your previous ' + action + ' call. Nothing has changed since then; reuse that observation instead of repeating it.)';
    } else {
      try {
        observation = await runTool(action, input, memory, env);
      } catch(e) {
        observation = 'Tool error: ' + String(e);
      }
    }
    lastAction = action;
    lastInput = input;

    steps.push({ type: 'observation', tool: action, result: observation });
    messages.push({ role: 'assistant', content: response });
    messages.push({ role: 'user', content: 'OBSERVATION: ' + observation.slice(0, 1500) });
  }

  if (!finalAnswer) {
    try {
      messages.push({ role: 'user', content: 'Summarize your findings and give FINAL_ANSWER:' });
      var synthRes = await callLLM(messages, 500, 0.4, AGENT_MODEL, env);
      var synthesis = synthRes.text;
      usedProvider = synthRes.provider;
      finalAnswer = synthesis.includes('FINAL_ANSWER:') ? synthesis.split('FINAL_ANSWER:').pop().trim() : synthesis;
    } catch(e) {
      finalAnswer = friendlyError(e) + (steps.length ? '（已完成 ' + steps.length + ' 個步驟，見上方紀錄）' : '');
    }
  }

  return json({ ok: true, goal: goal, steps: steps, final_answer: finalAnswer, memory: memory, provider: usedProvider });
}

// ── Meta (self-improve analysis) ─────────────────────────────
var META_SYSTEM = 'You are DOT\'s autonomous self-improvement engine for a NEAT neuroevolution system.\n\nThe system evolves neural networks on mathematical function approximation tasks.\nTasks use `_XS` (40 floats from -π to π). Each task is a list of (x, y) pairs.\n\nYour job: propose ONE change to improve or diversify the TASKS dictionary.\nOptions: add a new harder task, or replace the easiest task with a more complex one.\n\nSTRICT RULES:\n- Return ONLY a JSON object. No markdown. No text outside the JSON.\n- `old_code` must be an EXACT substring of the TASKS code provided.\n- `new_code` must be valid Python using only: math.sin, math.cos, math.tan, abs, x, and arithmetic operators.\n- Task format: "key": [(x, expression) for x in _XS]\n- Expressions must not raise exceptions for x in [-π, π] (avoid log, tan near π/2, division by zero).\n- Maximum 5 tasks total.\n\nReturn this JSON:\n{"analysis":"observation in 1-2 sentences","change_type":"tasks","old_code":"exact Python substring to replace","new_code":"replacement Python","expected_effect":"what this will cause the network to do"}';

async function handleMeta(request, env) {
  if (!getKey(env) && !hasWorkersAI(env)) return json({ error: '沒有設定任何 LLM provider' }, 400);
  var body;
  try { body = await request.json(); } catch(e) { return json({ error: 'invalid JSON' }, 400); }
  var userMsg = 'Evolution state:\n' + JSON.stringify(body.state || {}, null, 2) +
                '\n\nCurrent TASKS code:\n' + (body.tasks_code || '') +
                '\n\nRecent history:\n' + (body.history || '');
  try {
    var result = await callLLM([
      { role: 'system', content: META_SYSTEM },
      { role: 'user',   content: userMsg }
    ], 600, 0.2, null, env);
    var text = result.text;
    // 防呆:某些 provider/模型組合可能回傳非字串(物件、陣列、undefined),
    // 不要讓 .match() 直接炸成 500,先正規化成字串再處理
    if (typeof text !== 'string') {
      text = (text === null || text === undefined) ? '' :
             (typeof text === 'object' ? JSON.stringify(text) : String(text));
    }
    if (!text) return json({ error: 'empty response from ' + result.provider, provider: result.provider }, 422);
    var match = text.match(/\{[\s\S]*\}/);
    if (!match) return json({ error: 'no JSON in response', raw: text.slice(0, 400), provider: result.provider }, 422);
    var change;
    try { change = JSON.parse(match[0]); }
    catch(e) { return json({ error: 'JSON parse failed', raw: match[0].slice(0, 400) }, 422); }
    return json({ change: change, provider: result.provider });
  } catch(e) {
    return json({ error: String(e) }, 500);
  }
}
