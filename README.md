# DOT — Fabrica

A self-evolving NEAT neural network and autonomous AI agent, running entirely on free-tier
serverless infrastructure. (Vercel + Cloudflare Workers + Upstash Redis)

```
┌─────────────────────────────────────────────────────────────────┐
│  Browser (single terminal UI)                                   │
│  /agent /chat /evolve /status /improve /memory /clear /help     │
└───────────────┬───────────────────────────────┬─────────────────┘
                │                                │
                ▼                                ▼
   ┌─────────────────────────┐       ┌───────────────────────────────┐
   │  Vercel (Python/FastAPI)│       │ Cloudflare Worker (ES Modules)│
   │  api/index.py           │       │  worker.js                    │
   │                         │       │                               │
   │  • NEAT evolution engine│◄──────┤  • Chat                       │
   │  • Terminal UI (HTML/JS)│ self- │  • Autonomous agent (ReAct)   │
   │  • State storage        │improve│  • Self-improve analysis      │
   │  • GitHub commit API    │──────►│  • Multi-provider LLM router  │
   └───────────┬─────────────┘       └──────────────┬────────────────┘
               │                                    │
               ▼                                    ▼
      ┌──────────────────┐                ┌────────────────────────┐
      │ Upstash Redis    │                │ Groq (primary)         │
      │ (NEAT state,     │                │ Cloudflare Workers AI  │
      │  chat history)   │                │ (automatic fallback)   │
      └──────────────────┘                └────────────────────────┘
```

## What this actually is

Two systems sharing one terminal interface:

1. **NEAT neuroevolution** — a population of small neural networks evolves over generations
   to approximate math functions (`sin`, `cos`, etc.), with a complexity penalty that
   rewards compact solutions over large ones. State persists in Redis across calls, since
   Vercel functions have no memory between invocations.
2. **DOT, an autonomous agent** — a ReAct-style loop (THOUGHT → ACTION → OBSERVATION) with
   tool use, sub-agent spawning, persistent memory, and the ability to trigger NEAT
   evolution or propose changes to its own evolution tasks via GitHub commits.

Both are exposed through one command-driven terminal — there's no separate "NEAT tab";
evolution is just another thing the agent (or you, via slash commands) can do.

## Stack

| Layer | Tech |
|---|---|
| Frontend | Single-page terminal, vanilla JS, served as a string from FastAPI |
| Backend | Vercel serverless (Python/FastAPI) |
| Agent + chat runtime | Cloudflare Worker (ES Modules) |
| State | Upstash Redis (NEAT population state, chat history) |
| Primary LLM | Groq (`llama-3.3-70b-versatile`) |
| Fallback LLM | Cloudflare Workers AI (`@cf/meta/llama-3.3-70b-instruct-fp8-fast`), native binding, no external fetch |
| Web search | Tavily API |
| Self-improvement | GitHub REST API (commits directly to this repo) |

Everything is built and deployed from a phone — GitHub's web editor, the Vercel dashboard,
and the Cloudflare dashboard. No local dev environment, no CLI tools.

## Why two LLM providers

Groq's free tier has a per-minute token cap that gets hit fast under real usage (chat,
agent steps, and self-improve all share the same quota). Cloudflare Workers AI runs
natively inside the same Worker via an `AI` binding — not an external HTTP call — so it
has no exposure to the Cloudflare-WAF-blocks-AWS-Lambda problem that affects calls from
Vercel's Python runtime to most external APIs. `callLLM()` tries Groq first (faster),
falls back to Workers AI on failure, and every response reports which provider actually
handled it.

**Known limitation, not hidden:** the Workers AI fallback model is noticeably weaker than
Groq at following the strict multi-step THOUGHT/ACTION/INPUT format the agent loop
requires. Simple chat degrades fine on fallback; complex multi-tool agent tasks sometimes
don't. The agent detects when it's running on fallback and gives an honest "can't reliably
do this right now" answer instead of producing a confused half-result.

## Autonomous agent

Tools available to the agent:

| Tool | Does |
|---|---|
| `web_search` | Tavily search |
| `fetch_url` | Fetch and strip a webpage to plain text |
| `spawn_agent` | Run a specialized sub-agent with its own role/task |
| `neat_status` | Read current NEAT fitness/neuron stats + recent history, no side effects |
| `run_neat` | Run N *additional* evolution generations |
| `write_code` | Propose a change to the NEAT `TASKS` dict and commit it (narrow scope — not a general code generator) |
| `remember` | Save a fact to persistent memory, round-tripped across turns |

Loop has a hard cap of 6 steps, a loop-breaker for repeated non-progress, a code-level
guard against calling the same read-only tool twice with identical input, and requires
analytical answers to cite the actual numbers observed rather than generic conclusions.

Conversation history (last 8 turns) and persistent memory are both injected into every
agent call, so follow-ups like "tell me more" resolve correctly instead of starting from
a blank slate.

## Self-improvement loop

`/improve` → Vercel reads current NEAT state from Redis → fetches `api/index.py` from
GitHub → extracts the `TASKS` code block → sends it to the Worker's `/meta` endpoint →
the model proposes one JSON-formatted, `ast`-validated code change → committed back to
this repo via the GitHub API → Vercel auto-redeploys. History of every proposal (applied
or not) is kept in Redis, readable via `GET /api/improve-history`.

## Terminal commands

```
/agent          switch to autonomous agent mode
/chat           switch to conversational mode
/evolve [N]     run N NEAT generations (default 100)
/status         show current NEAT fitness/neuron stats + history
/improve        trigger the self-improvement loop
/memory         open the agent memory panel
/clear          clear the terminal
/help           list commands
```

`tab` also toggles agent/chat mode. Prefixing a message with `$` forces agent mode for
that one message. Multiple `$`-separated goals in one message are automatically split
into a sequential queue (with a delay between each) instead of being merged into one
garbled goal.

## Environment variables

**Vercel:**

| Var | Purpose |
|---|---|
| `UPSTASH_REDIS_REST_URL` / `KV_REST_API_URL` | Redis connection (either naming works) |
| `UPSTASH_REDIS_REST_TOKEN` / `KV_REST_API_TOKEN` | Redis auth |
| `CHAT_WORKER_URL` | URL of the Cloudflare Worker, used by the frontend |
| `GITHUB_TOKEN` | Classic PAT, `repo` scope only, for the self-improve commit loop |
| `GITHUB_REPO` | `username/repo` |
| `IMPROVE_SECRET` | Optional, guards the self-improve endpoint |
| `CRON_SECRET` | Optional, guards the scheduled evolve cron |

**Cloudflare Worker:**

| Var | Purpose |
|---|---|
| `GROQ_API_KEY` | Primary LLM provider |
| `GROQ_MODEL_NAME` | Chat model (default `llama-3.3-70b-versatile`) |
| `AGENT_MODEL_NAME` | Agent reasoning model (default `llama-3.3-70b-versatile`) |
| `WORKERS_AI_MODEL` | Fallback model (default `@cf/meta/llama-3.3-70b-instruct-fp8-fast`) |
| `TAVILY_API_KEY` | Web search |
| `DOT_VERCEL_URL` | URL of the Vercel deployment, so the Worker can call back into it |
| `AI` binding | Cloudflare Workers AI — must be added via Settings → Bindings, named exactly `AI` |

## Known cruft / not yet cleaned up

- `vercel.json` still routes `/api/chat` to `api/chat.js`, a file that no longer exists
  in the repo (an earlier Node.js chat endpoint, abandoned after repeated 404s — chat now
  goes browser → Worker directly). Dead route, harmless but should be removed.
- `api/index.py` retains an old direct Groq/HuggingFace call path (`GROQ_KEY`, `HF_TOKEN`,
  `HF_MODEL_ID`) from before chat moved to the Worker. Likely unreachable in the current
  flow; not yet removed.
- `train.py` (Colab LoRA fine-tuning on Qwen2.5-0.5B, pushed to HF Hub) was built early
  on for a "train a small model on our own conversation data" direction. Untouched since,
  unused while Groq/Workers AI remain primary.

## Honest current limitations

- NEAT evolves on four hand-picked `sin`/`cos`/`abs` tasks. They're simple enough that the
  population plateaus quickly (4-neuron solutions reaching ~0.998 fitness and staying
  there for hundreds of generations) — not because the search is broken, but because the
  task doesn't demand more structure. A genuine time-series prediction task (vector window
  → next value) would force real topology growth; not yet implemented.
- Free-tier rate limits are real and shared across chat, agent, and self-improve calls.
  Heavy testing sessions will hit them; the multi-provider fallback softens this but
  doesn't eliminate it.
- `write_code` only ever proposes changes to the NEAT `TASKS` dictionary. It is not, and
  is not intended to be, a general-purpose coding tool.

## Local setup

There isn't one — this project has never been run outside Vercel + Cloudflare's
dashboards. To fork it: deploy `api/index.py` to Vercel, deploy `worker.js` to a
Cloudflare Worker, connect an Upstash Redis database via the Vercel Storage marketplace,
add the environment variables above, and add the Workers AI binding from the Cloudflare
dashboard.

## License

MIT.
