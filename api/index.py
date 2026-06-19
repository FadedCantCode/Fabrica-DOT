"""
/api/index.py —— DOT: 會長出神經元的可演化網路(NEAT 式拓樸演化)

跟之前最大的差別:神經元數量不再固定。genome 是一組「隱藏單元」的集合,
演化過程可以:
  - add_neuron:長出一個新的隱藏單元(隨機初始化權重)
  - prune_neuron:刪掉貢獻最低的隱藏單元
  - perturb:擾動現有權重(self-adaptive sigma)
所以網路結構會從最小(0~1 個隱藏單元)隨演化長大或精簡,這是真正的拓樸演化,
不是固定結構調權重。

終點誠實聲明:它長成的是一個小型函數逼近器,不是語言模型。
"""

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
import json
import math
import os
import random


# ============================================================
# 輸入編碼:把純量 x 編成固定輸入向量(這層不演化,演化的是隱藏層)
# ============================================================

def encode(x: float) -> list:
    return [x, math.sin(x), math.cos(x), 1.0]  # 最後一項是 bias 輸入

N_INPUTS = len(encode(0.0))


# ============================================================
# genome:一個可變大小的隱藏單元集合 + 一組輸出權重
# 每個 hidden unit = {"w": [N_INPUTS 個輸入權重], "out": 對輸出的權重}
# ============================================================

def new_hidden_unit(rng):
    return {"w": [rng.uniform(-1, 1) for _ in range(N_INPUTS)], "out": rng.uniform(-1, 1)}


def new_genome(rng, n_hidden=1):
    return {
        "hidden": [new_hidden_unit(rng) for _ in range(n_hidden)],
        "out_bias": rng.uniform(-0.5, 0.5),
    }


def genome_forward(genome, x):
    feats = encode(x)
    total = genome["out_bias"]
    for unit in genome["hidden"]:
        z = sum(w * f for w, f in zip(unit["w"], feats))
        total += math.tanh(z) * unit["out"]
    return total


def genome_size(genome):
    return len(genome["hidden"])


def clone_genome(genome):
    return {
        "hidden": [{"w": list(u["w"]), "out": u["out"]} for u in genome["hidden"]],
        "out_bias": genome["out_bias"],
    }


def mse_loss(genome, data):
    n = len(data)
    return sum((genome_forward(genome, x) - y) ** 2 for x, y in data) / n


# ============================================================
# 結構距離:兩個 genome 差多少(單元數差 + 最近單元權重差)
# 給 version history / confidence 用,也是 speciation 的基礎
# ============================================================

def genome_distance(g1, g2):
    size_diff = abs(genome_size(g1) - genome_size(g2))
    h1, h2 = g1["hidden"], g2["hidden"]
    if not h1 or not h2:
        return size_diff + 1.0
    wdiff = 0.0
    for u in h1:
        best = min(
            sum((a - b) ** 2 for a, b in zip(u["w"], v["w"])) + (u["out"] - v["out"]) ** 2
            for v in h2
        )
        wdiff += math.sqrt(best)
    return size_diff + wdiff / len(h1)


# ============================================================
# 突變:結構(長/刪神經元)+ 權重擾動
# ============================================================

def mutate(genome, sigma, rng, add_prob=0.18, prune_prob=0.04, max_hidden=20):
    g = clone_genome(genome)

    # 長出新神經元
    if rng.random() < add_prob and len(g["hidden"]) < max_hidden:
        g["hidden"].append(new_hidden_unit(rng))

    # 刪掉貢獻最低的神經元(只在單元偏多時才考慮,避免結構塌回 1)
    if rng.random() < prune_prob and len(g["hidden"]) > 3:
        idx = min(range(len(g["hidden"])), key=lambda i: abs(g["hidden"][i]["out"]))
        g["hidden"].pop(idx)

    # 權重擾動
    for unit in g["hidden"]:
        for i in range(len(unit["w"])):
            if rng.random() < 0.7:
                unit["w"][i] += rng.gauss(0, sigma)
        if rng.random() < 0.7:
            unit["out"] += rng.gauss(0, sigma)
    if rng.random() < 0.7:
        g["out_bias"] += rng.gauss(0, sigma)
    return g


def crossover(g1, g2, rng):
    # 對齊較短的那個,逐單元隨機取;多出來的單元從較長的那個繼承
    short, long = (g1, g2) if genome_size(g1) <= genome_size(g2) else (g2, g1)
    child_hidden = []
    for i, u_long in enumerate(long["hidden"]):
        if i < len(short["hidden"]) and rng.random() < 0.5:
            src = short["hidden"][i]
        else:
            src = u_long
        child_hidden.append({"w": list(src["w"]), "out": src["out"]})
    return {
        "hidden": child_hidden,
        "out_bias": g1["out_bias"] if rng.random() < 0.5 else g2["out_bias"],
    }


# ============================================================
# Adam 局部精修(對可變 genome 的扁平化權重做梯度下降)
# ============================================================

def genome_to_flat(genome):
    flat = []
    for u in genome["hidden"]:
        flat.extend(u["w"])
        flat.append(u["out"])
    flat.append(genome["out_bias"])
    return flat


def flat_to_genome(flat, template):
    g = clone_genome(template)
    idx = 0
    for u in g["hidden"]:
        for i in range(len(u["w"])):
            u["w"][i] = flat[idx]; idx += 1
        u["out"] = flat[idx]; idx += 1
    g["out_bias"] = flat[idx]
    return g


def numerical_refine(genome, data, steps=20, lr=0.05):
    # 用有限差分梯度 + Adam 對 genome 做局部精修(結構不變,只調權重)
    flat = genome_to_flat(genome)
    n = len(flat)
    m = [0.0] * n
    v = [0.0] * n
    eps_fd = 1e-4
    pre = mse_loss(genome, data)
    for t in range(1, steps + 1):
        grad = [0.0] * n
        base = mse_loss(flat_to_genome(flat, genome), data)
        for i in range(n):
            flat[i] += eps_fd
            grad[i] = (mse_loss(flat_to_genome(flat, genome), data) - base) / eps_fd
            flat[i] -= eps_fd
        norm = math.sqrt(sum(g * g for g in grad))
        if norm > 5.0:
            grad = [g * 5.0 / norm for g in grad]
        for i in range(n):
            m[i] = 0.9 * m[i] + 0.1 * grad[i]
            v[i] = 0.999 * v[i] + 0.001 * grad[i] * grad[i]
            m_hat = m[i] / (1 - 0.9 ** t)
            v_hat = v[i] / (1 - 0.999 ** t)
            flat[i] -= lr * m_hat / (math.sqrt(v_hat) + 1e-8)
    refined = flat_to_genome(flat, genome)
    return refined if mse_loss(refined, data) < pre else genome


# ============================================================
# 演化族群(帶 self-adaptive sigma + 停滯偵測 + 結構多樣性保護)
# ============================================================

class Individual:
    __slots__ = ("genome", "sigma", "fitness")

    def __init__(self, genome, sigma):
        self.genome = genome
        self.sigma = sigma
        self.fitness = None


class NEATPopulation:
    def __init__(self, pop_size=24, sigma_init=0.4, stagnation_patience=15, seed=None):
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.stagnation_patience = stagnation_patience
        self.individuals = [Individual(new_genome(self.rng, 1), sigma_init) for _ in range(pop_size)]
        self.gens_since_improvement = 0
        self._last_best = None
        self.best = None

    def _shared_fitness(self):
        genomes = [ind.genome for ind in self.individuals]
        out = []
        for i, ind in enumerate(self.individuals):
            niche = sum(1 for g in genomes if genome_distance(genomes[i], g) < 1.5)
            out.append(ind.fitness / max(niche, 1))
        return out

    def _tournament(self, shared, k=3):
        idxs = self.rng.sample(range(len(self.individuals)), min(k, len(self.individuals)))
        return self.individuals[max(idxs, key=lambda i: shared[i])]

    def step(self, data, elitism=2):
        for ind in self.individuals:
            ind.fitness = 1.0 / (1.0 + mse_loss(ind.genome, data))
        order = sorted(range(len(self.individuals)), key=lambda i: self.individuals[i].fitness, reverse=True)
        self.best = self.individuals[order[0]]
        best_fit = self.best.fitness

        if self._last_best is not None and best_fit <= self._last_best + 1e-6:
            self.gens_since_improvement += 1
        else:
            self.gens_since_improvement = 0
        self._last_best = max(best_fit, self._last_best or 0.0)

        stagnation = self.gens_since_improvement >= self.stagnation_patience
        if stagnation:
            self.gens_since_improvement = 0
            for ind in self.individuals:
                ind.sigma = min(ind.sigma * 1.8, 1.5)

        shared = self._shared_fitness()
        nxt = [self.individuals[i] for i in order[:elitism]]  # elites by RAW fitness, structure preserved
        n_imm = 3 if stagnation else 0
        while len(nxt) < self.pop_size - n_imm:
            p1, p2 = self._tournament(shared), self._tournament(shared)
            child = crossover(p1.genome, p2.genome, self.rng)
            sigma = max(0.02, min(math.sqrt(p1.sigma * p2.sigma) * math.exp(0.2 * self.rng.gauss(0, 1)), 1.5))
            child = mutate(child, sigma, self.rng)
            nxt.append(Individual(child, sigma))
        while len(nxt) < self.pop_size:
            nxt.append(Individual(new_genome(self.rng, 1), 0.4))
        self.individuals = nxt
        return best_fit, stagnation


def evolve(pop, data, generations, refine_every=10):
    for gen in range(generations):
        pop.step(data)
        if refine_every and gen % refine_every == 0 and pop.best is not None:
            refined = numerical_refine(pop.best.genome, data, steps=15)
            pop.best.genome = refined
            pop.best.fitness = 1.0 / (1.0 + mse_loss(refined, data))


# ============================================================
# 任務池(輪換,避免過擬合單一任務)
# ============================================================

_XS = [-math.pi + i * (2 * math.pi / 39) for i in range(40)]
TASKS = {
    "a": [(x, math.sin(x) * math.cos(x / 2)) for x in _XS],
    "b": [(x, math.cos(2 * x) + 0.2 * x) for x in _XS],
    "c": [(x, math.sin(3 * x) * 0.5 + math.sin(x)) for x in _XS],
    "d": [(x, abs(x) / math.pi - 0.5) for x in _XS],
}
TASK_ORDER = ["a", "b", "c", "d"]


# ============================================================
# Redis 持久化
# ============================================================

REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
STATE_KEY = "dot_neat_state"
CHAT_KEY  = "dot_chat_history"
SYS_KEY   = "dot_system_prompt"

HF_TOKEN    = os.environ.get("HF_API_TOKEN")
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
GROQ_KEY         = os.environ.get("GROQ_API_KEY")
GROQ_MODEL        = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
CHAT_WORKER_URL   = os.environ.get("CHAT_WORKER_URL", "")

DEFAULT_SYSTEM = (
    "You are DOT, a concise and thoughtful AI assistant. "
    "You are honest about what you know and don't know. "
    "Keep replies short and direct unless asked to elaborate."
)

import urllib.request


def redis_command(*args):
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    req = urllib.request.Request(
        REDIS_URL,
        data=json.dumps(list(args)).encode(),
        headers={"Authorization": f"Bearer {REDIS_TOKEN}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def load_state():
    result = redis_command("GET", STATE_KEY)
    if not result or result.get("result") is None:
        return None
    return json.loads(result["result"])


def save_state(state):
    redis_command("SET", STATE_KEY, json.dumps(state))


# ============================================================
# Chat 層:先試 Groq,失敗才 fallback HF
# 用 urllib.request(跟 Redis 同一套),不用 asyncio
# ============================================================

def _call_api(url: str, token: str, payload: dict) -> str:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise ValueError(f"HTTP {e.code}: {body}")
    if "choices" not in data:
        raise ValueError(json.dumps(data)[:200])
    return data["choices"][0]["message"]["content"]


def sync_chat(messages: list, system: str) -> str:
    if not GROQ_KEY and not HF_TOKEN:
        return "⚠️ 請在 Vercel 環境變數設定 GROQ_API_KEY 或 HF_API_TOKEN。"
    payload_base = {
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }
    if GROQ_KEY:
        key_hint = GROQ_KEY[:8] + "..." if len(GROQ_KEY) > 8 else "?"
        try:
            return _call_api(
                "https://api.groq.com/openai/v1/chat/completions",
                GROQ_KEY,
                {**payload_base, "model": GROQ_MODEL},
            )
        except Exception as e:
            return f"⚠️ Groq error (key={key_hint}): {e}"
    try:
        return _call_api(
            "https://api-inference.huggingface.co/v1/chat/completions",
            HF_TOKEN,
            {**payload_base, "model": HF_MODEL_ID},
        )
    except Exception as e:
        return f"⚠️ HF error: {e}"


def get_system_prompt() -> str:
    result = redis_command("GET", SYS_KEY)
    if result and result.get("result"):
        return result["result"]
    return DEFAULT_SYSTEM


def save_system_prompt(prompt: str):
    redis_command("SET", SYS_KEY, prompt)


def get_chat_history() -> list:
    result = redis_command("GET", CHAT_KEY)
    if result and result.get("result"):
        return json.loads(result["result"])
    return []


def append_turns(turns: list):
    history = get_chat_history()
    history.extend(turns)
    redis_command("SET", CHAT_KEY, json.dumps(history[-200:]))


def pop_to_dict(pop):
    return {
        "pop_size": pop.pop_size,
        "stagnation_patience": pop.stagnation_patience,
        "gens_since_improvement": pop.gens_since_improvement,
        "last_best": pop._last_best,
        "individuals": [{"genome": ind.genome, "sigma": ind.sigma, "fitness": ind.fitness}
                        for ind in pop.individuals],
    }


def pop_from_dict(d):
    pop = NEATPopulation(pop_size=d["pop_size"], stagnation_patience=d["stagnation_patience"])
    pop.individuals = [Individual(r["genome"], r["sigma"]) for r in d["individuals"]]
    for ind, r in zip(pop.individuals, d["individuals"]):
        ind.fitness = r["fitness"]
    pop.gens_since_improvement = d["gens_since_improvement"]
    pop._last_best = d["last_best"]
    pop.best = max(pop.individuals, key=lambda i: i.fitness or -1)
    return pop


GENERATIONS_PER_RUN_DEFAULT = 40
GENERATIONS_PER_RUN_MAX = 300
SWITCH_EVERY = 500


def run_batch(generations: int) -> dict:
    state = load_state()
    redis_connected = bool(REDIS_URL and REDIS_TOKEN)

    if state is None:
        pop = NEATPopulation(pop_size=24)
        state = {"generation": 0, "task_idx": 0, "history": []}
    else:
        pop = pop_from_dict(state["population"])

    task_key = TASK_ORDER[state["task_idx"] % len(TASK_ORDER)]
    data = TASKS[task_key]

    evolve(pop, data, generations)

    prev = state["generation"]
    state["generation"] = prev + generations
    switched = False
    if prev // SWITCH_EVERY != state["generation"] // SWITCH_EVERY:
        state["task_idx"] += 1
        switched = True

    best_genome = pop.best.genome
    snapshot = {
        "gen": state["generation"],
        "task": task_key,
        "n_neurons": genome_size(best_genome),
        "fitness": round(pop.best.fitness, 4),
        "mse": round(mse_loss(best_genome, data), 4),
    }
    state["history"] = (state.get("history", []) + [snapshot])[-50:]
    state["population"] = pop_to_dict(pop)

    if redis_connected:
        save_state(state)

    return {
        "redis_connected": redis_connected,
        "total_generations": state["generation"],
        "ran_this_call": generations,
        "current_task": task_key,
        "switched_task_this_call": switched,
        "best_fitness": round(pop.best.fitness, 4),
        "n_neurons": genome_size(best_genome),
        "mse_current_task": round(mse_loss(best_genome, data), 4),
        "avg_neurons": round(sum(genome_size(i.genome) for i in pop.individuals) / len(pop.individuals), 2),
        "max_neurons": max(genome_size(i.genome) for i in pop.individuals),
    }


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI()
_HTML = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DOT</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fragment+Mono&family=Press+Start+2P&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0A0E14;--bg2:#0E131C;--card:#141A24;--card2:#1A2230;
  --border:#222C3A;--border-s:#1A222E;
  --ink:#E8EDF2;--ink-s:#8B97A8;--ink-f:#535E70;
  --rust:#D97757;--rust-b:#F08A66;
  --sage:#9DAE8B;--gold:#D9A441;--rose:#C77B6B;
  --mono:'Fragment Mono',monospace;--pixel:'Press Start 2P',monospace;--sans:'Inter',sans-serif;
}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;-webkit-font-smoothing:antialiased}
#stars{position:fixed;inset:0;z-index:0;pointer-events:none}

/* ── HEADER ── */
header{
  display:flex;align-items:center;height:38px;flex-shrink:0;
  background:var(--bg2);border-bottom:1px solid var(--border-s);z-index:10;position:relative;
}
.tab-btn{
  display:flex;align-items:center;gap:7px;padding:0 18px;height:100%;
  font-family:var(--mono);font-size:12px;color:var(--ink-f);
  border-right:1px solid var(--border-s);cursor:pointer;user-select:none;border:none;background:none;
}
.tab-btn.on{color:var(--ink);background:var(--card)}
.tdot{width:6px;height:6px;border-radius:50%;background:var(--ink-f)}
.tdot.live{background:var(--sage)}.tdot.err{background:var(--rose)}
.hdr-right{margin-left:auto;display:flex;align-items:center;gap:12px;padding-right:16px}
.state-txt{font-family:var(--mono);font-size:11px;color:var(--ink-s)}

/* ── PANELS ── */
.panel{flex:1;display:none;flex-direction:column;overflow:hidden;position:relative;z-index:1}
.panel.on{display:flex}

/* ════ NEAT TAB ════ */
.hero{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:20px;overflow-y:auto}
.brand-tag{font-family:var(--mono);font-size:11px;color:var(--ink-f);letter-spacing:.1em;margin-bottom:14px;text-align:center}
.brand{font-family:var(--pixel);font-size:34px;display:flex;gap:14px;align-items:baseline}
.brand .dot{color:var(--rust)}.brand .code{color:var(--ink-s)}
.ribbon{display:flex;gap:0;margin:22px 0 6px;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--card)}
.rcell{padding:10px 16px;border-right:1px solid var(--border-s);text-align:center;min-width:76px}
.rcell:last-child{border-right:none}
.rk{font-family:var(--mono);font-size:9px;color:var(--ink-f);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}
.rv{font-family:var(--mono);font-size:16px;color:var(--ink)}
.rv.sage{color:var(--sage)}.rv.rust{color:var(--rust)}.rv.gold{color:var(--gold)}.rv.faint{color:var(--ink-f)}
.icard{width:min(600px,92vw);margin-top:18px;background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.icard-top{display:flex;align-items:center;gap:10px;padding:13px 15px}
.badge{font-family:var(--mono);font-size:11px;color:var(--bg);background:var(--rust);padding:2px 8px;border-radius:5px;flex-shrink:0}
#ncmd{flex:1;background:none;border:none;outline:none;color:var(--ink);font-family:var(--mono);font-size:13.5px;caret-color:var(--rust)}
#ncmd::placeholder{color:var(--ink-f)}#ncmd:disabled{opacity:.5}
.nspin{display:none;color:var(--gold);font-family:var(--mono);animation:sp 1s linear infinite}
.icard-sub{padding:9px 15px;border-top:1px solid var(--border-s);font-family:var(--mono);font-size:11px;color:var(--ink-f)}
.icard-sub .a{color:var(--rust)}
.hints{display:flex;gap:20px;margin-top:16px;font-family:var(--mono);font-size:11px;color:var(--ink-f);flex-wrap:wrap;justify-content:center}
.hints b{color:var(--ink-s);font-weight:400}
.nlog{width:min(600px,92vw);margin-top:16px;max-height:28vh;overflow-y:auto;
  font-family:var(--mono);font-size:12px;line-height:1.75;border-top:1px solid var(--border-s);padding-top:10px}
.nlog:empty{display:none}
.nlog::-webkit-scrollbar{width:4px}.nlog::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.lr{display:flex;gap:10px;align-items:baseline;padding:1px 0}
.lt{color:var(--ink-f);font-size:10px;width:54px;flex-shrink:0}
.lc{color:var(--rust)}.lo{color:var(--ink)}.ls{color:var(--ink-s);font-style:italic}
.ler{color:var(--rose)}.lsw{color:var(--gold);font-weight:600}.lgrow{color:var(--sage);font-weight:600}
@keyframes sp{to{transform:rotate(360deg)}}

/* ════ CHAT TAB ════ */
.chat-wrap{flex:1;display:flex;flex-direction:column;overflow:hidden;max-width:780px;width:100%;margin:0 auto;padding:0 16px}

/* system prompt bar */
.sys-bar{flex-shrink:0;padding:10px 0 8px;border-bottom:1px solid var(--border-s)}
.sys-toggle{
  display:flex;align-items:center;gap:8px;cursor:pointer;
  font-family:var(--mono);font-size:11px;color:var(--ink-f);padding:3px 0;background:none;border:none;
}
.sys-toggle .arr{transition:transform .2s;color:var(--rust)}
.sys-toggle.open .arr{transform:rotate(90deg)}
.sys-editor{display:none;margin-top:8px;gap:6px;flex-direction:column}
.sys-editor.vis{display:flex}
#sys-input{
  width:100%;padding:10px 12px;background:var(--card);border:1px solid var(--border);border-radius:8px;
  color:var(--ink);font-family:var(--mono);font-size:12px;resize:vertical;min-height:72px;outline:none;
}
#sys-input:focus{border-color:var(--rust)}
.sys-btns{display:flex;gap:8px}
.sys-save{padding:6px 14px;background:var(--rust);color:var(--bg);font-family:var(--mono);font-size:12px;
  border:none;border-radius:6px;cursor:pointer;font-weight:600}
.sys-reset{padding:6px 14px;background:transparent;color:var(--ink-s);font-family:var(--mono);font-size:12px;
  border:1px solid var(--border);border-radius:6px;cursor:pointer}

/* messages */
.msgs{flex:1;overflow-y:auto;padding:16px 0;display:flex;flex-direction:column;gap:16px}
.msgs::-webkit-scrollbar{width:5px}.msgs::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.msg{display:flex;gap:10px;align-items:flex-start;max-width:100%}
.msg.user{flex-direction:row-reverse}
.avatar{width:28px;height:28px;border-radius:7px;flex-shrink:0;display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:10px;font-weight:700}
.avatar.user{background:var(--card2);color:var(--ink-s)}
.avatar.bot{background:var(--rust);color:var(--bg)}
.bubble{max-width:80%;padding:11px 14px;border-radius:12px;font-size:13.5px;line-height:1.65;white-space:pre-wrap;word-break:break-word}
.msg.user .bubble{background:var(--card2);color:var(--ink);border-bottom-right-radius:3px}
.msg.bot  .bubble{background:var(--card);color:var(--ink);border-bottom-left-radius:3px;border:1px solid var(--border-s)}
.typing .bubble{color:var(--ink-f);font-style:italic}

/* input */
.chat-input-wrap{flex-shrink:0;padding:12px 0 16px;border-top:1px solid var(--border-s)}
.chat-input-row{display:flex;align-items:flex-end;gap:10px;background:var(--card);
  border:1px solid var(--border);border-radius:12px;padding:10px 14px;transition:border-color .2s}
.chat-input-row:focus-within{border-color:var(--rust)}
#chat-in{
  flex:1;background:none;border:none;outline:none;resize:none;
  color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;
  max-height:140px;overflow-y:auto;caret-color:var(--rust);
}
#chat-in::placeholder{color:var(--ink-f)}
.send-btn{
  width:32px;height:32px;background:var(--rust);border:none;border-radius:8px;
  cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;
  color:var(--bg);font-size:14px;transition:opacity .15s;
}
.send-btn:disabled{opacity:.35;cursor:default}
.chat-hint{font-family:var(--mono);font-size:11px;color:var(--ink-f);text-align:center;margin-top:7px}
.chat-err{color:var(--rose);font-family:var(--mono);font-size:12px;margin-top:6px}

footer{
  padding:7px 16px;font-family:var(--mono);font-size:11px;color:var(--ink-f);
  display:flex;justify-content:space-between;border-top:1px solid var(--border-s);
  background:var(--bg2);z-index:10;flex-shrink:0;
}
</style>
</head>
<body>
<canvas id="stars"></canvas>

<header>
  <button class="tab-btn on" id="tab-neat" onclick="switchTab('neat')">
    <span class="tdot" id="neat-dot"></span>DOT NEAT
  </button>
  <button class="tab-btn" id="tab-chat" onclick="switchTab('chat')">
    <span class="tdot" id="chat-dot"></span>CHAT
  </button>
  <div class="hdr-right">
    <span class="state-txt" id="hdr-state">connecting…</span>
  </div>
</header>

<!-- ═══ NEAT PANEL ═══ -->
<div class="panel on" id="panel-neat">
  <div class="hero">
    <div class="brand-tag">NeuroEvolution of Augmenting Topologies</div>
    <div class="brand"><span class="dot">DOT</span><span class="code">NEAT</span></div>
    <div class="ribbon">
      <div class="rcell"><div class="rk">gen</div><div class="rv faint" id="r-gen">—</div></div>
      <div class="rcell"><div class="rk">neurons</div><div class="rv faint" id="r-neu">—</div></div>
      <div class="rcell"><div class="rk">fitness</div><div class="rv faint" id="r-fit">—</div></div>
      <div class="rcell"><div class="rk">task</div><div class="rv faint" id="r-task">—</div></div>
      <div class="rcell"><div class="rk">mse</div><div class="rv faint" id="r-mse">—</div></div>
    </div>
    <div class="icard">
      <div class="icard-top">
        <span class="badge">RUN</span>
        <input id="ncmd" placeholder="run [N] · status · help · clear" autocomplete="off" spellcheck="false">
        <span class="nspin" id="nspin">⟳</span>
      </div>
      <div class="icard-sub"><span class="a">›</span> DOT NEAT — 神經元隨演化長出與修剪，每 500 代換任務</div>
    </div>
    <div class="hints">
      <div><b>run N</b> 演化</div><div><b>status</b> 狀態</div>
      <div><b>↑↓</b> 歷史</div><div><b>clear</b> 清空</div>
    </div>
    <div class="nlog" id="nlog"></div>
  </div>
</div>

<!-- ═══ CHAT PANEL ═══ -->
<div class="panel" id="panel-chat">
  <div class="chat-wrap">

    <div class="sys-bar">
      <button class="sys-toggle" id="sys-toggle" onclick="toggleSys()">
        <span class="arr">▶</span><span>System Prompt</span>
        <span style="margin-left:6px;color:var(--rust);font-size:10px" id="sys-model"></span>
      </button>
      <div class="sys-editor" id="sys-editor">
        <textarea id="sys-input" rows="3" placeholder="在這裡輸入 system prompt…"></textarea>
        <div class="sys-btns">
          <button class="sys-save" onclick="saveSystem()">儲存</button>
          <button class="sys-reset" onclick="resetSystem()">還原預設</button>
        </div>
        <div class="chat-err" id="sys-err" style="display:none"></div>
      </div>
    </div>

    <div class="msgs" id="msgs"></div>

    <div class="chat-input-wrap">
      <div class="chat-input-row">
        <textarea id="chat-in" rows="1" placeholder="輸入訊息…（Enter 送出，Shift+Enter 換行）"></textarea>
        <button class="send-btn" id="send-btn" onclick="sendMsg()" title="送出">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
            <path d="M1 7h12M7 1l6 6-6 6" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </button>
      </div>
      <div class="chat-hint" id="chat-hint">model: <span id="model-hint">—</span> · <span id="hf-hint">checking…</span></div>
    </div>

  </div>
</div>

<footer>
  <span>~/dot:main</span>
  <span id="foot-txt">DOT v0.3.0</span>
  <span id="foot-r">0.3.0</span>
</footer>

<script>
// ── Starfield ──────────────────────────────────────────────
const cv=document.getElementById('stars'),cx=cv.getContext('2d');
let stars=[];
function resize(){cv.width=innerWidth;cv.height=innerHeight;
  stars=Array.from({length:Math.floor(innerWidth*innerHeight/9000)},()=>({
    x:Math.random()*cv.width,y:Math.random()*cv.height,r:Math.random()*1.2+0.2,
    a:Math.random()*.5+.15,tw:Math.random()*.02+.005}));}
resize();addEventListener('resize',resize);
(function draw(){cx.clearRect(0,0,cv.width,cv.height);
  for(const s of stars){s.a+=s.tw;if(s.a>.7||s.a<.1)s.tw*=-1;
    cx.fillStyle=`rgba(217,119,87,${s.a*.5})`;cx.beginPath();cx.arc(s.x,s.y,s.r,0,7);cx.fill();}
  requestAnimationFrame(draw);})();

// ── Tab switching ──────────────────────────────────────────
function switchTab(t){
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('on'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('on'));
  document.getElementById('tab-'+t).classList.add('on');
  document.getElementById('panel-'+t).classList.add('on');
  if(t==='chat'){loadSystem();document.getElementById('chat-in').focus();}
}

// ── NEAT tab ──────────────────────────────────────────────
const $nlog=document.getElementById('nlog'),$ncmd=document.getElementById('ncmd'),$nspin=document.getElementById('nspin');
let nhist=[],nhIdx=-1,nbusy=false,ncd=30,ntick=null,lastNeu=null;
const ts=()=>new Date().toTimeString().slice(0,8);
function nrow(h,c){const d=document.createElement('div');d.className='lr';
  d.innerHTML=`<span class="lt">${ts()}</span><span class="${c}">${h}</span>`;$nlog.appendChild(d);$nlog.scrollTop=$nlog.scrollHeight;}
function nplain(h){const d=document.createElement('div');d.className='lr';
  d.innerHTML=`<span class="lt"></span><span>${h}</span>`;$nlog.appendChild(d);$nlog.scrollTop=$nlog.scrollHeight;}
function applyNeat(d){
  const fit=d.best_fitness??0,gen=d.total_generations??0,neu=d.n_neurons??null;
  const g=document.getElementById('r-gen');g.textContent=gen;g.className='rv sage';
  const nu=document.getElementById('r-neu');nu.textContent=neu??'—';nu.className='rv rust';
  const f=document.getElementById('r-fit');f.textContent=fit.toFixed(3);f.className='rv '+(fit>.95?'sage':fit>.7?'rust':'gold');
  const t=document.getElementById('r-task');t.textContent=d.current_task?'task_'+d.current_task:'—';t.className='rv '+(d.current_task?'rust':'faint');
  document.getElementById('r-mse').textContent=(d.mse_current_task??d.mse_task_a)?.toFixed(3)??'—';
  document.getElementById('hdr-state').textContent='live';
  document.getElementById('neat-dot').classList.add('live');
}
function setNBusy(v){nbusy=v;$nspin.style.display=v?'inline':'none';$ncmd.disabled=v;}
function resetNcd(){clearInterval(ntick);ncd=30;ntick=setInterval(()=>{ncd--;if(ncd<=0){doNStatus(true);resetNcd();}},1000);}
async function doNStatus(silent){
  try{const r=await fetch('/api/status');if(!r.ok)throw r.status;const d=await r.json();
    if(d.status==='no_data'){document.getElementById('hdr-state').textContent='no data';return;}
    applyNeat(d);if(!silent)nrow(JSON.stringify(d),'lo');
  }catch(e){document.getElementById('hdr-state').textContent='err';if(!silent)nrow('status failed: '+e,'ler');}
}
async function doNRun(n){
  if(nbusy){nrow('busy — wait','ls');return;}
  setNBusy(true);nrow('run '+n,'lc');nrow('evolving '+n+' generations…','ls');
  try{const r=await fetch('/api/evolve?generations='+n);if(!r.ok)throw r.status;const d=await r.json();
    applyNeat(d);
    let growth='';
    if(lastNeu!==null&&d.n_neurons!==null){
      if(d.n_neurons>lastNeu)growth=` <span class="lgrow">↑ grew ${d.n_neurons-lastNeu} → ${d.n_neurons}</span>`;
      else if(d.n_neurons<lastNeu)growth=` <span class="ls">↓ pruned → ${d.n_neurons}</span>`;
    }
    lastNeu=d.n_neurons;
    const sw=d.switched_task_this_call?` <span class="lsw">★ task → ${d.current_task}</span>`:'';
    nrow(`done · gen ${d.total_generations} · n=${d.n_neurons}(avg ${d.avg_neurons} max ${d.max_neurons}) · fit ${d.best_fitness} · mse ${d.mse_current_task}${growth}${sw}`,'lo');
  }catch(e){nrow('error: '+e,'ler');}
  setNBusy(false);resetNcd();
}
function ncmd(raw){const p=raw.trim().split(/\s+/),c=p[0].toLowerCase();
  if(c==='run')doNRun(Math.max(1,Math.min(parseInt(p[1])||40,300)));
  else if(c==='status'){nrow('status','lc');doNStatus(false);}
  else if(c==='clear')$nlog.innerHTML='';
  else if(c==='help'){nplain('<span style="color:var(--rust)">run [N]</span> 演化N代');
    nplain('<span style="color:var(--rust)">status</span> 讀狀態');nplain('<span style="color:var(--rust)">clear</span> 清空');}
  else nrow('unknown: '+c,'ler');}
$ncmd.addEventListener('keydown',e=>{
  if(e.key==='Enter'){const v=$ncmd.value.trim();if(!v)return;nhist.unshift(v);nhIdx=-1;$ncmd.value='';ncmd(v);}
  else if(e.key==='ArrowUp'){e.preventDefault();nhIdx=Math.min(nhIdx+1,nhist.length-1);$ncmd.value=nhist[nhIdx]??'';}
  else if(e.key==='ArrowDown'){e.preventDefault();nhIdx=Math.max(nhIdx-1,-1);$ncmd.value=nhIdx<0?'':nhist[nhIdx];}});

// ── Chat tab ──────────────────────────────────────────────
const $msgs=document.getElementById('msgs');
let chatHistory=[], chatBusy=false, currentSystem='', chatWorkerUrl='';

function toggleSys(){
  const t=document.getElementById('sys-toggle');
  const e=document.getElementById('sys-editor');
  t.classList.toggle('open');
  e.classList.toggle('vis');
}
async function loadSystem(){
  try{const r=await fetch('/api/system');const d=await r.json();
    currentSystem=d.prompt;chatWorkerUrl=d.chat_worker_url||'';
    document.getElementById('sys-input').value=d.prompt;
    document.getElementById('sys-model').textContent=d.model||'';
    document.getElementById('model-hint').textContent=d.model||'—';
    document.getElementById('hf-hint').textContent=d.hf_ready?'HF ✓':'HF_API_TOKEN 未設定';
  }catch(e){}
}
async function saveSystem(){
  const p=document.getElementById('sys-input').value.trim();
  if(!p)return;
  const e=document.getElementById('sys-err');
  try{const r=await fetch('/api/system',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:p})});
    const d=await r.json();if(d.ok){currentSystem=p;e.style.display='none';toggleSys();}
    else{e.textContent=d.error||'error';e.style.display='';}
  }catch(err){e.textContent=String(err);e.style.display='';}
}
async function resetSystem(){
  const r=await fetch('/api/system/reset',{method:'POST'});
  const d=await r.json();
  currentSystem=d.prompt;document.getElementById('sys-input').value=d.prompt;
  document.getElementById('sys-err').style.display='none';
}
function addMsg(role,content,typing=false){
  const wrap=document.createElement('div');
  wrap.className='msg '+(role==='user'?'user':'bot')+(typing?' typing':'');
  const av=document.createElement('div');av.className='avatar '+(role==='user'?'user':'bot');
  av.textContent=role==='user'?'你':'D';
  const bub=document.createElement('div');bub.className='bubble';bub.textContent=content;
  wrap.appendChild(av);wrap.appendChild(bub);
  $msgs.appendChild(wrap);$msgs.scrollTop=$msgs.scrollHeight;
  return bub;
}
async function sendMsg(){
  const inp=document.getElementById('chat-in');
  const txt=inp.value.trim();if(!txt||chatBusy)return;
  inp.value='';autoResize();chatBusy=true;
  document.getElementById('send-btn').disabled=true;
  addMsg('user',txt);
  chatHistory.push({role:'user',content:txt});
  const typBub=addMsg('bot','thinking…',true);
  try{
    const chatUrl=chatWorkerUrl||'/api/chat';
    const r=await fetch(chatUrl,{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:chatHistory.slice(-6),system:currentSystem})});
    const txt=await r.text();
    let d;
    try{d=JSON.parse(txt);}catch(e){
      typBub.textContent='⚠️ 伺服器回傳非 JSON (HTTP '+r.status+')，請查看 Vercel 函式日誌。';
      typBub.parentElement.classList.remove('typing');
      chatBusy=false;document.getElementById('send-btn').disabled=false;
      inp.focus();return;
    }
    if(d.reply){
      typBub.textContent=d.reply;typBub.parentElement.classList.remove('typing');
      const ok=!d.reply.startsWith('⚠️')&&!d.reply.startsWith('⏳');
      if(ok){
        chatHistory.push({role:'assistant',content:d.reply});
        // 儲存對話到 Redis(非阻塞)
        fetch('/api/chat/store',{method:'POST',headers:{'Content-Type':'application/json'},
          body:JSON.stringify({turns:[{role:'user',content:txt},{role:'assistant',content:d.reply}]})
        }).catch(()=>{});
      }
    } else {
      typBub.textContent='⚠️ '+(d.error||JSON.stringify(d));
      typBub.parentElement.classList.remove('typing');
    }
  }catch(e){typBub.textContent='⚠️ fetch error: '+e;typBub.parentElement.classList.remove('typing');}
  chatBusy=false;document.getElementById('send-btn').disabled=false;
  inp.focus();$msgs.scrollTop=$msgs.scrollHeight;
}
function autoResize(){const t=document.getElementById('chat-in');t.style.height='auto';t.style.height=Math.min(t.scrollHeight,140)+'px';}
document.getElementById('chat-in').addEventListener('input',autoResize);
document.getElementById('chat-in').addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMsg();}});

// ── Boot ──────────────────────────────────────────────────
doNStatus(true);resetNcd();loadSystem();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def terminal():
    return HTMLResponse(_HTML)


@app.get("/api/status")
async def status_endpoint():
    redis_connected = bool(REDIS_URL and REDIS_TOKEN)
    state = load_state()
    if state is None:
        return JSONResponse({"redis_connected": redis_connected, "status": "no_data"})
    pop = pop_from_dict(state["population"])
    task_key = TASK_ORDER[state["task_idx"] % len(TASK_ORDER)]
    data = TASKS[task_key]
    best = pop.best.genome
    return JSONResponse({
        "redis_connected": redis_connected,
        "total_generations": state["generation"],
        "current_task": task_key,
        "best_fitness": round(pop.best.fitness or 0, 4),
        "n_neurons": genome_size(best),
        "mse_current_task": round(mse_loss(best, data), 4),
        "avg_neurons": round(sum(genome_size(i.genome) for i in pop.individuals) / len(pop.individuals), 2),
        "max_neurons": max(genome_size(i.genome) for i in pop.individuals),
        "history": state.get("history", [])[-12:],
    })


@app.get("/api/evolve")
async def evolve_endpoint(request: Request, generations: int = Query(default=GENERATIONS_PER_RUN_DEFAULT)):
    cron_secret = os.environ.get("CRON_SECRET")
    if cron_secret:
        if request.headers.get("authorization", "") != f"Bearer {cron_secret}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    generations = max(1, min(generations, GENERATIONS_PER_RUN_MAX))
    try:
        return JSONResponse(run_batch(generations))
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/system")
async def get_system_endpoint():
    using_groq = bool(GROQ_KEY)
    active_model = GROQ_MODEL if using_groq else HF_MODEL_ID
    provider = "Groq" if using_groq else "HuggingFace"
    return JSONResponse({
        "prompt": get_system_prompt(),
        "model": active_model,
        "provider": provider,
        "hf_token": HF_TOKEN or "",
        "hf_ready": bool(HF_TOKEN) or using_groq or bool(CHAT_WORKER_URL),
        "chat_worker_url": CHAT_WORKER_URL,
    })


@app.post("/api/system")
async def set_system_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    prompt = (body.get("prompt") or "").strip()
    if not prompt:
        return JSONResponse({"error": "prompt cannot be empty"}, status_code=400)
    save_system_prompt(prompt)
    return JSONResponse({"ok": True})


@app.post("/api/system/reset")
async def reset_system_endpoint():
    save_system_prompt(DEFAULT_SYSTEM)
    return JSONResponse({"ok": True, "prompt": DEFAULT_SYSTEM})


@app.post("/api/chat/store")
async def chat_store_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    turns = body.get("turns", [])
    if turns:
        try:
            append_turns(turns)
        except Exception:
            pass
    return JSONResponse({"ok": True})


@app.get("/api/chat/history")
async def chat_history_endpoint():
    return JSONResponse({"history": get_chat_history(), "model": GROQ_MODEL if GROQ_KEY else HF_MODEL_ID})

