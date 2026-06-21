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
import ast as _ast
import base64
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
    "a": [(x, math.sin(17 * x) * math.cos(11 * x) + 0.6 * x**5 + 0.4 * abs(x**3) + 0.2 * x**7) for x in _XS],
    "b": [(x, math.sin(13 * x) * math.cos(7 * x) + 0.4 * x**3 + 0.2 * abs(x) + 0.1 * x**5) for x in _XS],
    "c": [(x, math.sin(23 * x) * math.cos(13 * x) + 0.7 * x**6 + 0.4 * abs(x**4) + 0.1 * x**8) for x in _XS],
    "d": [(x, math.sin(7 * x) * math.cos(5 * x) + 0.5 * x**4 + 0.4 * abs(x**2) + 0.2 * x**6) for x in _XS],
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
MEMORY_KEY = "dot_agent_memory"

GROQ_MODEL        = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")  # 僅作為 Worker 查不到時的顯示用預設值
CHAT_WORKER_URL   = os.environ.get("CHAT_WORKER_URL", "")
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO       = os.environ.get("GITHUB_REPO", "")  # "username/repo"
IMPROVE_SECRET    = os.environ.get("IMPROVE_SECRET", "")

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


def get_agent_memory() -> list:
    result = redis_command("GET", MEMORY_KEY)
    if result and result.get("result"):
        return json.loads(result["result"])
    return []


def save_agent_memory(memory: list):
    # 跟 Worker 的 remember 工具裡的上限一致(50 筆),避免無限長
    redis_command("SET", MEMORY_KEY, json.dumps(memory[-50:]))


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
<title>DOT · Fabrica</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fragment+Mono&family=Press+Start+2P&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  /* ── Fictional Typeface 設計語言移植:flat 色塊 / 零陰影 / pill 形狀 / 暴力字級跳躍 / 彈性動效 ── */
  --bg:#0A0908;--bg1:#0F0D0A;--bg2:#171310;
  --line:#2A241C;--line-soft:#1C1812;
  --rust:#FF6B3D;--rust-deep:#D9501F;
  --gold:#FFC93C;--sage:#5EE291;--rose:#FF5C7A;
  --ink:#F5F0E8;--ink-soft:#B8AFA0;--ink-mute:#5C5347;--ink-faint:#332D24;
  --mono:'Fragment Mono',monospace;--pixel:'Press Start 2P',monospace;
  --bounce:cubic-bezier(.34,1.56,.64,1);
}
html,body{height:100%;overflow:hidden}
body{background:var(--bg);color:var(--ink);font-family:var(--mono);font-size:13px;
  display:flex;flex-direction:column;height:100vh;-webkit-font-smoothing:antialiased}

/* ── BRAND: 暴力字級跳躍,單一字體用到底但尺度誇張 ── */
.brand{flex-shrink:0;padding:18px 24px 10px;display:flex;align-items:baseline;gap:12px;position:relative}
.brand-name{font-family:var(--pixel);font-size:26px;color:var(--rust);letter-spacing:-.01em;position:relative}
.brand-spark{position:absolute;top:-6px;right:-14px;color:var(--gold);font-size:11px;transform:rotate(12deg);user-select:none}
.brand-tag{font-size:10px;color:var(--ink-mute);letter-spacing:.16em;text-transform:uppercase;
  background:var(--bg2);border:1px solid var(--line);border-radius:999px;padding:3px 10px}

/* ── TERMINAL OUTPUT ── */
.term{flex:1;overflow-y:auto;padding:8px 24px 6px;display:flex;flex-direction:column;gap:3px}
.term::-webkit-scrollbar{width:3px}.term::-webkit-scrollbar-thumb{background:var(--line)}
.msg{display:flex;gap:8px;align-items:flex-start;padding:3px 0;line-height:1.65;font-size:13px}
.mtag{flex-shrink:0;font-size:9px;font-weight:700;letter-spacing:.04em;padding:2px 7px;border-radius:999px;
  text-transform:uppercase;line-height:1.5;margin-top:1px;white-space:nowrap}
.mc{word-break:break-word;white-space:pre-wrap;flex:1;padding-top:1px}
.ts{font-size:10px;color:var(--ink-faint);flex-shrink:0;margin-top:3px;width:34px}

.msg-user .mtag{background:var(--bg2);color:var(--ink-soft);border:1px solid var(--line)}
.msg-user .mc{color:var(--ink-soft)}
.msg-dot .mtag{background:var(--rust);color:var(--bg)}
.msg-dot .mc{color:var(--ink)}
.msg-sys .mtag{background:transparent;color:var(--ink-faint)}
.msg-sys .mc{color:var(--ink-mute);font-size:11px}
.msg-err .mtag{background:var(--rose);color:var(--bg)}
.msg-err .mc{color:var(--rose)}
.msg-thought .mtag{background:var(--gold);color:var(--bg)}
.msg-thought .mc{color:var(--gold);font-size:12px}
.msg-action .mtag{background:var(--rust);color:var(--bg)}
.msg-action .mc{color:var(--rust);font-size:12px}
.msg-obs .mtag{background:var(--sage);color:var(--bg)}
.msg-obs .mc{color:var(--sage);font-size:12px}
.msg-final .mtag{background:var(--rust);color:var(--bg)}
.msg-final .mc{color:var(--ink);font-size:13px}

/* ── STATUS BAR: pill 徽章取代純文字 ── */
.statusbar{flex-shrink:0;padding:8px 24px;display:flex;gap:7px;align-items:center;flex-wrap:wrap;
  border-top:1px solid var(--line);background:var(--bg1)}
.sb-pill{font-size:10px;padding:3px 10px;border-radius:999px;border:1px solid var(--line);
  color:var(--ink-mute);background:var(--bg2);font-weight:600;letter-spacing:.02em}
.sb-pill.on{background:var(--rust);color:var(--bg);border-color:var(--rust)}
.sb-pill.live{background:var(--sage);color:var(--bg);border-color:var(--sage)}
.sb-pill.idle{background:var(--bg2);color:var(--ink-faint)}

/* ── COMMAND PALETTE ── */
.cmdpal{position:absolute;bottom:100%;left:0;right:0;background:var(--bg1);
  border:1.5px solid var(--line);border-bottom:none;border-radius:14px 14px 0 0;z-index:100;display:none;
  max-height:350px;overflow-y:auto;overflow:hidden}
.cmdpal.vis{display:block}
.cpi{display:flex;gap:14px;align-items:center;padding:8px 14px;cursor:pointer;transition:background .12s}
.cpi:hover,.cpi.sel{background:var(--bg2)}
.cpk{flex-shrink:0;font-size:10px;font-weight:700;color:var(--bg);background:var(--rust);
  padding:2px 9px;border-radius:999px;letter-spacing:.02em}
.cpd{color:var(--ink-mute);font-size:12px}

/* ── INPUT ── */
.input-section{flex-shrink:0;position:relative}
.input-border{margin:0 24px 0 24px;padding:11px 16px;border-radius:14px;
  border:2px solid var(--line);background:var(--bg1);
  display:flex;align-items:flex-start;gap:10px;
  transition:border-color .25s var(--bounce),background .25s}
.input-border.agent{border-color:var(--rust);background:rgba(255,107,61,.06)}
.input-border.busy{animation:pulse 1s var(--bounce) infinite}
@keyframes pulse{0%,100%{border-color:var(--rust)}50%{border-color:var(--gold)}}
.mode-chr{font-family:var(--pixel);color:var(--rust);font-size:12px;flex-shrink:0;margin-top:2px;
  transition:transform .3s var(--bounce)}
#inp{flex:1;background:none;border:none;outline:none;color:var(--ink);font-family:var(--mono);
  font-size:13px;resize:none;max-height:120px;caret-color:var(--rust);line-height:1.55}
#inp::placeholder{color:var(--ink-faint)}
.model-line{padding:7px 16px 0 calc(24px + 4px);font-size:10px;color:var(--ink-mute);display:flex;gap:8px;align-items:center}
.ml-pill{background:var(--bg2);border:1px solid var(--line);border-radius:999px;padding:2px 9px;color:var(--ink-soft)}

/* ── HINTS ── */
.hints{flex-shrink:0;padding:10px 24px 14px;display:flex;gap:8px;flex-wrap:wrap}
.hint-chip{font-size:10px;color:var(--ink-mute);background:var(--bg1);border:1px solid var(--line-soft);
  border-radius:999px;padding:3px 10px}
.hint-chip b{color:var(--rust);font-weight:700}

/* ── MEMORY PANEL: 對話泡泡風格,呼應 comic speech-bubble 但保持機能性 ── */
.mem-panel{position:fixed;inset:0;z-index:200;background:rgba(10,9,8,.85);display:none;
  align-items:center;justify-content:center}
.mem-panel.vis{display:flex}
.mem-box{background:var(--bg1);border:2px solid var(--line);border-radius:18px;
  width:min(500px,94vw);max-height:70vh;display:flex;flex-direction:column;overflow:hidden}
.mem-head{padding:14px 16px;border-bottom:1.5px solid var(--line);display:flex;justify-content:space-between;
  align-items:center}
.mh-t{font-size:11px;font-weight:700;color:var(--bg);background:var(--rust);padding:3px 11px;
  border-radius:999px;letter-spacing:.03em;text-transform:uppercase}
.mem-close{cursor:pointer;color:var(--ink-mute);font-size:18px;line-height:1;
  width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:999px;
  transition:background .15s}
.mem-close:hover{background:var(--bg2);color:var(--ink)}
.mem-list{overflow-y:auto;padding:10px;flex:1;display:flex;flex-direction:column;gap:7px}
.mem-item{display:flex;gap:9px;align-items:flex-start;padding:9px 12px;background:var(--bg2);
  border-radius:12px 12px 12px 3px;font-size:11.5px;border:1px solid var(--line-soft)}
.mem-idx{color:var(--bg);background:var(--gold);flex-shrink:0;width:18px;height:18px;border-radius:999px;
  font-size:9px;font-weight:700;display:flex;align-items:center;justify-content:center}
.mem-text{color:var(--ink-soft);flex:1;word-break:break-word}
.mem-del{color:var(--ink-faint);cursor:pointer;flex-shrink:0;transition:color .15s}
.mem-del:hover{color:var(--rose)}
.mem-empty{color:var(--ink-mute);font-size:11px;padding:20px;text-align:center;line-height:1.7}
.mem-footer{padding:10px;border-top:1.5px solid var(--line);display:flex;gap:8px}
.mem-add-inp{flex:1;background:var(--bg);border:1.5px solid var(--line);border-radius:999px;padding:8px 14px;
  color:var(--ink);font-family:var(--mono);font-size:11px;outline:none;transition:border-color .15s}
.mem-add-inp:focus{border-color:var(--rust)}
.mem-add-btn{padding:8px 16px;background:var(--rust);color:var(--bg);font-family:var(--mono);font-size:11px;
  font-weight:700;border:none;border-radius:999px;cursor:pointer;transition:transform .15s var(--bounce)}
.mem-add-btn:active{transform:scale(.92)}
</style>
</head>
<body>
<div class="brand">
  <span class="brand-name">DOT<span class="brand-spark">✦</span></span>
  <span class="brand-tag">Fabrica · Autonomous Agent</span>
</div>
<div class="term" id="term"></div>
<div class="statusbar">
  <span class="sb-pill on" id="sb-mode">CHAT</span>
  <span class="sb-pill idle" id="sb-model">loading…</span>
  <span class="sb-pill idle" id="sb-gen">NEAT —</span>
  <span class="sb-pill idle" id="sb-mem">0 memories</span>
  <span class="sb-pill idle" id="sb-live">connecting…</span>
</div>
<div class="input-section">
  <div class="cmdpal" id="cmdpal"></div>
  <div class="input-border" id="input-border">
    <span class="mode-chr" id="mode-chr">›</span>
    <textarea id="inp" rows="1" placeholder="Type your message… (/ commands · tab switch mode)"></textarea>
  </div>
  <div class="model-line"><span class="ml-pill" id="ml-mode">CHAT</span><span id="ml-model">—</span></div>
</div>
<div class="hints">
  <span class="hint-chip"><b>tab</b> switch mode</span>
  <span class="hint-chip"><b>ctrl+p</b> memory</span>
  <span class="hint-chip"><b>/</b> commands</span>
  <span class="hint-chip"><b>$goal</b> agent</span>
  <span class="hint-chip"><b>↑↓</b> history</span>
</div>
<div class="mem-panel" id="mem-panel">
  <div class="mem-box">
    <div class="mem-head"><span class="mh-t">Agent Memory</span><span class="mem-close" onclick="closeMemory()">✕</span></div>
    <div class="mem-list" id="mem-list"></div>
    <div class="mem-footer">
      <input class="mem-add-inp" id="mem-add-inp" placeholder="Add a memory fact…" autocomplete="off">
      <button class="mem-add-btn" onclick="addMemoryItem()">Add</button>
    </div>
  </div>
</div>
<script>
var mode='chat',busy=false,workerUrl='',sysPrompt='',model='';
var chatHistory=[],agentMemory=[],inputHist=[],histIdx=-1,cmdSelIdx=-1;
try{agentMemory=JSON.parse(sessionStorage.getItem('dot_memory')||'[]');}catch(e){}
updateMemCount();

var CMDS=[
  {k:'/agent',  d:'切換到 Agent 自主模式'},
  {k:'/chat',   d:'切換到對話模式'},
  {k:'/evolve', d:'/evolve [N]  跑 N 代 NEAT 演化'},
  {k:'/status', d:'顯示 NEAT 演化狀態'},
  {k:'/improve',d:'DOT 自動分析並 commit 改進'},
  {k:'/memory', d:'顯示/編輯 agent 記憶'},
  {k:'/clear',  d:'清空終端'},
  {k:'/help',   d:'顯示所有指令'},
];

function ts(){return new Date().toTimeString().slice(0,5)}
function addMsg(type,tag,content){
  var $t=document.getElementById('term');
  var d=document.createElement('div');d.className='msg msg-'+type;
  var s=(content||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\n/g,'<br>');
  d.innerHTML='<span class="ts">'+ts()+'</span><span class="mtag">'+tag+'</span><span class="mc">'+s+'</span>';
  $t.appendChild(d);$t.scrollTop=$t.scrollHeight;
  return d.querySelector('.mc');
}
function sys(m){addMsg('sys','·',m)}
function err(m){addMsg('err','ERR',m)}
function clearTerm(){document.getElementById('term').innerHTML='';sys('Terminal cleared')}

function toggleMode(m){
  mode=m||(mode==='chat'?'agent':'chat');
  var ib=document.getElementById('input-border');
  var chr=document.getElementById('mode-chr');
  var inp=document.getElementById('inp');
  var sbm=document.getElementById('sb-mode');
  if(mode==='agent'){
    ib.classList.add('agent');chr.textContent='$';
    inp.placeholder='Enter goal… (DOT will plan, search, spawn agents, iterate)';
    sbm.textContent='AGENT';
  }else{
    ib.classList.remove('agent');chr.textContent='›';
    inp.placeholder='Type your message… (/ commands · tab switch mode)';
    sbm.textContent='CHAT';
  }
  document.getElementById('ml-mode').textContent=mode.toUpperCase();
  sys('Mode → '+mode.toUpperCase());
}

function showCmdPal(filter){
  var pal=document.getElementById('cmdpal');
  var f=CMDS.filter(function(c){return !filter||c.k.startsWith(filter)});
  if(!f.length){pal.classList.remove('vis');return}
  pal.innerHTML='';cmdSelIdx=-1;
  f.forEach(function(c){
    var d=document.createElement('div');d.className='cpi';
    d.innerHTML='<span class="cpk">'+c.k+'</span><span class="cpd">'+c.d+'</span>';
    d.onclick=function(){runCmd(c.k,'');hideCmdPal()};
    pal.appendChild(d);
  });
  pal.classList.add('vis');
}
function hideCmdPal(){document.getElementById('cmdpal').classList.remove('vis');cmdSelIdx=-1}

function runCmd(key,args){
  if(key==='/agent')toggleMode('agent');
  else if(key==='/chat')toggleMode('chat');
  else if(key==='/clear')clearTerm();
  else if(key==='/memory')openMemory();
  else if(key==='/help'){sys('Commands:');CMDS.forEach(function(c){addMsg('sys','·',c.k+'  —  '+c.d)})}
  else if(key==='/status')cmdStatus();
  else if(key==='/improve')cmdImprove();
  else if(key==='/evolve')cmdEvolve(parseInt(args)||100);
  else err('Unknown command: '+key+'. /help');
}

async function cmdEvolve(n){
  sys('Running '+n+' NEAT generations…');
  try{var r=await fetch('/api/evolve?generations='+n);var d=await r.json();
    sys('NEAT gen='+d.total_generations+' fit='+d.best_fitness+' neurons='+d.n_neurons);updateStatus(d);}
  catch(e){err('evolve: '+e)}
}
async function cmdStatus(){
  try{var r=await fetch('/api/status');var d=await r.json();
    if(d.status==='no_data'){sys('No NEAT data');return}
    sys('NEAT: gen='+d.total_generations+' fit='+d.best_fitness+' n='+d.n_neurons+' mse='+d.mse_current_task);updateStatus(d);}
  catch(e){err('status: '+e)}
}
async function cmdImprove(){
  sys('DOT analyzing and proposing code changes…');
  try{var r=await fetch('/api/self-improve',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    var d=await r.json();
    if(d.ok){sys('✓ '+d.expected_effect);if(d.analysis)addMsg('sys','·',d.analysis)}
    else{err(d.error||'failed');if(d.analysis)addMsg('sys','·',d.analysis)}}
  catch(e){err('improve: '+e)}
}

async function sendChat(text){
  if(!workerUrl){err('CHAT_WORKER_URL not set');return}
  chatHistory.push({role:'user',content:text});
  var bub=addMsg('dot','DOT','…');
  try{
    var memCtx=agentMemory.length?'\n\n[Context Memory]\n'+agentMemory.map(function(m,i){return (i+1)+'. '+m}).join('\n'):'';
    var r=await fetch(workerUrl,{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({messages:chatHistory.slice(-12),system:sysPrompt+memCtx})});
    var d=await r.json();var reply=d.reply||d.error||'no response';
    bub.innerHTML=reply.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'<br>');
    if(d.provider)sys('via '+d.provider);
    chatHistory.push({role:'assistant',content:reply});
    fetch('/api/chat/store',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({turns:[{role:'user',content:text},{role:'assistant',content:reply}]})}).catch(function(){});
  }catch(e){bub.textContent='⚠️ '+e}
}

async function sendAgent(goal){
  if(!workerUrl){err('CHAT_WORKER_URL not set');return}
  sys('Agent activated → planning…');
  document.getElementById('input-border').classList.add('busy');
  try{
    var r=await fetch(workerUrl.replace(/\/$/,'')+'/agent',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({goal:goal,context:sysPrompt,memory:agentMemory,history:chatHistory.slice(-8)})});
    var d=await r.json();
    document.getElementById('input-border').classList.remove('busy');
    if(d.steps)d.steps.forEach(function(s){
      if(s.type==='step'){
        if(s.thought)addMsg('thought','THINK',s.thought);
        if(s.action)addMsg('action','RUN',s.action+(s.input?' → '+String(s.input).slice(0,80):''));
      }else if(s.type==='observation'){
        addMsg('obs','OBS · '+s.tool,String(s.result||'').slice(0,200));
      }else if(s.type==='error'){
        addMsg('err','ERR',String(s.content||'').slice(0,200));
      }else if(s.type==='loop_break'){
        sys(s.content||'迴圈中斷');
      }else if(s.type==='format_retry'){
        sys('格式不符，要求重新輸出 → '+String(s.content||'').slice(0,100));
      }
    });
    if(d.memory&&d.memory.length){agentMemory=d.memory;saveMemory();}
    if(d.provider){sys('via '+d.provider)}
    if(d.final_answer){
      addMsg('final','DOT',d.final_answer);
      chatHistory.push({role:'user',content:'[AGENT] '+goal});
      chatHistory.push({role:'assistant',content:d.final_answer});
      fetch('/api/chat/store',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({turns:[{role:'user',content:'[AGENT] '+goal},{role:'assistant',content:d.final_answer}]})}).catch(function(){});
    }
  }catch(e){document.getElementById('input-border').classList.remove('busy');err('Agent: '+e)}
}

function updateMemCount(){var el=document.getElementById('sb-mem');if(el)el.textContent=agentMemory.length+' memor'+(agentMemory.length===1?'y':'ies')}
function saveMemory(){
  try{sessionStorage.setItem('dot_memory',JSON.stringify(agentMemory))}catch(e){}
  updateMemCount();
  fetch('/api/memory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({memory:agentMemory})}).catch(function(){});
}
function openMemory(){document.getElementById('mem-panel').classList.add('vis');renderMemList()}
function closeMemory(){document.getElementById('mem-panel').classList.remove('vis')}
function renderMemList(){
  var list=document.getElementById('mem-list');list.innerHTML='';
  if(!agentMemory.length){list.innerHTML='<div class="mem-empty">No memories yet.<br>Agent adds facts automatically, or add below.</div>';return}
  agentMemory.forEach(function(m,i){
    var d=document.createElement('div');d.className='mem-item';
    d.innerHTML='<span class="mem-idx">'+(i+1)+'</span><span class="mem-text">'+m.replace(/</g,'&lt;')+'</span><span class="mem-del" onclick="deleteMemory('+i+')">✕</span>';
    list.appendChild(d);
  });
}
function deleteMemory(i){agentMemory.splice(i,1);saveMemory();renderMemList()}
function addMemoryItem(){var inp=document.getElementById('mem-add-inp');var v=inp.value.trim();if(!v)return;agentMemory.push(v);inp.value='';saveMemory();renderMemList()}
document.getElementById('mem-add-inp').addEventListener('keydown',function(e){if(e.key==='Enter')addMemoryItem()});
document.getElementById('mem-panel').addEventListener('click',function(e){if(e.target===this)closeMemory()});

function updateStatus(d){
  if(d.total_generations!==undefined){
    var g=document.getElementById('sb-gen');
    g.textContent='gen '+d.total_generations+' · '+d.best_fitness;
    g.className='sb-pill on';
    var l=document.getElementById('sb-live');
    l.textContent='NEAT ✓';l.className='sb-pill live';
  }
}

(async function boot(){
  sys('DOT v4.0 · Fabrica OS · type /help');
  try{var rm=await fetch('/api/memory');var dm=await rm.json();
    if(Array.isArray(dm.memory)){
      agentMemory=dm.memory;
      try{sessionStorage.setItem('dot_memory',JSON.stringify(agentMemory))}catch(e){}
      updateMemCount();
    }
  }catch(e){}
  try{var r=await fetch('/api/system');var d=await r.json();
    sysPrompt=d.prompt||'';workerUrl=d.chat_worker_url||'';model=d.model||'—';
    document.getElementById('ml-model').textContent=model;
    document.getElementById('sb-model').textContent=model;
    var l=document.getElementById('sb-live');
    l.textContent=workerUrl?'worker ✓':'no worker';
    l.className=workerUrl?'sb-pill live':'sb-pill idle';
    sys('Ready · '+model);
  }catch(e){err('boot: '+e)}
  try{var r2=await fetch('/api/status');var d2=await r2.json();if(d2.total_generations)updateStatus(d2);}catch(e){}
})();

var $inp=document.getElementById('inp');
$inp.addEventListener('input',function(){
  this.style.height='auto';this.style.height=Math.min(this.scrollHeight,120)+'px';
  if(this.value.startsWith('/'))showCmdPal(this.value);else hideCmdPal();
});
$inp.addEventListener('keydown',function(e){
  var pal=document.getElementById('cmdpal');
  if(pal.classList.contains('vis')){
    var items=pal.querySelectorAll('.cpi');
    if(e.key==='ArrowDown'){e.preventDefault();cmdSelIdx=Math.min(cmdSelIdx+1,items.length-1);items.forEach(function(el,n){el.classList.toggle('sel',n===cmdSelIdx)});return}
    if(e.key==='ArrowUp'){e.preventDefault();cmdSelIdx=Math.max(cmdSelIdx-1,0);items.forEach(function(el,n){el.classList.toggle('sel',n===cmdSelIdx)});return}
    if(e.key==='Tab'||e.key==='Enter'){e.preventDefault();if(cmdSelIdx>=0&&items[cmdSelIdx])items[cmdSelIdx].click();else if(e.key==='Enter'){hideCmdPal();var v=$inp.value.trim();if(v.startsWith('/')){runCmd(v.split(' ')[0],v.split(' ').slice(1).join(' '));$inp.value='';$inp.style.height='auto';}}return}
    if(e.key==='Escape'){hideCmdPal();return}
  }
  if(e.key==='Tab'&&!$inp.value.startsWith('/')){e.preventDefault();toggleMode();return}
  if(e.key==='p'&&(e.ctrlKey||e.metaKey)){e.preventDefault();openMemory();return}
  if(e.key==='ArrowUp'&&$inp.value===''&&inputHist.length){e.preventDefault();histIdx=Math.min(histIdx+1,inputHist.length-1);$inp.value=inputHist[histIdx];$inp.style.height='auto';$inp.style.height=Math.min($inp.scrollHeight,120)+'px';return}
  if(e.key==='ArrowDown'&&histIdx>=0){e.preventDefault();histIdx=Math.max(histIdx-1,-1);$inp.value=histIdx<0?'':inputHist[histIdx];$inp.style.height='auto';$inp.style.height=Math.min($inp.scrollHeight,120)+'px';return}
  if(e.key==='Enter'&&!e.shiftKey){
    e.preventDefault();var text=$inp.value.trim();if(!text||busy)return;
    $inp.value='';$inp.style.height='auto';hideCmdPal();histIdx=-1;inputHist.unshift(text);
    if(text.startsWith('/')){runCmd(text.split(' ')[0],text.split(' ').slice(1).join(' '));return}
    if(text.startsWith('@')){openMemory();return}
    if(text.indexOf('$')!==-1){
      var goals=text.split(/\$+/).map(function(s){return s.trim()}).filter(Boolean);
      if(goals.length>1){busy=true;runGoalQueue(goals).finally(function(){busy=false});return}
      var single=goals[0]||text.replace(/^\$+/,'').trim();
      if(single){busy=true;addMsg('user','YOU',single);sendAgent(single).finally(function(){busy=false});}
      return;
    }
    busy=true;
    if(mode==='agent'){addMsg('user','YOU',text);sendAgent(text).finally(function(){busy=false});}
    else{addMsg('user','YOU',text);sendChat(text).finally(function(){busy=false});}
  }
});

async function runGoalQueue(goals){
  sys('偵測到 '+goals.length+' 個目標，將依序執行（每個間隔 3 秒避免額度超限）');
  for(var i=0;i<goals.length;i++){
    addMsg('user','YOU','['+(i+1)+'/'+goals.length+'] '+goals[i]);
    await sendAgent(goals[i]);
    if(i<goals.length-1){sys('等待 3 秒…');await new Promise(function(r){setTimeout(r,3000)});}
  }
  sys('全部 '+goals.length+' 個目標已完成');
}
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


def get_worker_info() -> dict:
    """跟 Worker 要真正在用的 model 名稱,而不是在 Python 這邊用獨立、可能對不上的環境變數猜測。
    Worker 不可達時優雅降級成靜態預設值,不會讓整個 /api/system 端點掛掉。"""
    if not CHAT_WORKER_URL:
        return {}
    try:
        req = urllib.request.Request(CHAT_WORKER_URL.rstrip("/"), method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
    except Exception:
        return {}


@app.get("/api/system")
async def get_system_endpoint():
    worker_info = get_worker_info()
    model = worker_info.get("chat_model") or GROQ_MODEL
    if worker_info.get("groq_key"):
        provider = "Groq"
    elif worker_info.get("workers_ai"):
        provider = "Cloudflare Workers AI"
    else:
        provider = "unknown"
    return JSONResponse({
        "prompt": get_system_prompt(),
        "model": model,
        "provider": provider,
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
    return JSONResponse({"history": get_chat_history(), "model": GROQ_MODEL})


@app.get("/api/memory")
async def get_memory_endpoint():
    return JSONResponse({"memory": get_agent_memory()})


@app.post("/api/memory")
async def set_memory_endpoint(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON"}, status_code=400)
    memory = body.get("memory")
    if not isinstance(memory, list):
        return JSONResponse({"error": "memory must be a list"}, status_code=400)
    save_agent_memory(memory)
    return JSONResponse({"ok": True})


# ============================================================
# 自我改進:DOT 分析演化結果 → 寫程式 → commit GitHub
# ============================================================

def _gh_headers():
    return {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "DOT-Self-Improve",
        "Content-Type": "application/json",
    }


def github_get_file(path: str) -> dict:
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
        headers=_gh_headers(), method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        content = base64.b64decode(data["content"]).decode("utf-8")
        return {"content": content, "sha": data["sha"]}
    except urllib.error.HTTPError as e:
        return {"error": f"GitHub GET HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def github_commit_file(path: str, content: str, sha: str, message: str) -> dict:
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "sha": sha,
        "branch": "main",
    }
    req = urllib.request.Request(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
        data=json.dumps(payload).encode(),
        headers=_gh_headers(), method="PUT"
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read())
        return {"sha": data["commit"]["sha"], "url": data["content"]["html_url"]}
    except urllib.error.HTTPError as e:
        return {"error": f"GitHub PUT HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


def extract_tasks_code(content: str) -> str:
    lines = content.split("\n")
    start = next((i for i, l in enumerate(lines) if l.startswith("TASKS = {")), None)
    if start is None:
        return ""
    depth = 0
    for i, line in enumerate(lines[start:], start):
        depth += line.count("{") - line.count("}")
        if i > start and depth <= 0:
            return "\n".join(lines[start:i + 1])
    return ""


def call_worker_meta(state: dict, tasks_code: str, history_str: str, hint: str = "") -> dict:
    if not CHAT_WORKER_URL:
        return {"error": "CHAT_WORKER_URL not set"}
    payload = {"state": state, "tasks_code": tasks_code, "history": history_str, "hint": hint}
    req = urllib.request.Request(
        CHAT_WORKER_URL.rstrip("/") + "/meta",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=40) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"error": f"Worker HTTP {e.code}: {e.read().decode()[:200]}"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/self-improve")
async def self_improve_endpoint(request: Request):
    # 可選認證
    if IMPROVE_SECRET:
        if request.headers.get("x-improve-secret", "") != IMPROVE_SECRET:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    if not GITHUB_TOKEN or not GITHUB_REPO:
        return JSONResponse({"error": "GITHUB_TOKEN 或 GITHUB_REPO 未設定"})

    try:
        body = await request.json()
    except Exception:
        body = {}
    hint = str(body.get("hint") or "")[:400]  # agent 呼叫 write_code 時的診斷,之前完全被丟掉沒用到

    # 1. 讀演化狀態
    state = load_state()
    if not state:
        return JSONResponse({"error": "還沒有演化資料，先跑幾輪演化再試"})

    context = {
        "generation": state["generation"],
        "current_task": TASK_ORDER[state["task_idx"] % len(TASK_ORDER)],
        "history_tail": state.get("history", [])[-10:],
    }

    # 2. 從 GitHub 取現有程式碼
    file_data = github_get_file("api/index.py")
    if "error" in file_data:
        return JSONResponse({"error": f"無法讀取 GitHub 檔案: {file_data['error']}"})

    file_content = file_data["content"]
    file_sha = file_data["sha"]
    tasks_code = extract_tasks_code(file_content)
    if not tasks_code:
        return JSONResponse({"error": "找不到 TASKS 程式碼區塊"})

    # 3. 呼叫 Cloudflare Worker 讓 DOT 分析並提案
    meta_result = call_worker_meta(context, tasks_code, json.dumps(context["history_tail"], indent=2), hint)
    if "error" in meta_result:
        return JSONResponse({"error": f"DOT meta 分析失敗: {meta_result['error']}"})

    change = meta_result.get("change")
    if not change:
        return JSONResponse({"error": "DOT 沒有回傳有效的改進提案", "raw": meta_result})

    old_code = change.get("old_code", "")
    new_code = change.get("new_code", "")

    if not old_code or old_code not in file_content:
        return JSONResponse({
            "error": "DOT 提案的 old_code 在檔案裡找不到（可能是 LLM 幻覺）",
            "analysis": change.get("analysis"),
            "old_code_preview": old_code[:120],
        })

    # 4. 套用修改並驗證語法
    new_content = file_content.replace(old_code, new_code, 1)
    try:
        _ast.parse(new_content)
    except SyntaxError as e:
        return JSONResponse({
            "error": f"語法驗證失敗，已放棄 commit: {e}",
            "analysis": change.get("analysis"),
        })

    # 5. Commit 到 GitHub
    commit_msg = f"[DOT self-improve] gen={state['generation']}: {change.get('expected_effect', 'task update')[:80]}"
    commit_result = github_commit_file("api/index.py", new_content, file_sha, commit_msg)
    if "error" in commit_result:
        return JSONResponse({"error": f"Commit 失敗: {commit_result['error']}"})

    # 6. 記錄到 Redis
    record = {
        "generation": state["generation"],
        "analysis": change.get("analysis", ""),
        "change_type": change.get("change_type", ""),
        "expected_effect": change.get("expected_effect", ""),
        "commit_url": commit_result.get("url", ""),
    }
    try:
        old = redis_command("GET", "dot_improve_history")
        hist = json.loads(old["result"]) if old and old.get("result") else []
        hist.append(record)
        redis_command("SET", "dot_improve_history", json.dumps(hist[-20:]))
    except Exception:
        pass

    return JSONResponse({
        "ok": True,
        "analysis": change.get("analysis"),
        "expected_effect": change.get("expected_effect"),
        "commit_url": commit_result.get("url"),
    })


@app.get("/api/improve-history")
async def improve_history_endpoint():
    try:
        result = redis_command("GET", "dot_improve_history")
        hist = json.loads(result["result"]) if result and result.get("result") else []
    except Exception:
        hist = []
    return JSONResponse({"history": hist})
