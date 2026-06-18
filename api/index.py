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
  --bg:#0A0E14;--bg2:#0E131C;
  --card:#141A24;--card2:#1A2230;
  --border:#222C3A;--border-soft:#1A222E;
  --ink:#E8EDF2;--ink-soft:#8B97A8;--ink-faint:#535E70;
  --rust:#D97757;--rust-bright:#F08A66;
  --slate:#8B97A8;--sage:#9DAE8B;--gold:#D9A441;--rose:#C77B6B;
  --mono:'Fragment Mono',monospace;--pixel:'Press Start 2P',monospace;--sans:'Inter',sans-serif;
}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font-family:var(--sans);font-size:14px;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;position:relative;-webkit-font-smoothing:antialiased}
#stars{position:fixed;inset:0;z-index:0;pointer-events:none}

/* top tabs */
header{display:flex;align-items:center;gap:0;padding:0;flex-shrink:0;
  background:var(--bg2);border-bottom:1px solid var(--border-soft);z-index:2;position:relative;height:38px}
.tab{display:flex;align-items:center;gap:7px;padding:0 16px;height:100%;font-size:12px;color:var(--ink-faint);
  border-right:1px solid var(--border-soft);font-family:var(--mono)}
.tab.active{color:var(--ink);background:var(--card)}
.tab .x{color:var(--ink-faint);font-size:11px}
.tdot{width:6px;height:6px;border-radius:50%;background:var(--ink-faint)}
.tdot.live{background:var(--sage)}
.tdot.err{background:var(--rose)}
.spacer{flex:1}
.statetxt{font-family:var(--mono);font-size:11px;color:var(--ink-soft);padding-right:16px}

/* hero */
.hero{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  z-index:1;position:relative;padding:20px;overflow-y:auto}
.brandwrap{text-align:center;margin-bottom:8px}
.brand-tag{font-family:var(--mono);font-size:11px;color:var(--ink-faint);letter-spacing:0.1em;margin-bottom:14px}
.brand{font-family:var(--pixel);font-size:38px;line-height:1.1;letter-spacing:0;display:flex;gap:14px;align-items:baseline}
.brand .dot{color:var(--rust)}
.brand .code{color:var(--slate)}
@media(max-width:560px){.brand{font-size:24px;gap:8px}}

/* live stat ribbon under logo */
.ribbon{display:flex;gap:0;margin:22px 0 6px;border:1px solid var(--border);border-radius:10px;overflow:hidden;background:var(--card)}
.rcell{padding:10px 16px;border-right:1px solid var(--border-soft);text-align:center;min-width:78px}
.rcell:last-child{border-right:none}
.rk{font-family:var(--mono);font-size:9px;color:var(--ink-faint);letter-spacing:0.08em;text-transform:uppercase;margin-bottom:4px}
.rv{font-family:var(--mono);font-size:16px;color:var(--ink)}
.rv.sage{color:var(--sage)}.rv.rust{color:var(--rust)}.rv.gold{color:var(--gold)}.rv.faint{color:var(--ink-faint)}

/* input card */
.card{width:min(620px,92vw);margin-top:18px;background:var(--card);border:1px solid var(--border);
  border-radius:12px;overflow:hidden;box-shadow:0 8px 40px rgba(0,0,0,0.4)}
.card-top{display:flex;align-items:center;gap:10px;padding:14px 16px}
.badge{font-family:var(--mono);font-size:11px;color:var(--bg);background:var(--rust);
  padding:2px 8px;border-radius:5px;font-weight:600;flex-shrink:0}
#cmd{flex:1;background:none;border:none;outline:none;color:var(--ink);
  font-family:var(--mono);font-size:14px;caret-color:var(--rust)}
#cmd::placeholder{color:var(--ink-faint)}
#cmd:disabled{opacity:.5}
.spin{display:none;color:var(--gold);font-family:var(--mono);animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.card-sub{display:flex;align-items:center;gap:8px;padding:10px 16px;border-top:1px solid var(--border-soft);
  font-family:var(--mono);font-size:11px;color:var(--ink-faint)}
.card-sub .accent{color:var(--rust)}

.hints{display:flex;gap:22px;margin-top:18px;font-family:var(--mono);font-size:11px;color:var(--ink-faint);flex-wrap:wrap;justify-content:center}
.hints b{color:var(--ink-soft);font-weight:400}
.tip{margin-top:20px;font-family:var(--mono);font-size:11px;color:var(--ink-faint);display:flex;align-items:center;gap:7px}
.tip .led{width:6px;height:6px;border-radius:50%;background:var(--rust)}

/* log drawer */
.log{width:min(620px,92vw);margin-top:18px;max-height:30vh;overflow-y:auto;font-family:var(--mono);
  font-size:12px;line-height:1.7;border-top:1px solid var(--border-soft);padding-top:10px}
.log:empty{display:none}
.log::-webkit-scrollbar{width:4px}.log::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.row{display:flex;gap:10px;align-items:baseline;padding:1px 0}
.row .t{color:var(--ink-faint);font-size:10px;width:54px;flex-shrink:0}
.cmd{color:var(--rust)}.out{color:var(--ink)}.sys{color:var(--ink-soft);font-style:italic}
.err{color:var(--rose)}.sw{color:var(--gold);font-weight:600}.grow{color:var(--sage);font-weight:600}.accent{color:var(--rust)}
footer{padding:8px 16px;font-family:var(--mono);font-size:11px;color:var(--ink-faint);
  display:flex;justify-content:space-between;border-top:1px solid var(--border-soft);background:var(--bg2);z-index:2}
</style>
</head>
<body>
<canvas id="stars"></canvas>

<header>
  <div class="tab"><span class="tdot"></span>DOT</div>
  <div class="tab active"><span class="x">▸</span>evolving neuron</div>
  <div class="spacer"></div>
  <div class="statetxt" id="state">connecting…</div>
</header>

<div class="hero">
  <div class="brandwrap">
    <div class="brand-tag">NeuroEvolution of Augmenting Topologies</div>
    <div class="brand"><span class="dot">DOT</span><span class="code">NEAT</span></div>
  </div>

  <div class="ribbon">
    <div class="rcell"><div class="rk">gen</div><div class="rv faint" id="r-gen">—</div></div>
    <div class="rcell"><div class="rk">neurons</div><div class="rv faint" id="r-neu">—</div></div>
    <div class="rcell"><div class="rk">fitness</div><div class="rv faint" id="r-fit">—</div></div>
    <div class="rcell"><div class="rk">task</div><div class="rv faint" id="r-task">—</div></div>
    <div class="rcell"><div class="rk">mse</div><div class="rv faint" id="r-mse">—</div></div>
  </div>

  <div class="card">
    <div class="card-top">
      <span class="badge">RUN</span>
      <input id="cmd" placeholder="輸入指令… run [N] · status · help · clear" autocomplete="off" spellcheck="false">
      <span class="spin" id="spin">⟳</span>
    </div>
    <div class="card-sub"><span class="accent">›</span> DOT NEAT · 神經元會隨演化長出與修剪 · 狀態存於 Redis</div>
  </div>

  <div class="hints">
    <div><b>tab</b> 歷史指令</div>
    <div><b>run N</b> 演化 N 代</div>
    <div><b>status</b> 讀取狀態</div>
    <div><b>/ </b> 喚起命令</div>
  </div>

  <div class="tip"><span class="led"></span>提示：神經元數量上升代表網路正在「長大」，fitness 上升代表它學得更好</div>

  <div class="log" id="log"></div>
</div>

<footer>
  <span>~/dot-neat:main</span>
  <span id="foot-cl">continual evolution</span>
  <span>0.2.0</span>
</footer>

<script>
// starfield
const cv=document.getElementById('stars'),cx=cv.getContext('2d');
let stars=[];
function resize(){cv.width=innerWidth;cv.height=innerHeight;
  stars=Array.from({length:Math.floor(innerWidth*innerHeight/9000)},()=>({
    x:Math.random()*cv.width,y:Math.random()*cv.height,r:Math.random()*1.2+0.2,
    a:Math.random()*0.5+0.15,tw:Math.random()*0.02+0.005}));}
resize();addEventListener('resize',resize);
function draw(){cx.clearRect(0,0,cv.width,cv.height);
  for(const s of stars){s.a+=s.tw;if(s.a>0.7||s.a<0.1)s.tw*=-1;
    cx.fillStyle=`rgba(217,119,87,${s.a*0.5})`;cx.beginPath();cx.arc(s.x,s.y,s.r,0,7);cx.fill();}
  requestAnimationFrame(draw);}
draw();

const $log=document.getElementById('log'),$cmd=document.getElementById('cmd'),$spin=document.getElementById('spin'),
      $state=document.getElementById('state');
let hist=[],hIdx=-1,busy=false,tick=null,cd=30,lastNeu=null;
const ts=()=>new Date().toTimeString().slice(0,8);
function row(h,c){const d=document.createElement('div');d.className='row';
  d.innerHTML=`<span class="t">${ts()}</span><span class="${c}">${h}</span>`;$log.appendChild(d);$log.scrollTop=$log.scrollHeight;}
function plain(h){const d=document.createElement('div');d.className='row';
  d.innerHTML=`<span class="t"></span><span>${h}</span>`;$log.appendChild(d);$log.scrollTop=$log.scrollHeight;}

function apply(d){
  const fit=d.best_fitness??0,gen=d.total_generations??0,neu=d.n_neurons??null;
  const g=document.getElementById('r-gen');g.textContent=gen;g.className='rv sage';
  const nu=document.getElementById('r-neu');nu.textContent=neu??'—';nu.className='rv rust';
  const f=document.getElementById('r-fit');f.textContent=fit.toFixed(3);
  f.className='rv '+(fit>0.95?'sage':fit>0.7?'rust':'gold');
  const t=document.getElementById('r-task');t.textContent=d.current_task?'task_'+d.current_task:'—';
  t.className='rv '+(d.current_task?'rust':'faint');
  document.getElementById('r-mse').textContent=(d.mse_current_task??d.mse_task_a)?.toFixed(3)??'—';
  $state.textContent='live · '+(d.redis_connected?'redis ✓':'no redis');
  document.querySelector('.tdot').classList.add('live');
  return {neu,gen};
}

function setBusy(v){busy=v;$spin.style.display=v?'inline':'none';$cmd.disabled=v;}
function resetCd(){clearInterval(tick);cd=30;tick=setInterval(()=>{cd--;
  if(cd<=0){doStatus(true);resetCd();}},1000);}

async function doStatus(silent){
  try{const r=await fetch('/api/status');if(!r.ok)throw r.status;const d=await r.json();
    if(d.status==='no_data'){$state.textContent='live · no data yet';return;}
    apply(d);if(!silent)row(JSON.stringify(d),'out');
  }catch(e){$state.textContent='error';document.querySelector('.tdot').classList.add('err');if(!silent)row('status failed: '+e,'err');}
}
async function doRun(n){
  if(busy){row('busy — wait','sys');return;}
  setBusy(true);row('run '+n,'cmd');row('evolving '+n+' generations…','sys');
  try{const r=await fetch('/api/evolve?generations='+n);if(!r.ok)throw r.status;const d=await r.json();
    const {neu}=apply(d);
    let growth='';
    if(lastNeu!==null&&neu!==null){
      if(neu>lastNeu)growth=` <span class="grow">↑ grew ${neu-lastNeu} neuron(s) → ${neu}</span>`;
      else if(neu<lastNeu)growth=` <span class="sys">↓ pruned ${lastNeu-neu} → ${neu}</span>`;
    }
    lastNeu=neu;
    const sw=d.switched_task_this_call?` <span class="sw">★ task → ${d.current_task}</span>`:'';
    row(`done · gen ${d.total_generations} · neurons ${d.n_neurons} (avg ${d.avg_neurons}, max ${d.max_neurons}) · fit ${d.best_fitness} · mse ${d.mse_current_task}${growth}${sw}`,'out');
  }catch(e){row('error: '+e,'err');}
  setBusy(false);resetCd();
}
function cmd(raw){const p=raw.trim().replace(/^\//,'').split(/\s+/),c=p[0].toLowerCase();
  if(c==='run')doRun(Math.max(1,Math.min(parseInt(p[1])||40,300)));
  else if(c==='status'){row('status','cmd');doStatus(false);}
  else if(c==='clear')$log.innerHTML='';
  else if(c==='help'){plain('<span class="accent">run [N]</span> 演化 N 代 (預設40 上限300)');
    plain('<span class="accent">status</span> 讀取狀態不演化');plain('<span class="accent">clear</span> 清除輸出');
    plain('<span class="sys">↑↓ 命令歷史</span>');}
  else row('unknown: '+c,'err');}
$cmd.addEventListener('keydown',e=>{
  if(e.key==='Enter'){const v=$cmd.value.trim();if(!v)return;hist.unshift(v);hIdx=-1;$cmd.value='';cmd(v);}
  else if(e.key==='ArrowUp'){e.preventDefault();hIdx=Math.min(hIdx+1,hist.length-1);$cmd.value=hist[hIdx]??'';}
  else if(e.key==='ArrowDown'){e.preventDefault();hIdx=Math.max(hIdx-1,-1);$cmd.value=hIdx<0?'':hist[hIdx];}});
doStatus(true);resetCd();
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
