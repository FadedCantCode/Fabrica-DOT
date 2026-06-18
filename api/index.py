"""
/api/evolve —— FastAPI 版本(Vercel 新版 Python runtime 要求)

Vercel 現在需要 FastAPI/Flask 等 ASGI/WSGI framework 才能偵測到 Python function。
核心演化邏輯跟之前完全一樣,只有 HTTP handler 從 BaseHTTPRequestHandler 換成 FastAPI。
"""

from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
import json
import math
import os
import random
import urllib.request


# ============================================================
# 核心神經元邏輯(不變)
# ============================================================

def basis(x: float) -> list:
    return [x, x * x, math.sin(x), math.cos(x), math.sin(2 * x), math.cos(2 * x)]

def full_features(x: float) -> list:
    return basis(x) + [1.0]

N_GENES = len(full_features(0.0))

def forward(genome: list, x: float) -> float:
    return sum(g * f for g, f in zip(genome, full_features(x)))

def random_genome(rng: random.Random) -> list:
    return [rng.uniform(-1.0, 1.0) for _ in range(N_GENES)]

def mse_loss(genome: list, data: list) -> float:
    n = len(data)
    return sum((forward(genome, x) - y) ** 2 for x, y in data) / n

def total_loss(genome, data, ewc_importance=None, ewc_anchor=None, ewc_lambda=0.0):
    loss = mse_loss(genome, data)
    if ewc_importance is not None:
        loss += ewc_lambda * sum(
            imp * (g - a) ** 2 for imp, g, a in zip(ewc_importance, genome, ewc_anchor)
        )
    return loss

def total_loss_and_grad(genome, data, ewc_importance=None, ewc_anchor=None, ewc_lambda=0.0):
    n = len(data)
    grad = [0.0] * N_GENES
    se = 0.0
    for x, y in data:
        feats = full_features(x)
        pred = sum(g * f for g, f in zip(genome, feats))
        error = pred - y
        se += error ** 2
        for i, f in enumerate(feats):
            grad[i] += (2 * error * f) / n
    loss = se / n
    if ewc_importance is not None:
        for i in range(N_GENES):
            diff = genome[i] - ewc_anchor[i]
            loss += ewc_lambda * ewc_importance[i] * diff ** 2
            grad[i] += 2 * ewc_lambda * ewc_importance[i] * diff
    return loss, grad


class ReplayBuffer:
    def __init__(self, capacity=200, seed=None):
        self.capacity = capacity
        self.buffer = []
        self.rng = random.Random(seed)

    def add(self, x, y):
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            self.buffer[self.rng.randrange(self.capacity)] = (x, y)

    def sample(self, k):
        k = min(k, len(self.buffer))
        return self.rng.sample(self.buffer, k) if k > 0 else []


def compute_fisher_importance(genome, data):
    n = len(data)
    importance = [0.0] * N_GENES
    for x, y in data:
        feats = full_features(x)
        pred = sum(g * f for g, f in zip(genome, feats))
        error = pred - y
        for i, f in enumerate(feats):
            importance[i] += (2 * error * f) ** 2
    return [v / n for v in importance]


class AdamOptimizer:
    def __init__(self, n_params, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = [0.0] * n_params
        self.v = [0.0] * n_params
        self.t = 0

    def step(self, params, grads, clip_norm=5.0):
        norm = math.sqrt(sum(g * g for g in grads))
        if norm > clip_norm:
            grads = [g * clip_norm / norm for g in grads]
        self.t += 1
        new_params = []
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            new_params.append(p - self.lr * m_hat / (math.sqrt(v_hat) + self.eps))
        return new_params


class Individual:
    __slots__ = ("genome", "sigma", "fitness")
    def __init__(self, genome, sigma):
        self.genome = genome
        self.sigma = sigma
        self.fitness = None


def euclid_dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class EvolvingPopulation:
    def __init__(self, pop_size=24, sigma_init=0.5, sigma_share=1.0,
                 stagnation_patience=15, n_immigrants=4, seed=None):
        self.rng = random.Random(seed)
        self.pop_size = pop_size
        self.sigma_share = sigma_share
        self.stagnation_patience = stagnation_patience
        self.n_immigrants = n_immigrants
        self.individuals = [Individual(random_genome(self.rng), sigma_init) for _ in range(pop_size)]
        self.tau = 1.0 / math.sqrt(N_GENES)
        self.gens_since_improvement = 0
        self._last_best_fitness = None
        self.best_individual = None

    def _shared_fitness(self):
        genomes = [ind.genome for ind in self.individuals]
        shared = []
        for i, ind in enumerate(self.individuals):
            niche = sum(
                1.0 - euclid_dist(genomes[i], other) / self.sigma_share
                for other in genomes
                if euclid_dist(genomes[i], other) < self.sigma_share
            )
            shared.append(ind.fitness / niche if niche > 0 else ind.fitness)
        return shared

    def _tournament(self, shared_fit, k=3):
        idxs = self.rng.sample(range(len(self.individuals)), min(k, len(self.individuals)))
        return self.individuals[max(idxs, key=lambda i: shared_fit[i])]

    def step(self, loss_fn, elitism=2):
        for ind in self.individuals:
            ind.fitness = 1.0 / (1.0 + loss_fn(ind.genome))
        order = sorted(range(len(self.individuals)), key=lambda i: self.individuals[i].fitness, reverse=True)
        self.best_individual = self.individuals[order[0]]
        best_fitness = self.best_individual.fitness

        if self._last_best_fitness is not None and best_fitness <= self._last_best_fitness + 1e-6:
            self.gens_since_improvement += 1
        else:
            self.gens_since_improvement = 0
        self._last_best_fitness = max(best_fitness, self._last_best_fitness or 0.0)

        stagnation = self.gens_since_improvement >= self.stagnation_patience
        if stagnation:
            self.gens_since_improvement = 0
            for ind in self.individuals:
                ind.sigma = min(ind.sigma * 2.0, 2.0)

        shared_fit = self._shared_fitness()
        next_gen = [self.individuals[i] for i in order[:elitism]]
        n_new = self.n_immigrants if stagnation else 0
        while len(next_gen) < self.pop_size - n_new:
            p1, p2 = self._tournament(shared_fit), self._tournament(shared_fit)
            child = [self.rng.choice([g1, g2]) for g1, g2 in zip(p1.genome, p2.genome)]
            sigma = max(0.01, min(math.sqrt(p1.sigma * p2.sigma) * math.exp(self.tau * self.rng.gauss(0, 1)), 2.0))
            child = [g + sigma * self.rng.gauss(0, 1) for g in child]
            next_gen.append(Individual(child, sigma))
        while len(next_gen) < self.pop_size:
            next_gen.append(Individual(random_genome(self.rng), 0.5))
        self.individuals = next_gen


def evolve_on_task(population, task_data, generations, replay_buffer=None,
                   ewc_importance=None, ewc_anchor=None, ewc_lambda=20.0,
                   use_replay=True, use_ewc=True, memetic_every=5, memetic_steps=30):
    imp = ewc_importance if use_ewc else None
    anc = ewc_anchor if use_ewc else None
    lam = ewc_lambda if use_ewc else 0.0
    for gen in range(generations):
        replay_sample = replay_buffer.sample(len(task_data)) if (use_replay and replay_buffer) else []
        data = task_data + replay_sample
        population.step(lambda genome, d=data: total_loss(genome, d, imp, anc, lam))
        if memetic_every and gen % memetic_every == 0:
            elite = population.best_individual
            g = list(elite.genome)
            pre = total_loss(g, data, imp, anc, lam)
            adam = AdamOptimizer(N_GENES)
            for _ in range(memetic_steps):
                _, grad = total_loss_and_grad(g, data, imp, anc, lam)
                g = adam.step(g, grad)
            if total_loss(g, data, imp, anc, lam) < pre:
                elite.genome = g
                elite.fitness = 1.0 / (1.0 + total_loss(g, data, imp, anc, lam))


# ============================================================
# 任務定義
# ============================================================

_XS = [-math.pi + i * (2 * math.pi / 39) for i in range(40)]
DATA_A = [(x, math.sin(x) * math.cos(x / 2)) for x in _XS]
DATA_B = [(x, math.cos(2 * x) + 0.2 * x) for x in _XS]


# ============================================================
# Redis 持久化
# ============================================================

REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
STATE_KEY = "evolving_neuron_state"


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


def population_to_dict(pop):
    return {
        "pop_size": pop.pop_size, "sigma_share": pop.sigma_share,
        "stagnation_patience": pop.stagnation_patience, "n_immigrants": pop.n_immigrants,
        "gens_since_improvement": pop.gens_since_improvement,
        "last_best_fitness": pop._last_best_fitness,
        "individuals": [{"genome": ind.genome, "sigma": ind.sigma, "fitness": ind.fitness}
                        for ind in pop.individuals],
    }


def population_from_dict(d):
    pop = EvolvingPopulation(pop_size=d["pop_size"], sigma_share=d["sigma_share"],
                             stagnation_patience=d["stagnation_patience"], n_immigrants=d["n_immigrants"])
    pop.individuals = [Individual(r["genome"], r["sigma"]) for r in d["individuals"]]
    for ind, r in zip(pop.individuals, d["individuals"]):
        ind.fitness = r["fitness"]
    pop.gens_since_improvement = d["gens_since_improvement"]
    pop._last_best_fitness = d["last_best_fitness"]
    pop.best_individual = max(pop.individuals, key=lambda i: i.fitness or -1)
    return pop


GENERATIONS_PER_RUN_DEFAULT = 50
GENERATIONS_PER_RUN_MAX = 500
SWITCH_EVERY = 600


def run_batch(generations: int) -> dict:
    state = load_state()
    redis_connected = bool(REDIS_URL and REDIS_TOKEN)

    if state is None:
        pop = EvolvingPopulation(pop_size=24)
        state = {"generation": 0, "current_task": "a",
                 "ewc_importance": None, "ewc_anchor": None, "replay": []}
    else:
        pop = population_from_dict(state["population"])

    replay = ReplayBuffer(capacity=200)
    replay.buffer = [tuple(p) for p in state["replay"]]

    task_data = DATA_A if state["current_task"] == "a" else DATA_B
    have_protection = state["ewc_importance"] is not None

    evolve_on_task(pop, task_data, generations=generations,
                   replay_buffer=replay if have_protection else None,
                   ewc_importance=state["ewc_importance"], ewc_anchor=state["ewc_anchor"],
                   use_replay=have_protection, use_ewc=have_protection)

    prev_total = state["generation"]
    state["generation"] = prev_total + generations
    switched = False

    if prev_total // SWITCH_EVERY != state["generation"] // SWITCH_EVERY:
        finished = list(pop.best_individual.genome)
        state["ewc_importance"] = compute_fisher_importance(finished, task_data)
        state["ewc_anchor"] = finished
        for x, y in task_data:
            replay.add(x, y)
        state["current_task"] = "b" if state["current_task"] == "a" else "a"
        switched = True

    state["population"] = population_to_dict(pop)
    state["replay"] = [list(p) for p in replay.buffer]

    if redis_connected:
        save_state(state)

    return {
        "redis_connected": redis_connected,
        "total_generations": state["generation"],
        "ran_this_call": generations,
        "current_task": state["current_task"],
        "switched_task_this_call": switched,
        "best_fitness": round(pop.best_individual.fitness, 4),
        "mse_task_a": round(mse_loss(pop.best_individual.genome, DATA_A), 4),
        "mse_task_b": round(mse_loss(pop.best_individual.genome, DATA_B), 4),
        "continual_learning_active": have_protection,
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
<link href="https://fonts.googleapis.com/css2?family=Fragment+Mono:ital@0;1&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#1A1815;          /* 暖調深灰,帶一點褐 */
  --bg-raised:#232019;
  --bg-input:#2A261F;
  --border:#332E26;
  --border-soft:#2A2620;
  --ink:#EDE7DC;         /* 米白主文字 */
  --ink-soft:#A89E8E;    /* 次級文字 */
  --ink-faint:#6B6356;   /* 最淡 */
  --rust:#D97757;        /* 暖橘陶土,主強調色 */
  --sage:#9DAE8B;        /* 柔和鼠尾草綠,正向狀態 */
  --gold:#D9A441;        /* 暖金,警示/切換 */
  --rose:#C77B6B;        /* 柔紅,錯誤 */
  --mono:'Fragment Mono',monospace;
  --sans:'Inter',-apple-system,sans-serif;
}
html,body{height:100%}
body{
  background:var(--bg);color:var(--ink);
  font-family:var(--sans);font-size:14px;line-height:1.6;
  -webkit-font-smoothing:antialiased;
  display:flex;flex-direction:column;height:100vh;overflow:hidden;
}

/* HEADER */
header{
  display:flex;align-items:center;gap:12px;
  padding:14px 20px;flex-shrink:0;
  border-bottom:1px solid var(--border-soft);
}
.mark{
  width:22px;height:22px;border-radius:6px;
  background:var(--rust);
  display:flex;align-items:center;justify-content:center;
  font-family:var(--mono);font-size:13px;font-weight:600;color:var(--bg);
  flex-shrink:0;
}
.title{font-weight:600;font-size:15px;letter-spacing:-0.01em}
.title .dim{color:var(--ink-faint);font-weight:400;margin-left:8px;font-size:13px}
.status-pill{
  margin-left:auto;display:flex;align-items:center;gap:7px;
  font-family:var(--mono);font-size:12px;color:var(--ink-soft);
  padding:5px 11px;border:1px solid var(--border);border-radius:999px;
}
.dot{width:7px;height:7px;border-radius:50%;background:var(--ink-faint);transition:background .3s}
.dot.live{background:var(--sage);box-shadow:0 0 0 3px rgba(157,174,139,0.15)}
.dot.err{background:var(--rose)}

/* STATS */
.stats{
  display:grid;grid-template-columns:repeat(5,1fr) 1.4fr;
  border-bottom:1px solid var(--border-soft);flex-shrink:0;
}
.stat{padding:13px 20px;border-right:1px solid var(--border-soft)}
.stat:last-child{border-right:none}
.stat .k{font-family:var(--mono);font-size:10.5px;letter-spacing:0.05em;color:var(--ink-faint);text-transform:uppercase;margin-bottom:5px}
.stat .v{font-family:var(--mono);font-size:19px;font-weight:500;color:var(--ink);line-height:1}
.stat .v.sage{color:var(--sage)}
.stat .v.gold{color:var(--gold)}
.stat .v.rust{color:var(--rust)}
.stat .v.faint{color:var(--ink-faint)}
.cl{display:flex;flex-direction:column;justify-content:center;gap:8px;padding:13px 20px}
.track{height:4px;background:var(--border);border-radius:3px;overflow:hidden}
.fill{height:100%;width:0;background:linear-gradient(90deg,var(--rust),var(--gold));border-radius:3px;transition:width .7s cubic-bezier(.4,0,.2,1)}
.cl-txt{font-family:var(--mono);font-size:11px;color:var(--ink-faint);letter-spacing:0.02em}
.cl-txt.on{color:var(--sage)}

/* TERMINAL */
.term{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:8px 20px 0}
.log{flex:1;overflow-y:auto;padding:8px 0;font-family:var(--mono);font-size:13px;line-height:1.7}
.log::-webkit-scrollbar{width:5px}
.log::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
.log::-webkit-scrollbar-track{background:transparent}
.row{display:flex;gap:12px;align-items:baseline;padding:1px 0}
.row .t{color:var(--ink-faint);font-size:11px;flex-shrink:0;width:62px}
.cmd{color:var(--rust)}
.out{color:var(--ink)}
.sys{color:var(--ink-soft);font-style:italic}
.err{color:var(--rose)}
.head{color:var(--ink);font-weight:600;font-family:var(--sans)}
.sw{color:var(--gold);font-weight:600}
.kbd{color:var(--ink-soft)}
.accent{color:var(--rust)}

/* INPUT */
.inputbar{
  display:flex;align-items:center;gap:11px;
  margin:6px 0 16px;padding:13px 15px;
  background:var(--bg-input);border:1px solid var(--border);border-radius:11px;
  transition:border-color .2s;
}
.inputbar:focus-within{border-color:var(--rust)}
.chev{color:var(--rust);font-family:var(--mono);font-size:14px;user-select:none}
#cmd{
  flex:1;background:none;border:none;outline:none;
  color:var(--ink);font-family:var(--mono);font-size:13.5px;
  caret-color:var(--rust);
}
#cmd::placeholder{color:var(--ink-faint)}
#cmd:disabled{opacity:.5}
.spin{display:none;color:var(--gold);font-family:var(--mono);animation:sp 1s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
@media(max-width:720px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .stat:nth-child(5),.cl{grid-column:span 2}
  .stat{border-right:none;border-bottom:1px solid var(--border-soft)}
}
</style>
</head>
<body>

<header>
  <div class="mark">D</div>
  <div class="title">DOT<span class="dim">evolving neuron</span></div>
  <div class="status-pill"><span class="dot" id="dot"></span><span id="state">connecting</span></div>
</header>

<div class="stats">
  <div class="stat"><div class="k">generation</div><div class="v faint" id="s-gen">—</div></div>
  <div class="stat"><div class="k">best fitness</div><div class="v faint" id="s-fit">—</div></div>
  <div class="stat"><div class="k">task</div><div class="v faint" id="s-task">—</div></div>
  <div class="stat"><div class="k">mse a</div><div class="v faint" id="s-ma">—</div></div>
  <div class="stat"><div class="k">mse b</div><div class="v faint" id="s-mb">—</div></div>
  <div class="cl">
    <div class="track"><div class="fill" id="s-bar"></div></div>
    <div class="cl-txt" id="s-cl">continual learning —</div>
  </div>
</div>

<div class="term">
  <div class="log" id="log"></div>
  <div class="inputbar">
    <span class="chev">›</span>
    <input id="cmd" placeholder="run [N] · status · help · clear" autocomplete="off" spellcheck="false">
    <span class="spin" id="spin">⟳</span>
  </div>
</div>

<script>
const $log=document.getElementById('log'),$cmd=document.getElementById('cmd'),
      $spin=document.getElementById('spin'),$dot=document.getElementById('dot'),
      $state=document.getElementById('state');
let hist=[],hIdx=-1,busy=false,tick=null,cd=30;
const ts=()=>new Date().toTimeString().slice(0,8);

function row(html,cls){
  const d=document.createElement('div');d.className='row';
  d.innerHTML=`<span class="t">${ts()}</span><span class="${cls}">${html}</span>`;
  $log.appendChild(d);$log.scrollTop=$log.scrollHeight;
}
function plain(html){
  const d=document.createElement('div');d.className='row';
  d.innerHTML=`<span class="t"></span><span>${html}</span>`;
  $log.appendChild(d);$log.scrollTop=$log.scrollHeight;
}

function apply(d){
  const fit=d.best_fitness??0,gen=d.total_generations??0;
  const g=document.getElementById('s-gen');g.textContent=gen;g.className='v sage';
  const f=document.getElementById('s-fit');f.textContent=fit.toFixed(4);
  f.className='v '+(fit>0.95?'sage':fit>0.7?'rust':'gold');
  const t=document.getElementById('s-task');
  t.textContent=d.current_task?'task_'+d.current_task:'—';t.className='v '+(d.current_task?'rust':'faint');
  const ma=document.getElementById('s-ma');ma.textContent=d.mse_task_a?.toFixed(4)??'—';
  ma.className='v '+(d.mse_task_a<0.01?'sage':d.mse_task_a<0.15?'':'gold');
  document.getElementById('s-mb').textContent=d.mse_task_b?.toFixed(4)??'—';
  document.getElementById('s-bar').style.width=(fit*100).toFixed(1)+'%';
  const cl=document.getElementById('s-cl');
  if(d.continual_learning_active){cl.textContent='continual learning active — EWC + replay';cl.className='cl-txt on';}
  else{const e=gen<600?` · activates at gen 600 (${600-gen} left)`:'';cl.textContent='continual learning idle'+e;cl.className='cl-txt';}
  $dot.className='dot live';$state.textContent='live';
}

function setBusy(v){busy=v;$spin.style.display=v?'inline':'none';$cmd.disabled=v;}
function resetCd(){clearInterval(tick);cd=30;tick=setInterval(()=>{cd--;$state.textContent='live · refresh '+cd+'s';if(cd<=0){doStatus(true);resetCd();}},1000);}

async function doStatus(silent){
  try{
    const r=await fetch('/api/status');if(!r.ok)throw r.status;
    const d=await r.json();apply(d);
    if(!silent)row(JSON.stringify(d),'out');
  }catch(e){$dot.className='dot err';$state.textContent='error';if(!silent)row('status failed: '+e,'err');}
}
async function doRun(n){
  if(busy){row('busy — wait for current run','sys');return;}
  setBusy(true);row('run '+n,'cmd');row('evolving '+n+' generations…','sys');
  try{
    const r=await fetch('/api/evolve?generations='+n);if(!r.ok)throw r.status;
    const d=await r.json();apply(d);
    const sw=d.switched_task_this_call?` <span class="sw">★ task switch → task_${d.current_task}</span>`:'';
    row(`done · gen ${d.total_generations} · fit ${d.best_fitness} · mse_a ${d.mse_task_a} · mse_b ${d.mse_task_b}${sw}`,'out');
    if(d.continual_learning_active&&!d.switched_task_this_call)row('EWC + replay protecting prior task','sys');
  }catch(e){row('error: '+e,'err');}
  setBusy(false);resetCd();
}
function cmd(raw){
  const p=raw.trim().split(/\s+/),c=p[0].toLowerCase();
  if(c==='run')doRun(Math.max(1,Math.min(parseInt(p[1])||50,500)));
  else if(c==='status'){row('status','cmd');doStatus(false);}
  else if(c==='clear')$log.innerHTML='';
  else if(c==='help'){
    plain('<span class="head">commands</span>');
    plain('<span class="accent">run [N]</span>  evolve N generations · default 50 · max 500');
    plain('<span class="accent">status</span>   read state from Redis without evolving');
    plain('<span class="accent">clear</span>    clear output');
    plain('<span class="accent">help</span>     show this');
    plain('<span class="kbd">↑ ↓  command history</span>');
  }
  else row('unknown command: '+c+' — try help','err');
}
$cmd.addEventListener('keydown',e=>{
  if(e.key==='Enter'){const v=$cmd.value.trim();if(!v)return;hist.unshift(v);hIdx=-1;$cmd.value='';cmd(v);}
  else if(e.key==='ArrowUp'){e.preventDefault();hIdx=Math.min(hIdx+1,hist.length-1);$cmd.value=hist[hIdx]??'';}
  else if(e.key==='ArrowDown'){e.preventDefault();hIdx=Math.max(hIdx-1,-1);$cmd.value=hIdx<0?'':hist[hIdx];}
});

plain('<span class="head">DOT — self-evolving neuron</span>');
plain('<span class="sys">a single neuron that evolves on its own, persisting state across runs.</span>');
plain('<span class="sys">type <span class="accent">help</span> for commands · state auto-refreshes every 30s</span>');
plain('&nbsp;');
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
        return JSONResponse({"redis_connected": redis_connected, "status": "no_data",
                             "message": "no evolution has run yet"})
    individuals = state.get("population", {}).get("individuals", [])
    best_fitness = max((ind.get("fitness") or 0) for ind in individuals) if individuals else 0
    pop = population_from_dict(state["population"]) if "population" in state else None
    mse_a = round(mse_loss(pop.best_individual.genome, DATA_A), 4) if pop else None
    mse_b = round(mse_loss(pop.best_individual.genome, DATA_B), 4) if pop else None
    return JSONResponse({
        "redis_connected": redis_connected,
        "total_generations": state["generation"],
        "current_task": state["current_task"],
        "continual_learning_active": state.get("ewc_importance") is not None,
        "best_fitness": round(best_fitness, 4),
        "mse_task_a": mse_a,
        "mse_task_b": mse_b,
    })


@app.get("/api/evolve")
async def evolve_endpoint(
    request: Request,
    generations: int = Query(default=GENERATIONS_PER_RUN_DEFAULT),
):
    cron_secret = os.environ.get("CRON_SECRET")
    if cron_secret:
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {cron_secret}":
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    generations = max(1, min(generations, GENERATIONS_PER_RUN_MAX))
    try:
        result = run_batch(generations)
        return JSONResponse(result)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
