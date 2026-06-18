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
<title>DOT — evolving neuron</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080C0B;--bg2:#0F1512;--bg3:#172018;
  --green:#6FFF6F;--green-dim:#2A5C2A;
  --amber:#FFB454;--cyan:#5EE6E6;--red:#FF6B6B;--bone:#C8DDD0;
  --border:rgba(111,255,111,0.15);
  --font:'JetBrains Mono',monospace;
}
html,body{height:100%;background:var(--bg);color:var(--green);font:13px/1.6 var(--font);-webkit-font-smoothing:antialiased}
body{display:flex;flex-direction:column;height:100vh;overflow:hidden}

/* TOP BAR */
#topbar{
  display:flex;align-items:center;gap:12px;
  padding:10px 16px;background:var(--bg2);
  border-bottom:1px solid var(--border);flex-shrink:0;
}
#topbar .logo{font-size:15px;font-weight:700;letter-spacing:0.08em;color:var(--green)}
#topbar .sep{color:var(--green-dim)}
#topbar .sub{font-size:11px;color:var(--green-dim);letter-spacing:0.06em;text-transform:uppercase}
#live-dot{margin-left:auto;font-size:11px;letter-spacing:0.08em}
#auto-label{font-size:11px;color:var(--green-dim)}

/* STATS STRIP */
#statsbar{
  display:flex;gap:0;flex-shrink:0;
  border-bottom:1px solid var(--border);background:var(--bg2);
}
.sblock{
  padding:10px 18px;border-right:1px solid var(--border);
  display:flex;flex-direction:column;gap:2px;
}
.sblock:last-child{border-right:none}
.slabel{font-size:9px;color:var(--green-dim);letter-spacing:0.12em;text-transform:uppercase}
.sval{font-size:20px;font-weight:700;color:var(--cyan);line-height:1}
.sval.good{color:var(--green)}
.sval.warn{color:var(--amber)}
.sval.na{color:var(--green-dim)}
#bar-block{flex:1;display:flex;flex-direction:column;justify-content:center;padding:10px 18px;gap:6px}
.bartrack{height:3px;background:var(--green-dim);border-radius:2px;overflow:hidden}
.barfill{height:100%;background:var(--green);border-radius:2px;transition:width .6s ease}
#cl-line{font-size:10px;color:var(--green-dim);letter-spacing:0.04em}
#cl-line.active{color:var(--cyan)}

/* MAIN TERMINAL */
#terminal{
  flex:1;overflow:hidden;display:flex;flex-direction:column;
  padding:12px 16px 0;gap:0;
}
#log{
  flex:1;overflow-y:auto;font-size:12px;line-height:1.75;
  padding-bottom:4px;
}
#log::-webkit-scrollbar{width:4px}
#log::-webkit-scrollbar-track{background:transparent}
#log::-webkit-scrollbar-thumb{background:var(--green-dim);border-radius:2px}
.ll{display:flex;gap:10px;align-items:baseline}
.lt{color:var(--green-dim);flex-shrink:0;font-size:11px}
.lc{color:var(--amber)}
.lo{color:var(--bone)}
.ls{color:var(--green-dim);font-style:italic}
.ler{color:var(--red)}
.lh{color:var(--cyan)}
.lsw{color:var(--amber);font-weight:700}

/* INPUT ROW */
#inputrow{
  display:flex;align-items:center;gap:8px;
  padding:10px 0 12px;border-top:1px solid var(--border);margin-top:8px;
}
#prompt{color:var(--green);font-weight:700;user-select:none}
#cmdinput{
  flex:1;background:none;border:none;outline:none;
  color:var(--green);font:13px/1 var(--font);
  caret-color:var(--green);
}
#cmdinput::placeholder{color:var(--green-dim)}
#spinner{color:var(--amber);display:none;animation:spin 1s linear infinite;font-size:14px}
@keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>

<div id="topbar">
  <span class="logo">DOT</span>
  <span class="sep">/</span>
  <span class="sub">evolving neuron runtime</span>
  <span id="live-dot" style="color:var(--green-dim)">○ connecting</span>
  <span id="auto-label"></span>
</div>

<div id="statsbar">
  <div class="sblock">
    <div class="slabel">generation</div>
    <div class="sval na" id="s-gen">—</div>
  </div>
  <div class="sblock">
    <div class="slabel">best fitness</div>
    <div class="sval na" id="s-fit">—</div>
  </div>
  <div class="sblock">
    <div class="slabel">task</div>
    <div class="sval na" id="s-task">—</div>
  </div>
  <div class="sblock">
    <div class="slabel">mse task_a</div>
    <div class="sval na" id="s-ma">—</div>
  </div>
  <div class="sblock">
    <div class="slabel">mse task_b</div>
    <div class="sval na" id="s-mb">—</div>
  </div>
  <div id="bar-block">
    <div class="bartrack"><div class="barfill" id="s-bar" style="width:0"></div></div>
    <div id="cl-line">continual learning —</div>
  </div>
</div>

<div id="terminal">
  <div id="log"></div>
  <div id="inputrow">
    <span id="prompt">$</span>
    <input id="cmdinput" placeholder="run [N] · status · help · clear" autocomplete="off" spellcheck="false">
    <span id="spinner">⟳</span>
  </div>
</div>

<script>
const $log   = document.getElementById('log');
const $cmd   = document.getElementById('cmdinput');
const $spin  = document.getElementById('spinner');
const $live  = document.getElementById('live-dot');
const $auto  = document.getElementById('auto-label');

let hist=[], hIdx=-1, busy=false, autoTimer=null, countdown=30, tickTimer=null;

const ts = () => new Date().toTimeString().slice(0,8);

function line(text, cls){
  const d=document.createElement('div');
  d.className='ll';
  d.innerHTML=`<span class="lt">[${ts()}]</span><span class="${cls}">${text}</span>`;
  $log.appendChild(d);
  $log.scrollTop=$log.scrollHeight;
}

function applyStats(d){
  const fit = d.best_fitness ?? 0;
  const gen = d.total_generations ?? 0;

  const sg = document.getElementById('s-gen');
  sg.textContent = gen;
  sg.className = 'sval good';

  const sf = document.getElementById('s-fit');
  sf.textContent = fit.toFixed(4);
  sf.className = 'sval ' + (fit>0.95?'good':fit>0.7?'':'warn');

  const st = document.getElementById('s-task');
  st.textContent = d.current_task ? 'task_'+d.current_task : '—';
  st.className = 'sval ' + (d.current_task?'':'na');

  document.getElementById('s-ma').textContent = d.mse_task_a?.toFixed(4)??'—';
  document.getElementById('s-ma').className = 'sval ' + (d.mse_task_a<0.01?'good':d.mse_task_a<0.1?'':'warn');
  document.getElementById('s-mb').textContent = d.mse_task_b?.toFixed(4)??'—';

  document.getElementById('s-bar').style.width = (fit*100).toFixed(1)+'%';

  const cl = document.getElementById('cl-line');
  if(d.continual_learning_active){
    cl.textContent='● continual learning active — EWC + replay';
    cl.className='cl-line active';
  } else {
    const eta = gen<600 ? ` — activates at gen 600 (${600-gen} to go)` : '';
    cl.textContent='○ continual learning inactive'+eta;
    cl.className='cl-line';
  }

  $live.textContent='● live';
  $live.style.color='var(--green)';
}

function setBusy(v){
  busy=v;
  $spin.style.display=v?'inline':'none';
  $cmd.disabled=v;
}

function resetAutoCountdown(){
  clearInterval(tickTimer);
  countdown=30;
  tickTimer=setInterval(()=>{
    countdown--;
    $auto.textContent=`auto-refresh in ${countdown}s`;
    if(countdown<=0){ doStatus(true); resetAutoCountdown(); }
  },1000);
  $auto.textContent=`auto-refresh in ${countdown}s`;
}

async function doStatus(silent){
  try{
    const r=await fetch('/api/status');
    if(!r.ok) throw new Error(r.status);
    const d=await r.json();
    applyStats(d);
    if(!silent) line(JSON.stringify(d),'lo');
  }catch(e){
    $live.textContent='✕ error';
    $live.style.color='var(--red)';
    if(!silent) line('fetch /api/status failed: '+e,'ler');
  }
}

async function doRun(gens){
  if(busy){line('already running, wait...','ls');return;}
  setBusy(true);
  line(`run ${gens}`,'lc');
  line('evolving...','ls');
  try{
    const r=await fetch('/api/evolve?generations='+gens);
    if(!r.ok) throw new Error(r.status);
    const d=await r.json();
    applyStats(d);
    const sw=d.switched_task_this_call?` <span class="lsw">★ task switch → ${d.current_task}</span>`:'';
    line(`✓ gen=${d.total_generations} &nbsp;fit=${d.best_fitness} &nbsp;mse_a=${d.mse_task_a} &nbsp;mse_b=${d.mse_task_b}${sw}`,'lo');
    if(d.continual_learning_active && !d.switched_task_this_call)
      line('   EWC + replay active — old task memory protected','ls');
  }catch(e){
    line('error: '+e,'ler');
  }
  setBusy(false);
  resetAutoCountdown();
}

function handleCmd(raw){
  const p=raw.trim().split(/\s+/);
  const c=p[0].toLowerCase();
  if(c==='run'){
    doRun(Math.max(1,Math.min(parseInt(p[1])||50,500)));
  }else if(c==='status'){
    line('status','lc');
    doStatus(false);
  }else if(c==='clear'){
    $log.innerHTML='';
  }else if(c==='help'){
    [
      '<span class="lh">available commands</span>',
      '  <span class="lc">run [N]</span>   &nbsp;evolve N generations (default 50, max 500)',
      '  <span class="lc">status</span>    &nbsp;read current state from Redis without evolving',
      '  <span class="lc">clear</span>     &nbsp;clear terminal output',
      '  <span class="lc">help</span>      &nbsp;show this message',
      '',
      '  <span class="ls">↑ ↓ arrow keys navigate command history</span>',
    ].forEach(t=>{
      const d=document.createElement('div');
      d.className='ll';
      d.innerHTML=`<span class="lt">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span>${t}</span>`;
      $log.appendChild(d);
    });
    $log.scrollTop=$log.scrollHeight;
  }else{
    line(`unknown: ${c} — type help`,'ler');
  }
}

$cmd.addEventListener('keydown', e=>{
  if(e.key==='Enter'){
    const v=$cmd.value.trim();
    if(!v)return;
    hist.unshift(v); hIdx=-1; $cmd.value='';
    handleCmd(v);
  }else if(e.key==='ArrowUp'){
    e.preventDefault();
    hIdx=Math.min(hIdx+1,hist.length-1);
    $cmd.value=hist[hIdx]??'';
  }else if(e.key==='ArrowDown'){
    e.preventDefault();
    hIdx=Math.max(hIdx-1,-1);
    $cmd.value=hIdx<0?'':hist[hIdx];
  }
});

// boot
line('<span class="lh">DOT evolving neuron terminal</span>','lo');
line('type <span class="lc">help</span> to see commands &nbsp;·&nbsp; status polls every 30s','ls');
line('','ls');
doStatus(true);
resetAutoCountdown();
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
