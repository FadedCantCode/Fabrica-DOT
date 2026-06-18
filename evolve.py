"""
/api/evolve —— evolving_neuron.py 的「可以掛在網路上自己跑」版本

Vercel 沒有「一直跑的背景程式」這種東西,Function 是無狀態、每次呼叫都重新
啟動的——所以「自己跑」實際上是:每次被呼叫(cron 或外部排程器),從 Upstash
Redis 讀回上次的族群狀態,跑一批世代,再把新狀態存回去,然後結束。
連續不斷的「自己進化」感,是排程器的觸發頻率營造出來的,不是這支程式本身
在背景常駐。這點要先說清楚,不要讓人誤以為 Vercel 真的有一個程式在那邊跑。

跟本地的 evolving_neuron.py 邏輯完全相同(self-adaptive mutation、fitness
sharing、停滯偵測、experience replay、EWC、Adam+gradient clipping),只是
多包了一層「讀狀態 -> 跑 N 代 -> 存狀態 -> 回傳 JSON」。為了避免 Vercel
Python builder 處理多檔案匯入的不確定性,核心邏輯直接複製進這個檔案,
不是 import evolving_neuron。
"""

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import math
import os
import random
import urllib.request


# ============================================================
# 核心神經元邏輯(跟 evolving_neuron.py 同步)
# ============================================================

def basis(x: float) -> list:
    return [x, x * x, math.sin(x), math.cos(x), math.sin(2 * x), math.cos(2 * x)]


def full_features(x: float) -> list:
    return basis(x) + [1.0]


N_GENES = len(full_features(0.0))


def forward(genome: list, x: float) -> float:
    feats = full_features(x)
    return sum(g * f for g, f in zip(genome, feats))


def random_genome(rng: random.Random) -> list:
    return [rng.uniform(-1.0, 1.0) for _ in range(N_GENES)]


def mse_loss(genome: list, data: list) -> float:
    n = len(data)
    se = 0.0
    for x, y in data:
        se += (forward(genome, x) - y) ** 2
    return se / n


def total_loss(genome, data, ewc_importance=None, ewc_anchor=None, ewc_lambda=0.0) -> float:
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
    def __init__(self, capacity: int = 200, seed=None):
        self.capacity = capacity
        self.buffer: list = []
        self.rng = random.Random(seed)

    def add(self, x: float, y: float) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append((x, y))
        else:
            self.buffer[self.rng.randrange(self.capacity)] = (x, y)

    def sample(self, k: int) -> list:
        k = min(k, len(self.buffer))
        return self.rng.sample(self.buffer, k) if k > 0 else []


def compute_fisher_importance(genome: list, data: list) -> list:
    n = len(data)
    importance = [0.0] * N_GENES
    for x, y in data:
        feats = full_features(x)
        pred = sum(g * f for g, f in zip(genome, feats))
        error = pred - y
        for i, f in enumerate(feats):
            g_i = 2 * error * f
            importance[i] += g_i ** 2
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
            scale = clip_norm / norm
            grads = [g * scale for g in grads]
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


def euclid_dist(a, b) -> float:
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

    def _shared_fitness(self) -> list:
        genomes = [ind.genome for ind in self.individuals]
        shared = []
        for i, ind in enumerate(self.individuals):
            niche = 0.0
            for other in genomes:
                d = euclid_dist(genomes[i], other)
                if d < self.sigma_share:
                    niche += 1.0 - d / self.sigma_share
            shared.append(ind.fitness / niche if niche > 0 else ind.fitness)
        return shared

    def _tournament(self, shared_fit, k=3):
        idxs = self.rng.sample(range(len(self.individuals)), min(k, len(self.individuals)))
        best_idx = max(idxs, key=lambda i: shared_fit[i])
        return self.individuals[best_idx]

    def step(self, loss_fn, elitism=2) -> dict:
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

        stagnation_triggered = self.gens_since_improvement >= self.stagnation_patience
        if stagnation_triggered:
            self.gens_since_improvement = 0
            for ind in self.individuals:
                ind.sigma = min(ind.sigma * 2.0, 2.0)

        shared_fit = self._shared_fitness()
        next_gen = [self.individuals[i] for i in order[:elitism]]
        n_immigrants_this_gen = self.n_immigrants if stagnation_triggered else 0
        target_before_immigrants = self.pop_size - n_immigrants_this_gen
        while len(next_gen) < target_before_immigrants:
            p1 = self._tournament(shared_fit)
            p2 = self._tournament(shared_fit)
            child_genome = [self.rng.choice([g1, g2]) for g1, g2 in zip(p1.genome, p2.genome)]
            child_sigma = math.sqrt(p1.sigma * p2.sigma) * math.exp(self.tau * self.rng.gauss(0, 1))
            child_sigma = max(0.01, min(child_sigma, 2.0))
            child_genome = [g + child_sigma * self.rng.gauss(0, 1) for g in child_genome]
            next_gen.append(Individual(child_genome, child_sigma))
        while len(next_gen) < self.pop_size:
            next_gen.append(Individual(random_genome(self.rng), 0.5))
        self.individuals = next_gen

        return {
            "best_fitness": best_fitness,
            "stagnation_triggered": stagnation_triggered,
        }


def evolve_on_task(population, task_data, generations, replay_buffer=None,
                    ewc_importance=None, ewc_anchor=None, ewc_lambda=20.0,
                    use_replay=True, use_ewc=True, memetic_every=5, memetic_steps=30):
    imp = ewc_importance if use_ewc else None
    anc = ewc_anchor if use_ewc else None
    lam = ewc_lambda if use_ewc else 0.0

    for gen in range(generations):
        replay_sample = replay_buffer.sample(len(task_data)) if (use_replay and replay_buffer) else []
        data = task_data + replay_sample

        def loss_fn(genome, data=data):
            return total_loss(genome, data, imp, anc, lam)

        population.step(loss_fn)

        if memetic_every and gen % memetic_every == 0:
            elite = population.best_individual
            pre_loss = total_loss(elite.genome, data, imp, anc, lam)
            g = list(elite.genome)
            adam = AdamOptimizer(N_GENES, lr=0.05)
            for _ in range(memetic_steps):
                _, grad = total_loss_and_grad(g, data, imp, anc, lam)
                g = adam.step(g, grad, clip_norm=5.0)
            post_loss = total_loss(g, data, imp, anc, lam)
            if post_loss < pre_loss:
                elite.genome = g
                elite.fitness = 1.0 / (1.0 + post_loss)


# ============================================================
# 任務定義(跟 evolving_neuron.py 同步)
# ============================================================

def task_a(x):
    return math.sin(x) * math.cos(x / 2)


def task_b(x):
    return math.cos(2 * x) + 0.2 * x


_XS = [-math.pi + i * (2 * math.pi / 39) for i in range(40)]
DATA_A = [(x, task_a(x)) for x in _XS]
DATA_B = [(x, task_b(x)) for x in _XS]


# ============================================================
# 序列化:把 EvolvingPopulation 變成存得進 Redis 的 dict
# ============================================================

def population_to_dict(pop: EvolvingPopulation) -> dict:
    return {
        "pop_size": pop.pop_size,
        "sigma_share": pop.sigma_share,
        "stagnation_patience": pop.stagnation_patience,
        "n_immigrants": pop.n_immigrants,
        "gens_since_improvement": pop.gens_since_improvement,
        "last_best_fitness": pop._last_best_fitness,
        "individuals": [
            {"genome": ind.genome, "sigma": ind.sigma, "fitness": ind.fitness}
            for ind in pop.individuals
        ],
    }


def population_from_dict(d: dict) -> EvolvingPopulation:
    pop = EvolvingPopulation(
        pop_size=d["pop_size"], sigma_share=d["sigma_share"],
        stagnation_patience=d["stagnation_patience"], n_immigrants=d["n_immigrants"],
    )
    pop.individuals = [Individual(raw["genome"], raw["sigma"]) for raw in d["individuals"]]
    for ind, raw in zip(pop.individuals, d["individuals"]):
        ind.fitness = raw["fitness"]
    pop.gens_since_improvement = d["gens_since_improvement"]
    pop._last_best_fitness = d["last_best_fitness"]
    pop.best_individual = max(pop.individuals, key=lambda i: i.fitness if i.fitness is not None else -1)
    return pop


# ============================================================
# Upstash Redis REST 持久化(serverless function 之間沒有記憶體,
# 狀態一定要存在外部某個地方,這裡用 Upstash 的 HTTP REST API,
# 不需要額外裝 redis 套件)
# ============================================================

REDIS_URL = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
STATE_KEY = "evolving_neuron_state"


def redis_command(*args):
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    req = urllib.request.Request(
        REDIS_URL,
        data=json.dumps(list(args)).encode("utf-8"),
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


def save_state(state: dict) -> None:
    redis_command("SET", STATE_KEY, json.dumps(state))


# ============================================================
# 一次呼叫 = 跑一批世代
# ============================================================

GENERATIONS_PER_RUN_DEFAULT = 50
GENERATIONS_PER_RUN_MAX = 500
SWITCH_EVERY = 600  # 累積這麼多代之後切換任務,持續操練 replay+EWC 保護機制


def run_batch(generations: int) -> dict:
    state = load_state()
    redis_connected = REDIS_URL is not None and REDIS_TOKEN is not None

    if state is None:
        pop = EvolvingPopulation(pop_size=24)
        state = {
            "generation": 0,
            "current_task": "a",
            "ewc_importance": None,
            "ewc_anchor": None,
            "replay": [],
        }
    else:
        pop = population_from_dict(state["population"])

    replay = ReplayBuffer(capacity=200)
    replay.buffer = [tuple(p) for p in state["replay"]]

    task_data = DATA_A if state["current_task"] == "a" else DATA_B
    have_protection = state["ewc_importance"] is not None

    evolve_on_task(
        pop, task_data, generations=generations,
        replay_buffer=replay if have_protection else None,
        ewc_importance=state["ewc_importance"], ewc_anchor=state["ewc_anchor"],
        use_replay=have_protection, use_ewc=have_protection,
    )

    prev_total = state["generation"]
    state["generation"] = prev_total + generations
    switched = False

    if prev_total // SWITCH_EVERY != state["generation"] // SWITCH_EVERY:
        finished_genome = list(pop.best_individual.genome)
        importance = compute_fisher_importance(finished_genome, task_data)
        for x, y in task_data:
            replay.add(x, y)
        state["ewc_importance"] = importance
        state["ewc_anchor"] = finished_genome
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


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        cron_secret = os.environ.get("CRON_SECRET")
        if cron_secret:
            auth = self.headers.get("Authorization", "")
            if auth != f"Bearer {cron_secret}":
                self.send_response(401)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "unauthorized"}).encode("utf-8"))
                return

        query = parse_qs(urlparse(self.path).query)
        generations = int(query.get("generations", [GENERATIONS_PER_RUN_DEFAULT])[0])
        generations = max(1, min(generations, GENERATIONS_PER_RUN_MAX))

        try:
            result = run_batch(generations)
            status = 200
        except Exception as exc:  # noqa: BLE001
            result = {"error": str(exc)}
            status = 500

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
