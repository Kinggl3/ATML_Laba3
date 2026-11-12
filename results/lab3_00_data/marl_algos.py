"""
marl_algos — 8 MARL-алгоритмов и одноагентный PPO для среды marl_env (PyTorch, CPU).

Все алгоритмы реализованы в одном файле по одной схеме, чтобы сравнение было честным (PROTOCOL.md, разд. 5):
одна среда, одни наблюдения, один бюджет взаимодействий (шагов среды), одни seed, одна оценка на val для
выбора чекпойнта, одинаковый размер сетей (2 слоя × 128, LayerNorm на входе).

  on-policy (PPO-семейство):  IPPO, MAPPO, HAPPO, MAT, PPO (один агент, факторизованное действие)
  on-policy actor-critic:     COMA (контрфактический baseline)
  off-policy:                 MADDPG (Gumbel-Softmax для дискретных действий), VDN, QMIX

Награда внутри обучения сдвинута на +1 за шаг (расписание ≈ 0): длина эпизода фиксирована (24),
поэтому сдвиг не меняет оптимальную политику, но упрощает обучение критиков.

Источники: Schulman et al. 2017 (PPO); de Witt et al. 2020 (IPPO); Yu et al. 2022 (MAPPO); Kuba et al. 2022
(HAPPO); Wen et al. 2022 (MAT); Foerster et al. 2018 (COMA); Lowe et al. 2017 (MADDPG); Sunehag et al. 2018
(VDN); Rashid et al. 2018 (QMIX). Гиперпараметры по умолчанию — из этих работ и их библиотек (MAPPO, PyMARL).
"""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl_env import H, MetroEnv

REWARD_SHIFT = 1.0
HID = 128

DEFAULTS = {
    "common": dict(gamma=0.99, hid=HID, grad_clip=10.0),
    "ppo_family": dict(lr=5e-4, lr_critic=5e-4, n_days=32, epochs=5, minibatches=4, clip=0.2, ent=0.01,
                       gae_lambda=0.95),
    "coma": dict(lr=5e-4, lr_critic=5e-4, n_days=8, critic_epochs=2, minibatches=4, td_lambda=0.8, ent=0.01,
                 target_every=4),
    "maddpg": dict(lr=5e-4, lr_critic=5e-4, n_days=8, buffer=200_000, batch=256, upd_per_step=0.1, tau=0.01,
                   gumbel_tau=1.0, logit_reg=1e-3, warmup_steps=5_000),
    "q_family": dict(lr=5e-4, n_days=8, buffer_eps=5_000, batch_eps=32, upd_per_ep=1.0, eps_start=1.0,
                     eps_end=0.05, eps_frac=0.15, target_every_eps=200, mix_embed=32, hyper_hid=64),
}
FAMILY = {"ippo": "ppo_family", "mappo": "ppo_family", "happo": "ppo_family", "mat": "ppo_family",
          "ppo": "ppo_family", "coma": "coma", "maddpg": "maddpg", "vdn": "q_family", "qmix": "q_family"}
ALGOS = ["ippo", "mappo", "happo", "mat", "coma", "maddpg", "vdn", "qmix"]       # 8 MARL
BASELINES_LEARNED = ["ppo"]
SIMPLICITY = ["ippo", "vdn", "mappo", "ppo", "qmix", "maddpg", "coma", "happo", "mat"]  # правило 4 выбора
TITLE = {"ippo": "IPPO", "mappo": "MAPPO", "happo": "HAPPO", "mat": "MAT", "coma": "COMA", "maddpg": "MADDPG",
         "vdn": "VDN", "qmix": "QMIX", "ppo": "PPO (один агент)", "schedule": "Расписание MTA",
         "mpc_m4w": "MPC: медиана 4 нед.", "mpc_norm": "MPC: norm_level", "mpc_hyb": "MPC: гибрид ВКР",
         "mpc_oracle": "Оракул (идеальный прогноз)"}


def config(algo: str, **over) -> dict:
    c = {**DEFAULTS["common"], **DEFAULTS[FAMILY[algo]]}
    c.update(over)
    return c


# =============================================================================
# Сети
# =============================================================================
def mlp(inp, out, hid=HID, layers=2, out_gain=1.0):
    mods, d = [nn.LayerNorm(inp)], inp
    for _ in range(layers):
        lin = nn.Linear(d, hid)
        nn.init.orthogonal_(lin.weight, math.sqrt(2))
        nn.init.zeros_(lin.bias)
        mods += [lin, nn.ReLU()]
        d = hid
    last = nn.Linear(d, out)
    nn.init.orthogonal_(last.weight, out_gain)
    nn.init.zeros_(last.bias)
    return nn.Sequential(*mods, last)


class ValueNorm:
    """Нормировка целей критика скользящими средним и дисперсией (как ValueNorm в MAPPO)."""

    def __init__(self, beta=0.999):
        self.beta, self.m, self.v, self.w = beta, 0.0, 1.0, 0.0

    def update(self, x):
        x = x.detach().reshape(-1)
        b = self.beta
        self.m = b * self.m + (1 - b) * float(x.mean())
        self.v = b * self.v + (1 - b) * float((x ** 2).mean())
        self.w = b * self.w + (1 - b)

    @property
    def mean(self):
        return self.m / max(self.w, 1e-6)

    @property
    def std(self):
        return math.sqrt(max(self.v / max(self.w, 1e-6) - self.mean ** 2, 1e-6))

    def norm(self, x):
        return (x - self.mean) / self.std

    def denorm(self, x):
        return x * self.std + self.mean


def gae(rew, val, gamma, lam):
    """rew (T, …), val (T+1, …) с нулём в конце эпизода → advantages, returns."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(T)):
        delta = rew[t] + gamma * val[t + 1] - val[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + val[:-1]


# =============================================================================
# Базовый класс: цикл обучения с оценкой на val и выбором лучшего чекпойнта
# =============================================================================
class Learner:
    off_policy = False

    def __init__(self, env: MetroEnv, cfg: dict, seed: int):
        self.cfg, self.seed = cfg, seed
        self.N, self.od, self.sd, self.A = env.N, env.obs_dim, env.state_dim, env.n_act
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.steps = 0
        self.history = []

    # --- переопределяется ---
    def act(self, env, obs, state, deterministic=True):
        raise NotImplementedError

    def iterate(self, env, days_pool):
        """Один цикл: сбор данных + обновление. Возвращает (число шагов среды, средняя награда эпизода)."""
        raise NotImplementedError

    def modules(self) -> dict:
        raise NotImplementedError

    # --- общее ---
    def state_dict(self):
        return {k: copy.deepcopy(m.state_dict()) for k, m in self.modules().items()}

    def load_state_dict(self, sd):
        for k, m in self.modules().items():
            m.load_state_dict(sd[k])

    def n_params(self):
        return int(sum(p.numel() for m in self.modules().values() for p in m.parameters()))

    def train(self, env, train_days, budget, eval_fn, n_evals=12, log=None, time_limit=None):
        """budget — шагов среды (сутки × 24, по всем параллельным суткам). eval_fn(self) → оценка на val (больше — лучше)."""
        t0 = time.perf_counter()
        next_eval = budget / n_evals
        best, best_sd, best_step = -1e18, None, 0
        ep_rewards = []
        while self.steps < budget:
            n, r = self.iterate(env, train_days)
            self.steps += n
            ep_rewards.append(r)
            if self.steps >= next_eval or self.steps >= budget:
                with torch.no_grad():
                    v = float(eval_fn(self))
                rec = {"step": self.steps, "train_reward": float(np.mean(ep_rewards[-20:])), "val_score": v,
                       "sec": time.perf_counter() - t0}
                self.history.append(rec)
                if v > best:
                    best, best_sd, best_step = v, self.state_dict(), self.steps
                if log:
                    log(f"    {self.steps:>8d} шагов  train {rec['train_reward']:+.3f}  val {v:+.4f}  "
                        f"{rec['sec']:.0f} c")
                next_eval += budget / n_evals
            if time_limit and time.perf_counter() - t0 > time_limit:
                if log:
                    log("    остановлено по лимиту времени")
                break
        if best_sd is not None:
            self.load_state_dict(best_sd)
        self.best_step, self.best_val = best_step, best
        self.train_sec = time.perf_counter() - t0
        return self.history

    def sample_days(self, pool, n):
        return self.rng.choice(pool, size=n, replace=True)

    @torch.no_grad()
    def collect(self, env, days, policy_fn):
        """Прогон эпизодов с записью траекторий (T = 24). policy_fn(obs, state) → actions, extra(dict тензоров)."""
        obs, state = env.reset(days)
        buf = {"obs": [], "state": [], "act": [], "rew": [], "extra": []}
        done = False
        while not done:
            a, ex = policy_fn(env, obs, state)
            buf["obs"].append(obs)
            buf["state"].append(state)
            buf["act"].append(a)
            buf["extra"].append(ex)
            obs, state, r, done, info = env.step(a)
            buf["rew"].append(r + REWARD_SHIFT)
        out = {k: torch.stack(buf[k]) for k in ["obs", "state", "act", "rew"]}
        for k in buf["extra"][0]:
            out[k] = torch.stack([e[k] for e in buf["extra"]])
        out["ep_reward"] = float(sum(buf["rew"]).mean()) - REWARD_SHIFT * H
        return out


def categorical(logits, deterministic):
    if deterministic:
        return logits.argmax(-1)
    return torch.distributions.Categorical(logits=logits).sample()


def clip_step(opt, params, max_norm):
    nn.utils.clip_grad_norm_(params, max_norm)
    opt.step()


# =============================================================================
# PPO-семейство
# =============================================================================
class PPOBase(Learner):
    """Общий код для IPPO / MAPPO / HAPPO / PPO (один агент)."""
    centralized_v = True

    def __init__(self, env, cfg, seed):
        super().__init__(env, cfg, seed)
        self.vnorm = ValueNorm()
        self.build()

    def build(self):
        raise NotImplementedError

    def logits(self, obs, state):                # → (B, N, A)
        raise NotImplementedError

    def value(self, obs, state):                 # → (B,) или (B, N)
        raise NotImplementedError

    def act(self, env, obs, state, deterministic=True):
        return categorical(self.logits(obs, state), deterministic)

    def iterate(self, env, pool):
        c = self.cfg
        days = self.sample_days(pool, c["n_days"])

        def pf(env_, o, s):
            lg = self.logits(o, s)
            a = categorical(lg, False)
            lp = torch.log_softmax(lg, -1).gather(-1, a[..., None])[..., 0]
            return a, {"logp": lp, "v": self.value(o, s)}

        tr = self.collect(env, days, pf)
        self.update(tr)
        return c["n_days"] * H, tr["ep_reward"]

    def advantages(self, tr):
        c = self.cfg
        v = self.vnorm.denorm(tr["v"])                                 # (T, B) или (T, B, N)
        v = torch.cat([v, torch.zeros_like(v[:1])], 0)
        rew = tr["rew"] if v.dim() == 2 else tr["rew"][..., None].expand_as(v[:-1])
        adv, ret = gae(rew, v, c["gamma"], c["gae_lambda"])
        self.vnorm.update(ret)
        return adv, ret

    def policy_loss(self, lg, act, old_lp, adv, w=None):
        c = self.cfg
        lp_all = torch.log_softmax(lg, -1)
        lp = lp_all.gather(-1, act[..., None])[..., 0]
        ratio = torch.exp(lp - old_lp)
        if w is not None:
            adv = adv * w
        s1, s2 = ratio * adv, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * adv
        ent = -(lp_all.exp() * lp_all).sum(-1)
        return -torch.min(s1, s2).mean() - c["ent"] * ent.mean()

    def update(self, tr):
        c = self.cfg
        adv, ret = self.advantages(tr)
        T, B = tr["rew"].shape
        flat = lambda x: x.reshape(T * B, *x.shape[2:])
        obs, st, act, olp = flat(tr["obs"]), flat(tr["state"]), flat(tr["act"]), flat(tr["logp"])
        adv, ret = flat(adv), flat(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        if adv.dim() == 1:
            adv = adv[:, None].expand(-1, self.N)
        tgt = self.vnorm.norm(ret)
        n = T * B
        for _ in range(c["epochs"]):
            perm = torch.randperm(n)
            for mb in perm.chunk(c["minibatches"]):
                lg = self.logits(obs[mb], st[mb])
                loss_pi = self.policy_loss(lg, act[mb], olp[mb], adv[mb])
                self.opt_pi.zero_grad()
                loss_pi.backward()
                clip_step(self.opt_pi, self.pi_params(), c["grad_clip"])
                v = self.value(obs[mb], st[mb])
                loss_v = F.mse_loss(v, tgt[mb])
                self.opt_v.zero_grad()
                loss_v.backward()
                clip_step(self.opt_v, self.v_params(), c["grad_clip"])

    def pi_params(self):
        return list(self.actor.parameters())

    def v_params(self):
        return list(self.critic.parameters())

    def modules(self):
        return {"actor": self.actor, "critic": self.critic}


class IPPO(PPOBase):
    """Независимые PPO-агенты с общими параметрами; критик видит только своё наблюдение."""

    def build(self):
        c = self.cfg
        self.actor = mlp(self.od, self.A, c["hid"], out_gain=0.01)
        self.critic = mlp(self.od, 1, c["hid"])
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=c["lr"], eps=1e-5)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr_critic"], eps=1e-5)

    def logits(self, obs, state):
        return self.actor(obs)

    def value(self, obs, state):
        return self.critic(obs)[..., 0]


class MAPPO(IPPO):
    """PPO с централизованным критиком V(s) (CTDE), общий актор по наблюдениям агентов."""

    def build(self):
        c = self.cfg
        self.actor = mlp(self.od, self.A, c["hid"], out_gain=0.01)
        self.critic = mlp(self.sd, 1, c["hid"])
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=c["lr"], eps=1e-5)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr_critic"], eps=1e-5)

    def value(self, obs, state):
        return self.critic(state)[..., 0]


class SinglePPO(MAPPO):
    """Контрольный baseline: один агент PPO управляет всеми группами сразу (вход — полное состояние,
    действие факторизовано по агентам; отношение вероятностей — по совместному действию)."""

    def build(self):
        c = self.cfg
        self.actor = mlp(self.sd, self.N * self.A, c["hid"], out_gain=0.01)
        self.critic = mlp(self.sd, 1, c["hid"])
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=c["lr"], eps=1e-5)
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr_critic"], eps=1e-5)

    def logits(self, obs, state):
        return self.actor(state).reshape(*state.shape[:-1], self.N, self.A)

    def policy_loss(self, lg, act, old_lp, adv, w=None):
        c = self.cfg
        lp_all = torch.log_softmax(lg, -1)
        lp = lp_all.gather(-1, act[..., None])[..., 0].sum(-1)
        ratio = torch.exp(lp - old_lp.sum(-1))
        a = adv[:, 0]
        s1, s2 = ratio * a, ratio.clamp(1 - c["clip"], 1 + c["clip"]) * a
        ent = -(lp_all.exp() * lp_all).sum(-1).sum(-1)
        return -torch.min(s1, s2).mean() - c["ent"] * ent.mean()


class HAPPO(MAPPO):
    """Heterogeneous-Agent PPO: у каждого агента своя сеть; агенты обновляются по очереди в случайном порядке,
    преимущество взвешивается произведением отношений вероятностей уже обновлённых агентов (Kuba et al., 2022)."""

    def build(self):
        c = self.cfg
        self.actors = nn.ModuleList([mlp(self.od, self.A, c["hid"], out_gain=0.01) for _ in range(self.N)])
        self.critic = mlp(self.sd, 1, c["hid"])
        self.opts = [torch.optim.Adam(a.parameters(), lr=c["lr"], eps=1e-5) for a in self.actors]
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=c["lr_critic"], eps=1e-5)

    def logits(self, obs, state):
        return torch.stack([self.actors[i](obs[..., i, :]) for i in range(self.N)], -2)

    def modules(self):
        return {"actors": self.actors, "critic": self.critic}

    def update(self, tr):
        c = self.cfg
        adv, ret = self.advantages(tr)
        T, B = tr["rew"].shape
        flat = lambda x: x.reshape(T * B, *x.shape[2:])
        obs, st, act, olp = flat(tr["obs"]), flat(tr["state"]), flat(tr["act"]), flat(tr["logp"])
        adv, ret = flat(adv), flat(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tgt = self.vnorm.norm(ret)
        n = T * B
        for _ in range(c["epochs"]):                                   # критик
            for mb in torch.randperm(n).chunk(c["minibatches"]):
                loss_v = F.mse_loss(self.value(obs[mb], st[mb]), tgt[mb])
                self.opt_v.zero_grad()
                loss_v.backward()
                clip_step(self.opt_v, self.v_params(), c["grad_clip"])
        M = torch.ones(n)
        for i in torch.randperm(self.N).tolist():                      # акторы по очереди
            net, opt = self.actors[i], self.opts[i]
            for _ in range(c["epochs"]):
                for mb in torch.randperm(n).chunk(c["minibatches"]):
                    lg = net(obs[mb, i])
                    loss = self.policy_loss(lg, act[mb, i], olp[mb, i], adv[mb], w=M[mb])
                    opt.zero_grad()
                    loss.backward()
                    clip_step(opt, net.parameters(), c["grad_clip"])
            with torch.no_grad():
                new_lp = torch.log_softmax(net(obs[:, i]), -1).gather(-1, act[:, i, None])[:, 0]
                M = M * torch.exp(new_lp - olp[:, i]).clamp(0.0, 10.0)


# =============================================================================
# MAT — Multi-Agent Transformer (Wen et al., 2022)
# =============================================================================
class MATNet(nn.Module):
    def __init__(self, od, N, A, d=64, heads=2):
        super().__init__()
        self.N, self.A, self.d = N, A, d
        self.obs_emb = nn.Sequential(nn.LayerNorm(od), nn.Linear(od, d), nn.GELU())
        self.enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.v_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.act_emb = nn.Sequential(nn.Linear(A + 1, d), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(N, d))
        self.dec = nn.TransformerDecoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.pi_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, A))
        nn.init.normal_(self.pos, std=0.02)
        self.register_buffer("mask", torch.triu(torch.full((N, N), float("-inf")), 1))

    def encode(self, obs):
        rep = self.enc(self.obs_emb(obs))
        return rep, self.v_head(rep)[..., 0]

    def tokens(self, act_prev):
        """act_prev (…, N) действия предыдущих агентов (сдвиг на 1); первый токен — «старт»."""
        B = act_prev.shape[:-1]
        x = torch.zeros(*B, self.N, self.A + 1)
        x[..., 0, 0] = 1.0
        oh = F.one_hot(act_prev[..., :-1], self.A).float()
        x[..., 1:, 1:] = oh
        return x

    def decode(self, tok, rep, upto=None):
        n = tok.shape[-2] if upto is None else upto
        h = self.act_emb(tok[..., :n, :]) + self.pos[:n]
        h = self.dec(h, rep, tgt_mask=self.mask[:n, :n])
        return self.pi_head(h)


class MAT(PPOBase):
    def build(self):
        c = self.cfg
        self.net = MATNet(self.od, self.N, self.A)
        self.opt_pi = torch.optim.Adam(self.net.parameters(), lr=c["lr"], eps=1e-5)

    def modules(self):
        return {"net": self.net}

    def value(self, obs, state):
        return self.net.encode(obs)[1]

    @torch.no_grad()
    def _sample(self, obs, deterministic):
        rep, v = self.net.encode(obs)
        B = obs.shape[0]
        act = torch.zeros(B, self.N, dtype=torch.long)
        logits = torch.zeros(B, self.N, self.A)
        tok = torch.zeros(B, self.N, self.A + 1)
        tok[:, 0, 0] = 1.0
        for i in range(self.N):
            lg = self.net.decode(tok, rep, upto=i + 1)[:, i]
            a = categorical(lg, deterministic)
            act[:, i], logits[:, i] = a, lg
            if i + 1 < self.N:
                tok[:, i + 1, 1:] = F.one_hot(a, self.A).float()
        return act, logits, v

    def act(self, env, obs, state, deterministic=True):
        return self._sample(obs, deterministic)[0]

    def iterate(self, env, pool):
        c = self.cfg
        days = self.sample_days(pool, c["n_days"])

        def pf(env_, o, s):
            a, lg, v = self._sample(o, False)
            lp = torch.log_softmax(lg, -1).gather(-1, a[..., None])[..., 0]
            return a, {"logp": lp, "v": v}

        tr = self.collect(env, days, pf)
        self.update(tr)
        return c["n_days"] * H, tr["ep_reward"]

    def update(self, tr):
        c = self.cfg
        adv, ret = self.advantages(tr)
        T, B = tr["rew"].shape
        flat = lambda x: x.reshape(T * B, *x.shape[2:])
        obs, act, olp, adv, ret = flat(tr["obs"]), flat(tr["act"]), flat(tr["logp"]), flat(adv), flat(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        tgt = self.vnorm.norm(ret)
        n = T * B
        for _ in range(c["epochs"]):
            for mb in torch.randperm(n).chunk(c["minibatches"]):
                rep, v = self.net.encode(obs[mb])
                lg = self.net.decode(self.net.tokens(act[mb]), rep)
                loss = self.policy_loss(lg, act[mb], olp[mb], adv[mb]) + F.mse_loss(v, tgt[mb])
                self.opt_pi.zero_grad()
                loss.backward()
                clip_step(self.opt_pi, self.net.parameters(), c["grad_clip"])


# =============================================================================
# COMA — контрфактический многоагентный градиент политики (Foerster et al., 2018)
# =============================================================================
class COMA(Learner):
    def __init__(self, env, cfg, seed):
        super().__init__(env, cfg, seed)
        c = cfg
        self.actor = mlp(self.od, self.A, c["hid"], out_gain=0.01)
        cin = self.sd + self.od + self.N * self.A
        self.critic = mlp(cin, self.A, c["hid"])
        self.critic_t = copy.deepcopy(self.critic)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=c["lr"], eps=1e-5)
        self.opt_q = torch.optim.Adam(self.critic.parameters(), lr=c["lr_critic"], eps=1e-5)
        self.it = 0

    def modules(self):
        return {"actor": self.actor, "critic": self.critic}

    def act(self, env, obs, state, deterministic=True):
        return categorical(self.actor(obs), deterministic)

    def critic_in(self, state, obs, act):
        """Вход критика агента i: состояние, наблюдение i, действия остальных (действие i обнулено)."""
        N, A = self.N, self.A
        oh = F.one_hot(act, A).float()                                   # (…, N, A)
        others = oh[..., None, :, :].expand(*oh.shape[:-2], N, N, A).clone()
        idx = torch.arange(N)
        others[..., idx, idx, :] = 0.0
        others = others.reshape(*oh.shape[:-2], N, N * A)
        st = state[..., None, :].expand(*state.shape[:-1], N, state.shape[-1])
        return torch.cat([st, obs, others], -1)

    def iterate(self, env, pool):
        c = self.cfg
        days = self.sample_days(pool, c["n_days"])

        def pf(env_, o, s):
            return categorical(self.actor(o), False), {}

        tr = self.collect(env, days, pf)
        T, B = tr["rew"].shape
        x = self.critic_in(tr["state"], tr["obs"], tr["act"])            # (T, B, N, cin)
        with torch.no_grad():
            q_t = self.critic_t(x).gather(-1, tr["act"][..., None])[..., 0]   # (T, B, N)
            G = torch.zeros_like(q_t)
            nxt = torch.zeros_like(q_t[0])
            for t in reversed(range(T)):
                boot = (1 - c["td_lambda"]) * q_t[t + 1] if t + 1 < T else torch.zeros_like(nxt)
                G[t] = tr["rew"][t][:, None] + c["gamma"] * (boot + c["td_lambda"] * nxt)
                nxt = G[t] if t + 1 <= T else nxt
        xf, af, Gf = x.reshape(T * B, self.N, -1), tr["act"].reshape(T * B, self.N), G.reshape(T * B, self.N)
        for _ in range(c["critic_epochs"]):
            for mb in torch.randperm(T * B).chunk(c["minibatches"]):
                q = self.critic(xf[mb]).gather(-1, af[mb][..., None])[..., 0]
                loss = F.mse_loss(q, Gf[mb])
                self.opt_q.zero_grad()
                loss.backward()
                clip_step(self.opt_q, self.critic.parameters(), c["grad_clip"])
        with torch.no_grad():
            q_all = self.critic(xf)                                      # (TB, N, A)
        lg = self.actor(tr["obs"].reshape(T * B, self.N, -1))
        lp_all = torch.log_softmax(lg, -1)
        pi = lp_all.exp()
        base = (pi.detach() * q_all).sum(-1)
        qa = q_all.gather(-1, af[..., None])[..., 0]
        adv = qa - base
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        lp = lp_all.gather(-1, af[..., None])[..., 0]
        ent = -(pi * lp_all).sum(-1)
        loss = -(adv.detach() * lp).mean() - c["ent"] * ent.mean()
        self.opt_pi.zero_grad()
        loss.backward()
        clip_step(self.opt_pi, self.actor.parameters(), c["grad_clip"])
        self.it += 1
        if self.it % c["target_every"] == 0:
            self.critic_t.load_state_dict(self.critic.state_dict())
        return c["n_days"] * H, tr["ep_reward"]


# =============================================================================
# Буфер переходов для off-policy
# =============================================================================
class Replay:
    def __init__(self, cap):
        self.cap, self.n, self.i, self.d = cap, 0, 0, None

    def add(self, **kw):
        k0 = next(iter(kw.values()))
        m = k0.shape[0]
        if self.d is None:
            self.d = {k: torch.zeros((self.cap,) + v.shape[1:], dtype=v.dtype) for k, v in kw.items()}
        idx = (torch.arange(m) + self.i) % self.cap
        for k, v in kw.items():
            self.d[k][idx] = v
        self.i = (self.i + m) % self.cap
        self.n = min(self.n + m, self.cap)

    def sample(self, m, rng):
        j = torch.as_tensor(rng.integers(0, self.n, size=m))
        return {k: v[j] for k, v in self.d.items()}


def episodes_to_transitions(tr):
    """(T, B, …) → переходы (T·B, …) с next и done; на последнем шаге next — нули, done = 1."""
    T, B = tr["rew"].shape
    nxt = lambda x: torch.cat([x[1:], torch.zeros_like(x[:1])], 0)
    done = torch.zeros(T, B)
    done[-1] = 1.0
    f = lambda x: x.reshape(T * B, *x.shape[2:])
    return dict(obs=f(tr["obs"]), state=f(tr["state"]), act=f(tr["act"]), rew=f(tr["rew"]),
                obs2=f(nxt(tr["obs"])), state2=f(nxt(tr["state"])), done=f(done))


# =============================================================================
# MADDPG (дискретные действия через Gumbel-Softmax)
# =============================================================================
class MADDPG(Learner):
    off_policy = True

    def __init__(self, env, cfg, seed):
        super().__init__(env, cfg, seed)
        c = cfg
        self.actor = mlp(self.od, self.A, c["hid"], out_gain=0.01)
        self.critic = mlp(self.sd + self.N * self.A, 1, c["hid"])
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.opt_pi = torch.optim.Adam(self.actor.parameters(), lr=c["lr"])
        self.opt_q = torch.optim.Adam(self.critic.parameters(), lr=c["lr_critic"])
        self.buf = Replay(c["buffer"])

    def modules(self):
        return {"actor": self.actor, "critic": self.critic}

    def act(self, env, obs, state, deterministic=True):
        lg = self.actor(obs)
        if deterministic:
            return lg.argmax(-1)
        return F.gumbel_softmax(lg, tau=self.cfg["gumbel_tau"], hard=True).argmax(-1)

    def q(self, net, state, oh):
        return net(torch.cat([state, oh.reshape(*oh.shape[:-2], -1)], -1))[..., 0]

    def iterate(self, env, pool):
        c = self.cfg
        days = self.sample_days(pool, c["n_days"])
        warm = self.steps < c["warmup_steps"]

        def pf(env_, o, s):
            if warm:
                return torch.randint(0, self.A, o.shape[:-1]), {}
            return self.act(env_, o, s, deterministic=False), {}

        tr = self.collect(env, days, pf)
        self.buf.add(**episodes_to_transitions(tr))
        n = c["n_days"] * H
        if self.buf.n >= c["batch"]:
            for _ in range(max(1, int(n * c["upd_per_step"]))):
                self.update(self.buf.sample(c["batch"], self.rng))
        return n, tr["ep_reward"]

    def update(self, b):
        c = self.cfg
        oh = F.one_hot(b["act"], self.A).float()
        with torch.no_grad():
            a2 = F.one_hot(self.actor_t(b["obs2"]).argmax(-1), self.A).float()
            y = b["rew"] + c["gamma"] * (1 - b["done"]) * self.q(self.critic_t, b["state2"], a2)
        loss_q = F.mse_loss(self.q(self.critic, b["state"], oh), y)
        self.opt_q.zero_grad()
        loss_q.backward()
        clip_step(self.opt_q, self.critic.parameters(), c["grad_clip"])
        lg = self.actor(b["obs"])                                          # (m, N, A)
        g = F.gumbel_softmax(lg, tau=c["gumbel_tau"], hard=True)
        loss_pi = 0.0
        for i in range(self.N):                                            # подстановка действия агента i
            joint = torch.cat([oh[:, :i], g[:, i:i + 1], oh[:, i + 1:]], 1)
            loss_pi = loss_pi - self.q(self.critic, b["state"], joint).mean()
        loss_pi = loss_pi / self.N + c["logit_reg"] * (lg ** 2).mean()
        self.opt_pi.zero_grad()
        loss_pi.backward()
        clip_step(self.opt_pi, self.actor.parameters(), c["grad_clip"])
        with torch.no_grad():
            for net, tgt in [(self.actor, self.actor_t), (self.critic, self.critic_t)]:
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(1 - c["tau"]).add_(c["tau"] * p)


# =============================================================================
# VDN и QMIX
# =============================================================================
class QMixer(nn.Module):
    def __init__(self, N, sd, embed=32, hyper=64):
        super().__init__()
        self.N, self.e = N, embed
        self.ln = nn.LayerNorm(sd)
        self.w1 = nn.Sequential(nn.Linear(sd, hyper), nn.ReLU(), nn.Linear(hyper, N * embed))
        self.b1 = nn.Linear(sd, embed)
        self.w2 = nn.Sequential(nn.Linear(sd, hyper), nn.ReLU(), nn.Linear(hyper, embed))
        self.b2 = nn.Sequential(nn.Linear(sd, embed), nn.ReLU(), nn.Linear(embed, 1))

    def forward(self, q, state):                     # q (m, N), state (m, sd) → (m,)
        s = self.ln(state)
        w1 = self.w1(s).abs().view(-1, self.N, self.e)
        h = F.elu(torch.bmm(q[:, None, :], w1)[:, 0] + self.b1(s))
        w2 = self.w2(s).abs()[..., None]
        return (torch.bmm(h[:, None, :], w2)[:, 0, 0] + self.b2(s)[:, 0])


class VDN(Learner):
    off_policy = True
    mixer_cls = None

    def __init__(self, env, cfg, seed):
        super().__init__(env, cfg, seed)
        c = cfg
        self.agent = mlp(self.od, self.A, c["hid"])
        self.mixer = QMixer(self.N, self.sd, c["mix_embed"], c["hyper_hid"]) if self.mixer_cls == "qmix" else None
        self.agent_t = copy.deepcopy(self.agent)
        self.mixer_t = copy.deepcopy(self.mixer) if self.mixer is not None else None
        params = list(self.agent.parameters()) + (list(self.mixer.parameters()) if self.mixer is not None else [])
        self.params = params
        self.opt = torch.optim.Adam(params, lr=c["lr"])
        self.buf = Replay(c["buffer_eps"] * H)
        self.eps_count = 0
        self.budget = None

    def modules(self):
        d = {"agent": self.agent}
        if self.mixer is not None:
            d["mixer"] = self.mixer
        return d

    def eps(self):
        c = self.cfg
        frac = min(1.0, self.steps / max(1.0, c["eps_frac"] * (self.budget or 1e6)))
        return c["eps_start"] + frac * (c["eps_end"] - c["eps_start"])

    def act(self, env, obs, state, deterministic=True):
        q = self.agent(obs)
        if deterministic:
            return q.argmax(-1)
        a = q.argmax(-1)
        rnd = torch.rand(a.shape) < self.eps()
        return torch.where(rnd, torch.randint(0, self.A, a.shape), a)

    def mix(self, mixer, q, state):
        return q.sum(-1) if mixer is None else mixer(q, state)

    def train(self, env, train_days, budget, eval_fn, **kw):
        self.budget = budget
        return super().train(env, train_days, budget, eval_fn, **kw)

    def iterate(self, env, pool):
        c = self.cfg
        days = self.sample_days(pool, c["n_days"])
        tr = self.collect(env, days, lambda e, o, s: (self.act(e, o, s, deterministic=False), {}))
        self.buf.add(**episodes_to_transitions(tr))
        prev = self.eps_count
        self.eps_count += c["n_days"]
        if self.buf.n >= c["batch_eps"] * H:
            for _ in range(max(1, int(c["n_days"] * c["upd_per_ep"]))):
                self.update(self.buf.sample(c["batch_eps"] * H, self.rng))
        if self.eps_count // c["target_every_eps"] > prev // c["target_every_eps"]:
            self.agent_t.load_state_dict(self.agent.state_dict())
            if self.mixer is not None:
                self.mixer_t.load_state_dict(self.mixer.state_dict())
        return c["n_days"] * H, tr["ep_reward"]

    def update(self, b):
        c = self.cfg
        q = self.agent(b["obs"]).gather(-1, b["act"][..., None])[..., 0]          # (m, N)
        qt = self.mix(self.mixer, q, b["state"])
        with torch.no_grad():
            a2 = self.agent(b["obs2"]).argmax(-1, keepdim=True)                     # double Q
            q2 = self.agent_t(b["obs2"]).gather(-1, a2)[..., 0]
            y = b["rew"] + c["gamma"] * (1 - b["done"]) * self.mix(self.mixer_t, q2, b["state2"])
        loss = F.smooth_l1_loss(qt, y)
        self.opt.zero_grad()
        loss.backward()
        clip_step(self.opt, self.params, c["grad_clip"])


class QMIX(VDN):
    mixer_cls = "qmix"


CLASSES = {"ippo": IPPO, "mappo": MAPPO, "happo": HAPPO, "mat": MAT, "ppo": SinglePPO, "coma": COMA,
           "maddpg": MADDPG, "vdn": VDN, "qmix": QMIX}


def make(algo: str, env: MetroEnv, seed: int, **over) -> Learner:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return CLASSES[algo](env, config(algo, **over), seed)
