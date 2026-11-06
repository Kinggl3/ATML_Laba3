"""
marl_env — среда ЛР3: распределение частоты поездов метро Нью-Йорка между группами линий.

Формальная спецификация — report/env_spec.md, параметры и их обоснование — PROTOCOL.md (разд. 3–4).

Кратко. Эпизод = одни сутки (24 шага по часу). Агенты: 8 групп линий + координатор парка.
Каждый час агент линии выбирает множитель к расписанию MTA (0,7…1,3), координатор — часовой лимит
поездов в работе (0,85…1,15 от расписания). Защитный слой приводит запрос к допустимому: не реже
нормативного интервала, не больше пропускной способности, не больше лимита парка. Спрос на группу линий
экзогенный (входы на станциях, распределённые по линиям). Награда команды — минус обобщённые потери
времени пассажиров (ожидание + неуспевшие сесть + переполнение) в долях от расписания, минус штраф за
смену частоты, в конце суток — штраф за перерасход бюджета поездо-часов (бюджет = расписание).

Среда векторная: B суток идут параллельно, все эпизоды одной длины. Работает на torch (CPU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

# =============================================================================
# Постановка
# =============================================================================
GROUPS = [  # (имя агента, маршруты) — группы по общим магистралям (trunk lines)
    ("1·2·3", ["1", "2", "3"]),
    ("4·5·6", ["4", "5", "6"]),
    ("7", ["7"]),
    ("A·C·E", ["A", "C", "E"]),
    ("B·D·F·M", ["B", "D", "F", "M"]),
    ("N·Q·R·W", ["N", "Q", "R", "W"]),
    ("J·Z·L", ["J", "Z", "L"]),
    ("G", ["G"]),
]
DIVISION = {"A": [0, 1, 2], "B": [3, 4, 5, 6, 7]}       # дивизион A (IRT) и B (BMT/IND) — задание 4
ROUTE_MAP = {"6X": "6", "7X": "7", "FX": "F"}            # экспрессы — та же магистраль
NOT_CONTROLLED = {"S", "GS", "FS", "H", "SI", "SIR"}     # челноки и Staten Island — вне задачи
ROUTE2GROUP = {r: g for g, (_, rs) in enumerate(GROUPS) for r in rs}

MULTS = np.array([0.70, 0.85, 1.00, 1.15, 1.30], dtype="float32")    # действия агента линии
COORD = np.array([0.85, 0.925, 1.00, 1.075, 1.15], dtype="float32")  # действия координатора
N_ACT = 5
A_SCHED = 2                                                        # индекс действия «как в расписании»
H = 24


@dataclass
class EnvParams:
    """Параметры среды. Все значения зафиксированы в PROTOCOL.md до запуска."""
    rho_peak: float = 0.90      # калибровка: наполнение в типичный будний час пик по расписанию = 90% норматива
    rho0: float = 0.60          # с этого наполнения пассажир начинает «платить» за тесноту
    rho_max: float = 1.35       # физический предел посадки (135% норматива); сверх — не уехали с первым поездом
    ivt: float = 10.0           # мин. эффективной поездки в зоне максимального наполнения
    alpha: float = 1.0          # штраф тесноты: доп. минуты на минуту поездки на единицу (ρ − ρ0)
    head_floor_tph: float = 3.0  # нормативный интервал ≤ 20 мин на каждую работающую линию
    fleet_spare: float = 0.05   # запас парка сверх пикового выпуска по расписанию
    lam_budget: float = 5.0     # штраф за перерасход бюджета: 1% перерасхода = 5% суточных потерь
    lam_switch: float = 0.1     # штраф за смену частоты (оборот)
    over_tol: float = 0.005     # перерасход > 0,5% бюджета — нарушение
    use_coord: bool = True      # абляция: без координатора (лимит = весь парк)
    comm: bool = True           # абляция: без обмена сообщениями между агентами
    forecast: str = "norm"      # источник прогноза в наблюдении: norm | m4w | hyb | oracle | none
    kappa_scale: float = 1.0    # стресс: сдвиг калибровки наполнения
    fleet_scale: float = 1.0    # стресс: нехватка подвижного состава
    budget_scale: float = 1.0   # стресс: урезанный бюджет поездо-часов
    groups: tuple = tuple(range(len(GROUPS)))   # задание 4: подсеть (дивизион A или B)

    def tag(self) -> str:
        base = EnvParams()
        diff = {k: v for k, v in self.__dict__.items() if getattr(base, k) != v}
        return ",".join(f"{k}={v}" for k, v in sorted(diff.items())) or "default"


# =============================================================================
# Данные среды
# =============================================================================
@dataclass
class EnvData:
    """Всё, что нужно среде, в виде массивов «сутки × час × группа». Готовит marl_data.build_env_data."""
    day: np.ndarray            # (D,) datetime64[D]
    part: np.ndarray           # (D,) train | val | test | holdout
    dem: np.ndarray            # (D, 24, G) спрос группы линий, входов в час
    dem_prev: np.ndarray       # (D, 2, G) спрос в 22:00 и 23:00 предыдущих суток
    sched: np.ndarray          # (D, 24, G) поездов в час по расписанию (среднее двух направлений)
    nlines: np.ndarray         # (D, 24, G) число работающих линий группы в этот час
    cal: np.ndarray            # (D, 24, C) календарь: sin/cos часа, sin/cos дня недели, выходной, праздник, школа
    fc: dict                   # имя → (D, 24, 24, G): прогноз из точки h на час h+k (k = 0…23), NaN за концом суток
    rtt: np.ndarray            # (G,) время оборота поезда, ч (поезд в работе = поездов в час × оборот)
    kappa: np.ndarray          # (G,) наполнение ρ = kappa · спрос / поездов в час (калибровка по train)
    scale_d: np.ndarray        # (G,) нормировка спроса (среднее по train)
    scale_s: np.ndarray        # (G,) нормировка расписания (среднее по train)
    track_cap: np.ndarray      # (G,) пропускная способность: максимум поездов в час по расписанию за весь период
    fleet_max: float           # поездов в работе одновременно: пик расписания × (1 + запас)
    flags: dict = field(default_factory=dict)   # (D,) режимы дня: праздник, плановые работы, событие…
    meta: dict = field(default_factory=dict)

    @property
    def D(self) -> int:
        return len(self.day)

    @property
    def G(self) -> int:
        return self.dem.shape[-1]

    def idx(self, part: str) -> np.ndarray:
        return np.where(self.part == part)[0]

    # --- сохранение ---
    def save(self, path: Path):
        arrs = {k: getattr(self, k) for k in ["day", "part", "dem", "dem_prev", "sched", "nlines", "cal", "rtt",
                                              "kappa", "scale_d", "scale_s", "track_cap"]}
        arrs["part"] = arrs["part"].astype("U8")
        arrs["fleet_max"] = np.array(self.fleet_max)
        for k, v in self.fc.items():
            arrs[f"fc__{k}"] = v.astype("float32")
        for k, v in self.flags.items():
            arrs[f"flag__{k}"] = v
        np.savez_compressed(path, **arrs)
        import json
        Path(str(path).replace(".npz", "_meta.json")).write_text(json.dumps(self.meta, ensure_ascii=False, indent=1,
                                                                            default=str), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> "EnvData":
        import json
        z = np.load(path, allow_pickle=False)
        fc = {k[4:]: z[k] for k in z.files if k.startswith("fc__")}
        flags = {k[6:]: z[k] for k in z.files if k.startswith("flag__")}
        mp = Path(str(path).replace(".npz", "_meta.json"))
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}
        return EnvData(day=z["day"], part=z["part"].astype(object), dem=z["dem"], dem_prev=z["dem_prev"],
                       sched=z["sched"], nlines=z["nlines"], cal=z["cal"], fc=fc, rtt=z["rtt"], kappa=z["kappa"],
                       scale_d=z["scale_d"], scale_s=z["scale_s"], track_cap=z["track_cap"],
                       fleet_max=float(z["fleet_max"]), flags=flags, meta=meta)

    def subset_groups(self, groups) -> "EnvData":
        """Подсеть из части групп (задание 4: дивизион A / B). Парк делится пропорционально пиковому выпуску."""
        g = list(groups)
        insvc = (self.sched * self.rtt).sum(-1)
        insvc_g = (self.sched[..., g] * self.rtt[g]).sum(-1)
        share = float(np.nanmax(insvc_g) / max(np.nanmax(insvc), 1e-9))
        return replace(self, dem=self.dem[..., g], dem_prev=self.dem_prev[..., g], sched=self.sched[..., g],
                       nlines=self.nlines[..., g], fc={k: v[..., g] for k, v in self.fc.items()}, rtt=self.rtt[g],
                       kappa=self.kappa[g], scale_d=self.scale_d[g], scale_s=self.scale_s[g],
                       track_cap=self.track_cap[g], fleet_max=self.fleet_max * share,
                       meta={**self.meta, "groups": [GROUPS[i][0] for i in g]})


# =============================================================================
# Модель потерь пассажиров (одна для среды, MPC, оракула и метрик)
# =============================================================================
def hour_cost(d, n, kappa, P: EnvParams):
    """Потери пассажиро-минут за час на группе линий.

    d — входов в час, n — поездов в час, kappa — коэффициент наполнения. Возвращает словарь тензоров/массивов:
    J (всего), wait (ожидание H/2), extra (неуспевшие сесть ждут ещё интервал), crowd (теснота), rho (наполнение).
    """
    xp = torch if isinstance(d, torch.Tensor) else np
    n_safe = xp.clip(n, 1e-3, None) if xp is np else n.clamp(min=1e-3)
    rho = kappa * P.kappa_scale * d / n_safe
    wait = d * 30.0 / n_safe
    if xp is np:
        denied = d * np.clip(1.0 - P.rho_max / np.maximum(rho, 1e-6), 0.0, None)
        crowd = d * P.ivt * P.alpha * np.clip(rho - P.rho0, 0.0, None)
    else:
        denied = d * (1.0 - P.rho_max / rho.clamp(min=1e-6)).clamp(min=0.0)
        crowd = d * P.ivt * P.alpha * (rho - P.rho0).clamp(min=0.0)
    extra = denied * 60.0 / n_safe
    return {"J": wait + extra + crowd, "wait": wait, "extra": extra, "crowd": crowd, "rho": rho, "denied": denied}


def floors_caps(sched, nlines, track_cap, P: EnvParams):
    """Нижняя граница: нормативный интервал (≤ 20 мин на работающую линию), но не ниже 0,7 и не выше расписания.
    Верхняя: 1,3 расписания, но не выше пропускной способности магистрали (если расписание само выше — расписание)."""
    if isinstance(sched, torch.Tensor):
        lo = torch.minimum(sched, torch.maximum(MULTS[0] * sched, P.head_floor_tph * nlines))
        hi = torch.minimum(MULTS[-1] * sched, torch.maximum(track_cap.expand_as(sched), sched))
    else:
        lo = np.minimum(sched, np.maximum(MULTS[0] * sched, P.head_floor_tph * nlines))
        hi = np.minimum(MULTS[-1] * sched, np.maximum(track_cap, sched))
    return lo, hi


# =============================================================================
# Векторная среда
# =============================================================================
COMMON_DIM = 10
OWN_DIM = 11
MSG_DIM = 2


class MetroEnv:
    """B суток параллельно. reset(days) → obs, state; step(actions) → obs, state, reward, done, info."""

    def __init__(self, data: EnvData, P: EnvParams | None = None, device="cpu"):
        self.P = P or EnvParams()
        if tuple(self.P.groups) != tuple(range(len(GROUPS))):
            data = data.subset_groups(self.P.groups)
        self.data = data
        self.dev = device
        t = lambda a: torch.as_tensor(np.nan_to_num(np.asarray(a, dtype="float32")), device=device)
        self.G = data.G
        self.N = self.G + (1 if self.P.use_coord else 0)
        self.dem, self.dem_prev, self.sched = t(data.dem), t(data.dem_prev), t(data.sched)
        self.nlines, self.cal = t(data.nlines), t(data.cal)
        self.rtt, self.kappa = t(data.rtt), t(data.kappa)
        self.scale_d, self.scale_s = t(data.scale_d), t(data.scale_s)
        self.track_cap = t(data.track_cap)
        src = self.P.forecast
        if src == "oracle":
            fc = self._oracle_fc(data.dem)
        elif src == "none":
            fc = np.zeros_like(data.fc["norm"])
        else:
            if src not in data.fc:
                raise KeyError(f"нет прогноза '{src}' в данных: есть {list(data.fc)}")
            fc = data.fc[src]
        self.fc_nan = torch.as_tensor(np.isnan(fc).all(axis=(1, 2, 3)), device=device)   # сутки без прогноза
        self.fc = t(fc)
        # производные по суткам
        self.lo, self.hi = floors_caps(self.sched, self.nlines, self.track_cap, self.P)
        self.insvc_sched = (self.sched * self.rtt).sum(-1)                         # (D, 24) поездов в работе
        self.budget = self.insvc_sched.sum(-1) * self.P.budget_scale               # (D,) поездо-часов
        fm = float(data.fleet_max) * self.P.fleet_scale
        self.fleet_max = torch.full_like(self.insvc_sched, fm)
        if self.P.fleet_scale >= 1.0:   # расписание всегда выполнимо, если нет стресса нехватки парка
            self.fleet_max = torch.maximum(self.fleet_max, self.insvc_sched)
        J_s = hour_cost(self.dem, self.sched, self.kappa, self.P)["J"].sum(-1)    # (D, 24)
        self.J_sched_day = J_s.sum(-1).clamp(min=1.0)                              # (D,)
        self.obs_dim = COMMON_DIM + OWN_DIM + self.G * MSG_DIM + self.N
        self.state_dim = COMMON_DIM + (self.G + 1) * OWN_DIM
        self.n_act = N_ACT

    @staticmethod
    def _oracle_fc(dem):
        D, Hh, G = dem.shape
        out = np.full((D, Hh, Hh, G), np.nan, dtype="float32")
        for h in range(Hh):
            out[:, h, :Hh - h] = dem[:, h:]
        return out

    # ------------------------------------------------------------------
    def reset(self, days):
        self.b = torch.as_tensor(np.asarray(days), dtype=torch.long, device=self.dev)
        B = len(self.b)
        self.h = 0
        self.used = torch.zeros(B, device=self.dev)
        self.n_prev = self.sched[self.b, 0].clone()
        self.dev_prev = torch.zeros(B, self.G, device=self.dev)
        self.cum_dev = torch.zeros(B, self.G, device=self.dev)
        self.c_prev = torch.ones(B, device=self.dev)
        self.util_prev = torch.ones(B, device=self.dev)
        return self._obs()

    def _forecast(self, k):
        h = self.h + k
        if h > H - 1:
            return torch.zeros(len(self.b), self.G, device=self.dev)
        return self.fc[self.b, self.h, k]

    def _obs(self):
        b, h, G, B = self.b, self.h, self.G, len(self.b)
        sd, ss = self.scale_d, self.scale_s
        f0, f1, f2 = self._forecast(0), self._forecast(1), self._forecast(2)
        if h >= 2:
            dl1, dl2 = self.dem[b, h - 1], self.dem[b, h - 2]
        elif h == 1:
            dl1, dl2 = self.dem[b, 0], self.dem_prev[b, 1]
        else:
            dl1, dl2 = self.dem_prev[b, 1], self.dem_prev[b, 0]
        s_now = self.sched[b, h]
        s_next = self.sched[b, min(h + 1, H - 1)]
        kap = self.kappa * self.P.kappa_scale
        rho_hat = (kap * f0 / s_now.clamp(min=1e-3)).clamp(max=3.0)
        rho_last = (kap * dl1 / self.n_prev.clamp(min=1e-3)).clamp(max=3.0)
        m_last = self.n_prev / s_now.clamp(min=1e-3) - 1.0 if h == 0 else \
            self.n_prev / self.sched[b, h - 1].clamp(min=1e-3) - 1.0
        bud = self.budget[b]
        own = torch.stack([f0 / sd, f1 / sd, f2 / sd, dl1 / sd, dl2 / sd, s_now / ss, s_next / ss, rho_hat, rho_last,
                           m_last.clamp(-1, 1), self.cum_dev / (bud[:, None] / H)], dim=-1)          # (B, G, 11)
        sched_rem = self.insvc_sched[b, h:].sum(-1)
        slack = (bud - self.used - sched_rem) / bud
        common = torch.cat([self.cal[b, h], torch.stack([torch.full((B,), h / H, device=self.dev), slack,
                                                          self.c_prev - 1.0], -1)], -1)            # (B, 10)
        fm = self.fleet_max[b, h]
        coord = torch.stack([f0.sum(-1) / sd.sum(), rho_hat.mean(-1), rho_hat.max(-1).values,
                             self.insvc_sched[b, h] / fm, self.used / bud, slack, self.util_prev,
                             self.c_prev - 1.0, f1.sum(-1) / sd.sum(), dl1.sum(-1) / sd.sum(),
                             torch.full((B,), float(self.P.use_coord), device=self.dev)], -1)      # (B, 11)
        msg = torch.stack([rho_hat, m_last.clamp(-1, 1)], -1).reshape(B, G * MSG_DIM)
        if not self.P.comm:
            msg = torch.zeros_like(msg)
        own_all = torch.cat([own, coord[:, None]], 1) if self.P.use_coord else own                # (B, N, 11)
        eye = torch.eye(self.N, device=self.dev).expand(B, -1, -1)
        obs = torch.cat([common[:, None].expand(-1, self.N, -1), own_all,
                         msg[:, None].expand(-1, self.N, -1), eye], -1)
        state = torch.cat([common, own.reshape(B, -1), coord], -1)
        return obs, state

    # ------------------------------------------------------------------
    def apply_actions(self, actions):
        """Действия → поездов в час после защитного слоя. Возвращает n (B, G), c (B,), флаги вмешательства."""
        b, h = self.b, self.h
        s = self.sched[b, h]
        mult = torch.as_tensor(MULTS, device=self.dev)[actions[:, :self.G]]
        c = torch.as_tensor(COORD, device=self.dev)[actions[:, self.G]] if self.P.use_coord \
            else torch.full((len(b),), float("inf"), device=self.dev)
        lo, hi = self.lo[b, h], self.hi[b, h]
        req = mult * s
        n = torch.minimum(torch.maximum(req, lo), hi)
        clip_flag = (n - req).abs() > 1e-4
        fm = self.fleet_max[b, h]
        cap = torch.where(torch.isinf(c), fm, torch.minimum(c.clamp(max=1e6) * self.insvc_sched[b, h], fm))
        use = (n * self.rtt).sum(-1)
        floor_use = (lo * self.rtt).sum(-1)
        over = use > cap + 1e-4
        alpha = ((cap - floor_use) / (use - floor_use).clamp(min=1e-6)).clamp(0.0, 1.0)
        n = torch.where(over[:, None], lo + (n - lo) * alpha[:, None], n)
        return n, torch.where(torch.isinf(c), torch.ones_like(c), c), clip_flag, over, cap

    def step(self, actions):
        actions = torch.as_tensor(actions, device=self.dev, dtype=torch.long)
        b, h = self.b, self.h
        n, c, clip_flag, cap_flag, cap = self.apply_actions(actions)
        s = self.sched[b, h]
        d = self.dem[b, h]
        cost = hour_cost(d, n, self.kappa, self.P)
        th = (n * self.rtt).sum(-1)
        self.used = self.used + th
        dev = (n - s) * self.rtt
        switch = (dev - self.dev_prev).abs().sum(-1)
        bud = self.budget[b]
        jbar = self.J_sched_day[b] / H
        r = -cost["J"].sum(-1) / jbar - self.P.lam_switch * switch / (bud / H)
        self.cum_dev = self.cum_dev + dev
        self.dev_prev = dev
        self.n_prev = n
        self.c_prev = c
        self.util_prev = th / cap.clamp(min=1e-6)
        info = {"J": cost["J"].sum(-1), "wait": cost["wait"].sum(-1), "extra": cost["extra"].sum(-1),
                "crowd": cost["crowd"].sum(-1), "pax": d.sum(-1), "pax_crowded": (d * (cost["rho"] > 1.0)).sum(-1),
                "breach": (cost["rho"] > 1.0).float().sum(-1), "denied": cost["denied"].sum(-1),
                "trainh": th, "switch": switch, "clip": clip_flag.float().sum(-1), "capcut": cap_flag.float(),
                "n": n, "rho": cost["rho"], "J_g": cost["J"], "c": c}
        self.h += 1
        done = self.h >= H
        if done:
            overrun = (self.used / bud - 1.0).clamp(min=0.0)
            r = r - self.P.lam_budget * H * overrun
            info["overrun"] = overrun
            info["used_frac"] = self.used / bud
            obs, state = None, None
        else:
            obs, state = self._obs()
        return obs, state, r, done, info


# =============================================================================
# Неучащиеся политики: расписание, MPC на прогнозе, оракул
# =============================================================================
class SchedulePolicy:
    """Расписание MTA без изменений (контрольный baseline 1)."""
    name = "schedule"

    def act(self, env: MetroEnv, obs, state, deterministic=True):
        return torch.full((len(env.b), env.N), A_SCHED, dtype=torch.long)


class MPCPolicy:
    """Прогноз + оптимизация с горизонтом до конца суток (model predictive control).

    Каждый час: по прогнозу из источника `src` на оставшиеся часы выбирает множители из того же набора действий,
    что у агентов, минимизируя ту же функцию потерь при остатке бюджета и часовом лимите парка (двойственная
    декомпозиция: множитель λ на бюджет, μ_h на часы, бисекция). Исполняет решение текущего часа.
    src = 'oracle' — верхняя граница (идеальное знание спроса). Координатор держит максимум (лимит решает MPC).
    """

    def __init__(self, src="norm", name=None):
        self.src = src
        self.name = name or f"mpc_{src}"
        self._fc = None

    def _fc_for(self, env):
        if self._fc is None or self._fc[0] is not env:
            if self.src == "oracle":
                fc = MetroEnv._oracle_fc(env.data.dem)
            else:
                fc = env.data.fc[self.src]
            self._fc = (env, torch.as_tensor(np.nan_to_num(fc, nan=0.0)))
        return self._fc[1]

    @torch.no_grad()
    def act(self, env: MetroEnv, obs, state, deterministic=True):
        b, h, G = env.b, env.h, env.G
        fc = self._fc_for(env)[b, h, :H - h]                                     # (B, H', G)
        s = env.sched[b, h:]
        lo, hi = env.lo[b, h:], env.hi[b, h:]
        mult = torch.as_tensor(MULTS)
        n = torch.minimum(torch.maximum(mult * s[..., None], lo[..., None]), hi[..., None])  # (B, H', G, 5)
        C = hour_cost(fc[..., None], n, env.kappa[None, None, :, None], env.P)["J"]
        C = C + 1e-6 * C.mean() * (mult - 1.0).abs()          # при равенстве — действие ближе к расписанию
        U = n * env.rtt[None, None, :, None]
        R = (env.budget[b] - env.used)[:, None]                                 # (B, 1)
        cap = torch.minimum(float(COORD[-1]) * env.insvc_sched[b, h:], env.fleet_max[b, h:])   # как в среде при c = max
        scale = C.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1.0) / U.mean(dim=(1, 2, 3), keepdim=True).clamp(min=1e-3)

        def choose(lam, mu):
            score = C + (lam[:, None, None, None] + mu[..., None, None]) * scale * U
            a = score.argmin(-1)
            return a, torch.gather(U, -1, a[..., None])[..., 0]

        def hour_fix(lam):
            mu = torch.zeros_like(cap)
            a, u = choose(lam, mu)
            bad = u.sum(-1) > cap + 1e-4
            if bad.any():
                lo_m, hi_m = torch.zeros_like(cap), torch.full_like(cap, 1e3)
                for _ in range(30):
                    mid = 0.5 * (lo_m + hi_m)
                    _, u = choose(lam, mid)
                    ok = u.sum(-1) <= cap + 1e-4
                    hi_m = torch.where(ok, mid, hi_m)
                    lo_m = torch.where(ok, lo_m, mid)
                mu = torch.where(bad, hi_m, mu)
                a, u = choose(lam, mu)
            return a, u

        lam = torch.zeros(len(b))
        a, u = hour_fix(lam)
        need = u.sum(dim=(1, 2)) > R[:, 0] + 1e-3
        if need.any():
            lo_l, hi_l = torch.full_like(lam, -4.0), torch.full_like(lam, 3.0)      # log10 λ
            for _ in range(30):
                mid = 0.5 * (lo_l + hi_l)
                _, uu = hour_fix(10 ** mid)
                ok = uu.sum(dim=(1, 2)) <= R[:, 0] + 1e-3
                hi_l = torch.where(ok, mid, hi_l)
                lo_l = torch.where(ok, lo_l, mid)
            lam = torch.where(need, 10 ** hi_l, lam)
            a, u = hour_fix(lam)
        act = torch.full((len(b), env.N), len(COORD) - 1, dtype=torch.long)
        act[:, :G] = a[:, 0]
        return act


# =============================================================================
# Прогон политики и метрики по суткам
# =============================================================================
@torch.no_grad()
def rollout(env: MetroEnv, policy, days, deterministic=True, record=False, override=None):
    """Прогоняет политику на сутках `days`. Возвращает словарь суточных метрик (numpy, по одной строке на сутки).

    override(env, actions) → actions — для анализа вклада агентов (подмена действий одного агента расписанием).
    """
    obs, state = env.reset(days)
    acc = {k: 0.0 for k in ["J", "wait", "extra", "crowd", "pax", "pax_crowded", "breach", "denied", "trainh",
                            "switch", "clip", "capcut", "reward"]}
    rec = {"n": [], "act": [], "rho": [], "J_g": [], "c": []}
    done = False
    while not done:
        a = policy.act(env, obs, state, deterministic=deterministic)
        if override is not None:
            a = override(env, a)
        obs, state, r, done, info = env.step(a)
        for k in acc:
            acc[k] = acc[k] + (r if k == "reward" else info[k])
        if record:
            for k in ["n", "rho", "J_g", "c"]:
                rec[k].append(info[k].cpu().numpy())
            rec["act"].append(np.asarray(a.cpu() if isinstance(a, torch.Tensor) else a))
    out = {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)) for k, v in acc.items()}
    out["overrun"] = info["overrun"].cpu().numpy()
    out["used_frac"] = info["used_frac"].cpu().numpy()
    out["J_sched"] = env.J_sched_day[env.b].cpu().numpy()
    out["day"] = np.asarray(days)
    if record:
        out["trace"] = {k: np.stack(v, 1) for k, v in rec.items()}       # (B, 24, ...)
    return out
