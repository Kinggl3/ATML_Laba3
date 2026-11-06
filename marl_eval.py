"""
marl_eval — оценка политик ЛР3: суточные метрики, доверительные интервалы, сводный балл и правило выбора,
кооперация агентов, стресс-тесты, запуск обучения (в т.ч. параллельно) с журналом параметров.

Соответствие метрик заданию (финансовый пример → метро), PROTOCOL.md разд. 6:
  доходность            → выигрыш: 1 − потери политики / потери расписания за сутки, среднее по суткам
  Sharpe                → IR: среднее / ст. откл. суточного выигрыша
  Sortino               → среднее / нижнее отклонение (только сутки хуже расписания)
  максимальная просадка → худшие сутки (минимальный суточный выигрыш)
  CVaR 95%              → средний выигрыш в 5% худших суток
  оборот                → перераспределение поездо-часов между часами, % суточного бюджета
  издержки              → поездо-часы (% бюджета) и штраф за смену частоты
  нарушения лимитов     → перерасход бюджета > 0,5% (доля суток); часы с наполнением выше норматива;
                          вмешательства защитного слоя
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from marl_env import (A_SCHED, GROUPS, H, EnvData, EnvParams, MetroEnv, MPCPolicy, SchedulePolicy, rollout)

TOL_OVERRUN = 0.005
WEIGHTS = {"gain": 0.35, "ir": 0.20, "cvar": 0.20, "crowded": 0.10, "turnover": 0.10, "seed_sd": 0.05}
HIGHER_BETTER = {"gain": True, "ir": True, "cvar": True, "crowded": False, "turnover": False, "seed_sd": False}
BOOT = 2000


def to_md(df: pd.DataFrame, floatfmt="{:.4g}") -> str:
    """Markdown-таблица без зависимости от tabulate."""
    cols = [str(c) for c in df.columns]
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


# =============================================================================
# Суточные метрики
# =============================================================================
def daily_frame(r: dict, env: MetroEnv, policy: str, seed, part: str) -> pd.DataFrame:
    b = r["day"]
    bud = env.budget[torch.as_tensor(b)].numpy()
    G = env.G
    df = pd.DataFrame({
        "policy": policy, "seed": seed, "part": part, "day": env.data.day[b],
        "J": r["J"], "J_sched": r["J_sched"], "wait": r["wait"], "extra": r["extra"], "crowd": r["crowd"],
        "pax": r["pax"], "pax_crowded": r["pax_crowded"], "breach_share": r["breach"] / (H * G),
        "denied": r["denied"], "used_frac": r["used_frac"], "overrun": r["overrun"],
        "turnover": r["switch"] / bud, "clip_share": r["clip"] / (H * G), "cap_share": r["capcut"] / H,
        "reward": r["reward"],
    })
    df["gain"] = 1.0 - df["J"] / df["J_sched"]
    for k, v in env.data.flags.items():
        df[f"flag_{k}"] = v[b]
    return df


def evaluate(policy, env: MetroEnv, days, name: str, seed=None, part="") -> pd.DataFrame:
    if len(days) == 0:
        return pd.DataFrame()
    t = time.perf_counter()
    r = rollout(env, policy, days)
    df = daily_frame(r, env, name, seed, part)
    df["infer_ms"] = (time.perf_counter() - t) / (len(days) * H) * 1000
    return df


def summarize(df: pd.DataFrame) -> dict:
    """Метрики по набору суток одной политики (одного seed или нескольких — тогда среднее по seed по каждым суткам)."""
    if df["seed"].nunique() > 1:
        num = df.groupby("day").mean(numeric_only=True)
    else:
        num = df.set_index("day").select_dtypes("number")
    g = num["gain"].to_numpy()
    k = max(1, int(np.ceil(0.05 * len(g))))
    down = np.sqrt(np.mean(np.minimum(g, 0) ** 2))
    sortino = g.mean() / down if down > 1e-9 else (np.inf if g.mean() > 0 else np.nan)   # ∞ — нет суток хуже расписания
    return {
        "gain": g.mean(), "ir": g.mean() / (g.std() + 1e-9), "sortino": sortino, "worst": g.min(),
        "cvar": np.sort(g)[:k].mean(), "cost_per_pax": num["J"].sum() / num["pax"].sum(),
        "wait_per_pax": num["wait"].sum() / num["pax"].sum(),
        "crowded": num["pax_crowded"].sum() / num["pax"].sum(), "breach": num["breach_share"].mean(),
        "turnover": num["turnover"].mean(), "used_frac": num["used_frac"].mean(),
        "viol_budget": (num["overrun"] > TOL_OVERRUN).mean(), "clip_share": num["clip_share"].mean(),
        "cap_share": num["cap_share"].mean(), "reward": num["reward"].mean(), "days": len(g),
    }


def table(daily: pd.DataFrame, by=("policy",)) -> pd.DataFrame:
    """Сводная таблица: метрики по среднему seed + разброс по seed (ст. откл. выигрыша и IR между seed)."""
    rows = []
    for key, grp in daily.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(summarize(grp))
        per_seed = [summarize(s) for _, s in grp.groupby("seed", dropna=False)]
        row["n_seeds"] = len(per_seed)
        row["seed_sd"] = float(np.std([p["gain"] for p in per_seed])) if len(per_seed) > 1 else 0.0
        row["ir_seed_sd"] = float(np.std([p["ir"] for p in per_seed])) if len(per_seed) > 1 else 0.0
        if "infer_ms" in grp:
            row["infer_ms"] = grp["infer_ms"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Доверительные интервалы: стратифицированный бутстреп (seed × недельные блоки суток)
# =============================================================================
def _week_blocks(days):
    d = pd.DatetimeIndex(days)
    wk = ((d - d.min()).days // 7).to_numpy()
    return [np.where(wk == w)[0] for w in np.unique(wk)]


def boot_ci(daily: pd.DataFrame, col="gain", n=BOOT, seed=0, alpha=0.05) -> tuple[float, float]:
    """95% ДИ среднего col: случайно выбираем seed (с возвращением) и недели (с возвращением)."""
    rng = np.random.default_rng(seed)
    piv = daily.pivot_table(index="day", columns="seed", values=col, dropna=False)
    M = piv.to_numpy()
    blocks = _week_blocks(piv.index)
    S = M.shape[1]
    est = np.empty(n)
    for i in range(n):
        ss = rng.integers(0, S, S)
        bb = np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])
        est[i] = np.nanmean(M[np.ix_(bb, ss)])
    return float(np.quantile(est, alpha / 2)), float(np.quantile(est, 1 - alpha / 2))


def paired_diff(da: pd.DataFrame, db: pd.DataFrame, col="gain", n=BOOT, seed=0) -> dict:
    """Разница A − B (средние по seed по каждым суткам), блочный бутстреп по неделям: ДИ и двусторонний p."""
    rng = np.random.default_rng(seed)
    a = da.groupby("day")[col].mean()
    b = db.groupby("day")[col].mean()
    d = (a - b).dropna()
    blocks = _week_blocks(d.index)
    v = d.to_numpy()
    est = np.array([v[np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), len(blocks))])].mean()
                    for _ in range(n)])
    p = 2 * min((est <= 0).mean(), (est >= 0).mean())
    return {"diff": float(v.mean()), "lo": float(np.quantile(est, 0.025)), "hi": float(np.quantile(est, 0.975)),
            "p": float(min(1.0, p)), "days": len(v)}


# =============================================================================
# Сводный балл и правило выбора (PROTOCOL.md разд. 7)
# =============================================================================
def composite(tab: pd.DataFrame) -> pd.Series:
    sc = pd.Series(0.0, index=tab.index)
    for k, w in WEIGHTS.items():
        x = tab[k].astype(float)
        lo, hi = x.min(), x.max()
        z = (x - lo) / (hi - lo) if hi > lo else pd.Series(1.0, index=x.index)
        sc += w * (z if HIGHER_BETTER[k] else 1 - z)
    return sc


def select(tab_val: pd.DataFrame, daily_val: pd.DataFrame, simplicity: list, log=print) -> dict:
    """1) допустимы: перерасход бюджета ≤ 2% суток и нижняя граница ДИ выигрыша > 0;
    2) максимум сводного балла; 3) если более простой кандидат не хуже по выигрышу значимо (ДИ разницы
    содержит 0) и его балл отстаёт не больше чем на 0,10 — берётся более простой."""
    t = tab_val.copy()
    t["score"] = composite(t)
    t["ci_lo"] = [boot_ci(daily_val[daily_val["policy"] == p])[0] for p in t["policy"]]
    t["admissible"] = (t["viol_budget"] <= 0.02) & (t["ci_lo"] > 0)
    pool = t[t["admissible"]] if t["admissible"].any() else t
    if not t["admissible"].any():
        log("ВНИМАНИЕ: ни один алгоритм не прошёл условия допуска — выбор среди всех (записать в отступления)")
    top = pool.sort_values("score", ascending=False).iloc[0]
    choice, reason = top["policy"], "максимум сводного балла"
    rank = {p: i for i, p in enumerate(simplicity)}
    for _, row in pool.sort_values("score", ascending=False).iterrows():
        if rank.get(row["policy"], 99) < rank.get(choice, 99) and row["score"] >= top["score"] - 0.10:
            pd_ = paired_diff(daily_val[daily_val["policy"] == top["policy"]],
                              daily_val[daily_val["policy"] == row["policy"]])
            if pd_["lo"] <= 0 <= pd_["hi"]:
                choice = row["policy"]
                reason = (f"более простой, разница выигрыша с {top['policy']} незначима "
                          f"({pd_['diff'] * 100:+.2f} п.п., ДИ [{pd_['lo'] * 100:+.2f}; {pd_['hi'] * 100:+.2f}])")
    return {"choice": choice, "reason": reason, "table": t.sort_values("score", ascending=False)}


# =============================================================================
# Кооперация агентов
# =============================================================================
def cooperation(policy, env: MetroEnv, days, name="") -> dict:
    """Вклад агентов и согласованность действий на сутках days (детерминированная политика).

    loo_i   — насколько падает выигрыш, если действие агента i заменить расписанием (остальные — политика);
    solo_i  — выигрыш, если по политике действует только агент i (остальные — расписание);
    synergy — выигрыш всех минус сумма solo: > 0 — агенты дополняют друг друга (перераспределение ресурса);
    realloc — доля поездо-часов, перенесённых между группами относительно расписания;
    align   — корреляция «добавленные поезда ↔ ожидаемое наполнение по расписанию» (куда отправляются поезда).
    """
    G, N = env.G, env.N
    full = rollout(env, policy, days, record=True)
    g_full = float(np.mean(1 - full["J"] / full["J_sched"]))

    def with_override(f):
        r = rollout(env, policy, days, override=f)
        return float(np.mean(1 - r["J"] / r["J_sched"]))

    loo, solo = {}, {}
    for i in range(N):
        def drop_i(e, a, i=i):
            a = a.clone()
            a[:, i] = A_SCHED
            return a

        def only_i(e, a, i=i):
            b = torch.full_like(a, A_SCHED)
            b[:, i] = a[:, i]
            return b

        nm = GROUPS[env.P.groups[i]][0] if i < G else "координатор"
        loo[nm] = g_full - with_override(drop_i)
        solo[nm] = with_override(only_i)
    tr = full["trace"]
    n = tr["n"]                                                    # (B, 24, G)
    bi = torch.as_tensor(days)
    s = env.sched[bi].numpy()
    rtt = env.rtt.numpy()
    dev = (n - s) * rtt
    realloc = float((np.abs(dev).sum(-1) / 2).sum(-1).mean() / env.budget[bi].numpy().mean())
    rho_s = (env.kappa.numpy() * env.dem[bi].numpy() / np.maximum(s, 1e-3))
    x, y = (n / np.maximum(s, 1e-3) - 1).ravel(), rho_s.ravel()
    ok = s.ravel() > 0
    align = float(np.corrcoef(x[ok], y[ok])[0, 1]) if x[ok].std() > 1e-9 else 0.0
    return {"policy": name, "gain": g_full, "synergy": g_full - sum(solo.values()), "realloc": realloc,
            "align": align, **{f"loo|{k}": v for k, v in loo.items()}, **{f"solo|{k}": v for k, v in solo.items()}}


# =============================================================================
# Стресс-тест (PROTOCOL.md разд. 9)
# =============================================================================
STRESS = {
    "S1_surge": "непредвиденный всплеск: +40% спроса на одной группе 17–21 ч (прогноз не знает)",
    "S1b_surge_known": "то же, но всплеск есть в прогнозе (плановое событие)",
    "S2_storm": "непогода: −40% спроса весь день (прогноз не знает)",
    "S3_fleet": "нехватка подвижного состава: парк −15%, бюджет поездо-часов −10%",
    "S4_fc_bias": "сбой прогноза: прогноз занижен на 25%",
    "S5_calib": "ошибка калибровки: реальное наполнение на 25% выше принятого",
}


def stress(data: EnvData, P: EnvParams, name: str, days, seed=0):
    """Возвращает (EnvData, EnvParams) со стрессом на сутках days. Обучение не повторяется."""
    rng = np.random.default_rng(seed)
    d = replace(data, dem=data.dem.copy(), fc={k: v.copy() for k, v in data.fc.items()})
    P2 = P
    G = data.G
    if name.startswith("S1"):
        gs = rng.integers(0, G, len(days))
        for di, g in zip(days, gs):
            d.dem[di, 17:21, g] *= 1.4
            if name == "S1b_surge_known":
                for k, fc in d.fc.items():
                    for h in range(H):
                        lo, hi = max(17 - h, 0), max(21 - h, 0)
                        if hi > 0:
                            fc[di, h, lo:hi, g] *= 1.4
    elif name == "S2_storm":
        d.dem[days] *= 0.6
    elif name == "S3_fleet":
        P2 = replace(P, fleet_scale=0.85, budget_scale=0.90)
    elif name == "S4_fc_bias":
        for k in d.fc:
            d.fc[k][days] *= 0.75
    elif name == "S5_calib":
        P2 = replace(P, kappa_scale=1.25)
    else:
        raise KeyError(name)
    return d, P2


# =============================================================================
# Обучение одного прогона и пакетный запуск
# =============================================================================
def results_dir() -> Path:
    p = Path("/kaggle/working/results") if Path("/kaggle/working").exists() else Path("results")
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_job(job: dict) -> dict:
    """job: algo, seed, budget, data_path, P (dict), cfg (dict), tag, parts, out. Пишет чекпойнт, кривые и суточные метрики."""
    import marl_algos as ma
    torch.set_num_threads(1)
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    tag = job["tag"]
    done_flag = out / f"{tag}.done.json"
    if done_flag.exists() and not job.get("force"):
        return json.loads(done_flag.read_text())
    t0 = time.perf_counter()
    try:
        data = EnvData.load(job["data_path"])
        P = EnvParams(**job.get("P", {}))
        env = MetroEnv(data, P)
        venv = MetroEnv(data, P)
        va = data.idx("val")
        learner = ma.make(job["algo"], env, job["seed"], **job.get("cfg", {}))
        eval_fn = lambda pol: float(rollout(venv, pol, va)["reward"].mean())
        logs = []
        learner.train(env, data.idx("train"), job["budget"], eval_fn, n_evals=job.get("n_evals", 12),
                      log=logs.append, time_limit=job.get("time_limit"))
        torch.save({"state": learner.state_dict(), "cfg": learner.cfg, "P": P.__dict__, "algo": job["algo"],
                    "seed": job["seed"]}, out / f"{tag}.pt")
        hist = pd.DataFrame(learner.history).assign(tag=tag, algo=job["algo"], seed=job["seed"])
        hist.to_csv(out / f"{tag}.curve.csv", index=False)
        frames = []
        for part in job.get("parts", ["val", "test", "holdout"]):
            days = data.idx(part)
            if len(days):
                frames.append(evaluate(learner, venv, days, job.get("name", job["algo"]), job["seed"], part))
        daily = pd.concat(frames, ignore_index=True)
        daily["tag"] = tag
        daily.to_parquet(out / f"{tag}.daily.parquet", index=False)
        res = {"tag": tag, "algo": job["algo"], "name": job.get("name", job["algo"]), "seed": job["seed"],
               "budget": job["budget"], "steps": learner.steps, "best_step": learner.best_step,
               "best_val_reward": learner.best_val, "train_sec": learner.train_sec,
               "total_sec": time.perf_counter() - t0, "n_params": learner.n_params(),
               "cfg": json.dumps(learner.cfg, ensure_ascii=False), "P": P.tag(),
               "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "ok": True}
    except Exception as e:  # noqa: BLE001
        res = {"tag": tag, "algo": job["algo"], "seed": job["seed"], "ok": False, "error": repr(e),
               "trace": traceback.format_exc()[-2000:]}
    done_flag.write_text(json.dumps(res, ensure_ascii=False, default=str))
    return res


def run_many(jobs: list, workers: int = None, log=print) -> pd.DataFrame:
    """Параллельный запуск (процессы, по 1 потоку torch на процесс). Журнал — runs.csv."""
    import multiprocessing as mp
    workers = workers or max(1, (os.cpu_count() or 2))
    t0 = time.perf_counter()
    res = []
    if workers == 1:
        for j in jobs:
            r = run_job(j)
            res.append(r)
            log(f"  {r['tag']}: {'ок' if r.get('ok') else 'ОШИБКА ' + r.get('error', '')} "
                f"{r.get('train_sec', 0):.0f} c")
    else:
        ctx = mp.get_context("fork" if os.name != "nt" else "spawn")
        with ctx.Pool(workers, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(run_job, jobs):
                res.append(r)
                log(f"  [{len(res)}/{len(jobs)} {time.perf_counter() - t0:.0f} c] {r['tag']}: "
                    f"{'ок' if r.get('ok') else 'ОШИБКА ' + r.get('error', '')} {r.get('train_sec', 0):.0f} c")
    df = pd.DataFrame(res)
    f = results_dir() / "runs.csv"
    df.to_csv(f, mode="a", header=not f.exists(), index=False)
    return df


def find_all(pattern: str, roots=("/kaggle/input", "/kaggle/working/results", "results")) -> list[Path]:
    """Все файлы по шаблону во всех подключённых выводах ноутбуков (и локально). Дубликаты по имени — первый."""
    seen, out = set(), []
    for r in roots:
        if Path(r).exists():
            for p in sorted(Path(r).rglob(pattern)):
                if p.name not in seen:
                    seen.add(p.name)
                    out.append(p)
    return out


def load_runs(pattern="*.daily.parquet", roots=("/kaggle/input", "/kaggle/working/results", "results")) -> pd.DataFrame:
    fs = find_all(pattern, roots)
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_policy(ckpt: Path, env: MetroEnv):
    import marl_algos as ma
    ck = torch.load(ckpt, weights_only=False)
    learner = ma.make(ck["algo"], env, ck["seed"], **{k: v for k, v in ck["cfg"].items()})
    learner.load_state_dict(ck["state"])
    return learner


def baseline_frames(data: EnvData, P: EnvParams, parts=("val", "test", "holdout"), mpc=("m4w", "norm", "hyb", "oracle")):
    """Расписание и MPC-планировщики (детерминированы, seed не нужен)."""
    env = MetroEnv(data, P)
    frames = []
    pols = [SchedulePolicy()] + [MPCPolicy(s) for s in mpc if s == "oracle" or s in data.fc]
    for part in parts:
        days = data.idx(part)
        if not len(days):
            continue
        for pol in pols:
            if pol.name == "mpc_hyb":
                ok = ~np.isnan(data.fc["hyb"][days, 0, 0, 0])
                dd = days[ok]
                if not len(dd):
                    continue
                frames.append(evaluate(pol, env, dd, pol.name, 0, part))
            else:
                frames.append(evaluate(pol, env, days, pol.name, 0, part))
    return pd.concat(frames, ignore_index=True)
