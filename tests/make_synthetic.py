"""Синтетические сырые данные той же формы, что датасет ВКР + расписания MTA — только для отладки кода.

Цифры на синтетике ничего не говорят о метро; они проверяют, что пайплайн и алгоритмы работают.
python tests/make_synthetic.py --out tests/synth  →  tests/synth/env_data.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from marl_env import GROUPS  # noqa: E402
import marl_data as md  # noqa: E402

ROUTES = [r for _, rs in GROUPS for r in rs]


def make(n_st=150, start="2023-01-01", end="2026-10-31", seed=0):
    rng = np.random.default_rng(seed)
    grid = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    days = pd.DatetimeIndex(grid[::24])
    D = len(days)
    # станции и маршруты
    names, lines = [], []
    for s in range(n_st):
        g = rng.integers(len(GROUPS))
        rs = list(rng.choice(GROUPS[g][1], size=rng.integers(1, len(GROUPS[g][1]) + 1), replace=False))
        if rng.random() < 0.15:                     # пересадочный узел
            g2 = rng.integers(len(GROUPS))
            rs += [rng.choice(GROUPS[g2][1])]
        if rng.random() < 0.02:
            rs = ["S"]
        names.append(f"Station {s} ({','.join(sorted(set(rs)))})")
        lines.append(sorted(set(rs)))
    static = pd.DataFrame({"station_complex_id": [str(1000 + i) for i in range(n_st)], "station_complex": names})
    static = static.set_index("station_complex_id")
    # календарь
    hol_days = set()
    for yy in range(2023, 2027):
        for md_ in ["01-01", "05-27", "07-04", "09-02", "11-28", "12-25"]:
            hol_days.add(pd.Timestamp(f"{yy}-{md_}"))
    hol = np.array([d in hol_days for d in days], dtype=float)
    dow = days.dayofweek.to_numpy()
    school = ((dow < 5) & (hol == 0) & ~days.month.isin([7, 8])).astype(float)
    event = (rng.random(D) < 0.05).astype(float)
    # спрос
    h = np.arange(24)
    wkd = 0.2 + 2.6 * np.exp(-0.5 * ((h - 8) / 1.2) ** 2) + 2.2 * np.exp(-0.5 * ((h - 17.5) / 1.6) ** 2) \
        + 0.9 * ((h >= 6) & (h <= 21)) - 0.15 * (h < 5)
    wke = 0.15 + 1.2 * np.exp(-0.5 * ((h - 14) / 4.0) ** 2) - 0.1 * (h < 6)
    wkd, wke = np.clip(wkd, 0.05, None), np.clip(wke, 0.05, None)
    scale = rng.lognormal(5.0, 0.9, n_st)
    commute = rng.uniform(-0.4, 0.4, n_st)
    trend = 1 + 0.08 * np.linspace(0, 1, D)
    season = 1 - 0.07 * np.cos(2 * np.pi * (days.dayofyear.to_numpy() - 200) / 365)
    shock = np.exp(rng.normal(0, 0.06, D)) * np.where(rng.random(D) < 0.01, 0.5, 1.0)
    prof = np.where((dow >= 5) | (hol > 0), 0, 1)[:, None] * wkd[None] + \
        np.where((dow >= 5) | (hol > 0), 1, 0)[:, None] * wke[None]                     # (D, 24)
    base = (prof * trend[:, None] * season[:, None] * shock[:, None])                  # (D, 24)
    tilt = 1 + commute[None, None, :] * np.where(h < 12, 1, -1)[None, :, None]
    lam = base[..., None] * scale[None, None, :] * tilt * (1 + 2.5 * event[:, None, None] *
                                                            ((h >= 18) & (h <= 21))[None, :, None] *
                                                            (rng.random(n_st) < 0.03)[None, None, :])
    Y = rng.poisson(lam).astype("float32").reshape(D * 24, n_st)
    y = pd.DataFrame(Y, index=grid, columns=static.index)
    cov = pd.DataFrame({"is_holiday": np.repeat(hol, 24), "is_school_day": np.repeat(school, 24),
                        "is_city_event": np.repeat(event, 24)}, index=grid)
    cov = cov[cov.index <= pd.Timestamp("2026-09-24 23:00")]       # ковариаты ВКР кончаются 24.09.2026
    # расписание: отправления в час на направление
    rows = []
    sched_wkd = np.clip(np.round(3 + 7 * np.exp(-0.5 * ((h - 8) / 1.5) ** 2) + 6 * np.exp(-0.5 * ((h - 17.5) / 1.8) ** 2)
                                 + 4 * ((h >= 6) & (h <= 21))), 2, None)
    sched_wke = np.clip(np.round(3 + 5 * ((h >= 8) & (h <= 20))), 2, None)
    for di, d in enumerate(days):
        if d > pd.Timestamp("2026-12-31"):
            continue
        we = (dow[di] >= 5) or hol[di] > 0
        prof_s = sched_wke if we else sched_wkd
        for r in ROUTES:
            mult = 1.0
            if r in ("B", "W", "Z") and (h < 6).any():
                pass
            work = we and rng.random() < 0.04                        # плановые работы в выходные
            for hh in range(24):
                n = prof_s[hh] * (0.5 if work and 0 <= hh <= 10 else 1.0) * mult
                if r in ("B", "W", "Z") and (hh < 6 or hh >= 22):
                    n = 0
                if n > 0:
                    for dirc in "NS":
                        rows.append((d + pd.Timedelta(hours=hh), r, dirc, "0", int(n)))
    trips = pd.DataFrame(rows, columns=["ts", "line", "direction", "trip_type", "n"])
    dur = []
    for r in ROUTES:
        run = rng.uniform(0.6, 1.4)
        for k in range(30):
            dur.append((r, "N", f"{r}{k}", run * rng.uniform(0.9, 1.1)))
    durations = pd.DataFrame(dur, columns=["line", "direction", "train_id", "run_h"])
    return y, cov, static, trips, durations


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tests/synth")
    ap.add_argument("--stations", type=int, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    y, cov, static, trips, durations = make(a.stations)
    # на синтетике «гибрид ВКР» — шумный оракул по станциям для val/test (проверка агрегирования)
    rng = np.random.default_rng(1)
    hyb = {}
    origins = y.index[(y.index >= "2026-04-01") & (y.index <= "2026-09-24 23:00")][::1]
    Yv = y.to_numpy()
    pos = {t: i for i, t in enumerate(y.index)}
    t0 = np.array([pos[t] for t in origins])
    yh = np.stack([Yv[np.clip(t0 + k, 0, len(Yv) - 1)] for k in range(24)], 1) * rng.lognormal(0, 0.08, (len(t0), 24, 1))
    hyb["hyb"] = (origins, yh.astype("float32"))
    data, pas = md.build_env_data(y, cov, static, trips, durations, hyb=hyb)
    data.save(out / "env_data.npz")
    import json
    print(json.dumps({k: v for k, v in pas.items() if k != "norm_coef"}, ensure_ascii=False, indent=1, default=str)[:4000])
