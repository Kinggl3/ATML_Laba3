"""
marl_data — данные ЛР3: из датасета ВКР и расписаний MTA собирает EnvData для среды (marl_env).

Источники
1. Датасет ВКР `VKR_PM232_Pevin_MTA` (Kaggle): почасовые входы по 424 станционным комплексам, ковариаты,
   статические признаки станций (линии в названии комплекса).
2. MTA Subway Schedules 2023–2026 (data.ny.gov): расписание с поправками на плановые работы — число отправлений
   поездов с начальной станции по маршруту, направлению и часу; время рейса (для времени оборота).
3. MTA Subway Hourly Ridership 2025+ (data.ny.gov `5wq4-mkjj`): свежий отложенный период после конца датасета ВКР.
4. (необязательно) прогнозы гибридной модели ВКР `hyb_blend__{val,test}.npz` — вывод ноутбука 16 ВКР.

Всё время — местное время Нью-Йорка без часового пояса, как в ВКР.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from marl_env import GROUPS, ROUTE2GROUP, ROUTE_MAP, NOT_CONTROLLED, EnvData, EnvParams, H

# --- разбиение ЛР3 (PROTOCOL.md, разд. 2) — совпадает с ВКР; holdout — свежие данные после конца датасета ВКР
SPLIT = {
    "train": ("2023-01-30", "2026-03-31"),     # первые 4 недели — только история для прогноза «норма»
    "val": ("2026-04-01", "2026-06-30"),
    "test": ("2026-07-01", "2026-09-24"),
    "holdout": ("2026-09-25", "2026-12-31"),   # фактический конец — последний полный день в выгрузке MTA
}
EPS_G = 100.0            # сглаживание в логарифмах отношений (спрос группы — тысячи входов в час)
LAYOVER_H = 8 / 60       # стоянка на конечной, ч (в обе стороны оборота)
RTT_DEFAULT_H = 2.3      # время оборота, если по расписанию оценить не удалось
NY = "https://data.ny.gov"
SCHED_DS = {2023: "7pnn-mafy", 2024: "ebrw-j62c", 2025: "q9nv-uegs", 2026: "g8es-h7gb"}
RIDERSHIP_DS = "5wq4-mkjj"


# =============================================================================
# Поиск файлов на Kaggle
# =============================================================================
def walk(name: str, root: str = "/kaggle/input") -> list[Path]:
    out = []
    if os.path.isdir(root):
        for dp, _, fn in os.walk(root, followlinks=True):
            if name in fn:
                out.append(Path(dp) / name)
    return sorted(out)


def find_vkr_root() -> Path:
    env = os.environ.get("VKR_DATA_ROOT")
    if env:
        return Path(env)
    hits = [p for p in walk("split.json") if p.parent.name == "lab"]
    if hits:
        return hits[0].parent.parent
    for cand in [Path("dataset_kaggle"), Path("../dataset_kaggle"), Path("data"), Path("../data")]:
        if (cand / "lab" / "split.json").exists():
            return cand
    raise FileNotFoundError("Не найден lab/split.json: подключите датасет VKR_PM232_Pevin_MTA или задайте VKR_DATA_ROOT")


def find_file(name: str) -> Path | None:
    hits = walk(name)
    if hits:
        return hits[0]
    for cand in [Path("results"), Path("../results"), Path("/kaggle/working/results")]:
        for p in cand.glob(f"**/{name}"):
            return p
    return None


def station_lines(name: str) -> set[str]:
    """Маршруты из скобок названия комплекса: 'Hewes St (M,J)' → {'M','J'}. Как в ВКР и ЛР2."""
    groups = re.findall(r"\(([^)]*)\)", str(name))
    out = set()
    for g in groups:
        for r in g.split(","):
            r = ROUTE_MAP.get(r.strip(), r.strip())
            if re.fullmatch(r"[0-9A-Z]{1,2}", r) and r not in NOT_CONTROLLED:
                out.add(r)
    return out


def load_vkr(root: Path):
    lab = root / "lab"
    y = pd.read_parquet(lab / "ridership_wide.parquet")
    y.columns = y.columns.astype(str)
    cov = pd.read_parquet(lab / "covariates_time.parquet")
    static = pd.read_parquet(lab / "station_static.parquet")
    static["station_complex_id"] = static["station_complex_id"].astype(str)
    static = static.set_index("station_complex_id").loc[y.columns]
    return y, cov, static


# =============================================================================
# data.ny.gov (Socrata)
# =============================================================================
class SocrataError(RuntimeError):
    pass


def _get(url, params, tries=6, timeout=180):
    """GET с повторами. Ошибки 4xx (неверный запрос) не повторяются — сразу исключение с текстом ответа сервера."""
    import requests
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout,
                             headers={"User-Agent": "ATML-Laba3 student project (Kaggle notebook)"})
        except Exception as e:  # noqa: BLE001  (сеть)
            if i == tries - 1:
                raise RuntimeError(f"{url}: {e}") from e
            time.sleep(5 * (i + 1))
            continue
        if 400 <= r.status_code < 500 and r.status_code != 429:
            raise SocrataError(f"{r.status_code} {url}: {r.text[:600]}")
        if r.ok:
            return r.json()
        if i == tries - 1:
            raise RuntimeError(f"{r.status_code} {url}: {r.text[:600]}")
        time.sleep(5 * (i + 1))


def socrata(ds, select, where, group=None, order=None, page=50000):
    rows, off = [], 0
    while True:
        p = {"$select": select, "$where": where, "$limit": page, "$offset": off}
        if group:
            p["$group"] = group
        if order:
            p["$order"] = order
        part = _get(f"{NY}/resource/{ds}.json", p)
        rows += part
        if len(part) < page:
            return pd.DataFrame(rows)
        off += page


TRIP_QUERIES = [
    # (имя, как фильтровать месяц, select, group). Порядок: от быстрой серверной агрегации к сырым строкам.
    ("agg_departure", "departure",
     "date_trunc_ymdh(departure_time) AS ts, line, direction, trip_type, count(*) AS n",
     "date_trunc_ymdh(departure_time), line, direction, trip_type"),
    ("agg_service_date", "service",
     "date_trunc_ymdh(departure_time) AS ts, line, direction, trip_type, count(*) AS n",
     "date_trunc_ymdh(departure_time), line, direction, trip_type"),
    ("agg_no_type", "service",
     "date_trunc_ymdh(departure_time) AS ts, line, direction, count(*) AS n",
     "date_trunc_ymdh(departure_time), line, direction"),
    ("raw_rows", "service", "departure_time, line, direction, trip_type", None),
]


def _trip_where(kind, lo, hi):
    col = "departure_time" if kind == "departure" else "service_date"
    return f"stop_order = 1 AND {col} >= '{lo:%Y-%m-%dT%H:%M:%S}' AND {col} < '{hi:%Y-%m-%dT%H:%M:%S}'"


def _trips_frame(df, raw):
    if not len(df):
        return df
    if raw:                                                     # сырые строки → агрегируем сами
        df["ts"] = pd.to_datetime(df["departure_time"]).dt.floor("h")
        if "trip_type" not in df:
            df["trip_type"] = "NA"
        df = df.groupby(["ts", "line", "direction", "trip_type"], dropna=False).size().rename("n").reset_index()
    if "trip_type" not in df:
        df["trip_type"] = "NA"
    return df


def pick_trip_query(log=print):
    """Пробует формы запроса на одном дне (лимит 5 строк), возвращает первую, которую принимает сервер."""
    ds = SCHED_DS[2024]
    lo, hi = pd.Timestamp("2024-03-06"), pd.Timestamp("2024-03-07")
    for name, kind, sel, grp in TRIP_QUERIES:
        p = {"$select": sel, "$where": _trip_where(kind, lo, hi), "$limit": 5}
        if grp:
            p["$group"] = grp
        try:
            rows = _get(f"{NY}/resource/{ds}.json", p)
            log(f"  форма запроса расписания «{name}»: принята, пример {rows[:1]}")
            if rows:
                return name, kind, sel, grp
        except SocrataError as e:
            log(f"  форма запроса расписания «{name}»: отклонена — {str(e)[:300]}")
    raise RuntimeError("сервер не принял ни одну форму запроса расписания — см. сообщения выше")


def download_trips(start: pd.Timestamp, end: pd.Timestamp, log=print) -> pd.DataFrame:
    """Отправления с начальной станции (stop_order = 1): час × маршрут × направление × тип рейса → число рейсов.

    Наборы разбиты по году даты обслуживания (service_date). При фильтре по service_date каждый рейс попадает
    ровно в один набор и один месяц. При фильтре по времени отправления рейс после полуночи лежит в наборе
    прошлого года, поэтому декабрь запрашивается до 06:00 1 января следующего года. Результаты суммируются."""
    name, kind, sel, grp = pick_trip_query(log)
    raw = grp is None
    parts = []
    for year, ds in SCHED_DS.items():
        a = max(start, pd.Timestamp(f"{year}-01-01"))
        b = min(end, pd.Timestamp(f"{year + 1}-01-01 06:00") if kind == "departure" else pd.Timestamp(f"{year + 1}-01-01"))
        if a >= b:
            continue
        for ms in pd.date_range(a.normalize().replace(day=1), b, freq="MS"):
            lo = max(ms, a)
            hi = min(ms + pd.offsets.MonthBegin(1), b)
            if ms.month == 12 and kind == "departure":
                hi = min(pd.Timestamp(f"{year + 1}-01-01 06:00"), b)
            if lo >= hi:
                continue
            df = _trips_frame(socrata(ds, sel, _trip_where(kind, lo, hi), group=grp), raw)
            if len(df):
                parts.append(df)
            log(f"  расписание {year} {lo:%Y-%m-%d}…{hi:%Y-%m-%d}: {len(df)} строк")
    t = pd.concat(parts, ignore_index=True)
    t["ts"] = pd.to_datetime(t["ts"])
    t["n"] = pd.to_numeric(t["n"])
    t["trip_type"] = t["trip_type"].fillna("NA").astype(str)
    t = t.groupby(["ts", "line", "direction", "trip_type"], as_index=False)["n"].sum()
    t.attrs["query"] = name
    return t


def check_trip_duplicates(dates, log=print) -> pd.DataFrame:
    """Проверка: один ли рейс = одна строка с stop_order = 1 (нет ли в наборе базового и изменённого рейса сразу)."""
    out = []
    for d in dates:
        d = pd.Timestamp(d)
        ds = SCHED_DS.get(d.year)
        if not ds:
            continue
        w = f"stop_order = 1 AND service_date = '{d:%Y-%m-%dT00:00:00}'"
        df = socrata(ds, "train_id, line, direction, departure_time, supplement_schedule_number", w)
        if not len(df):
            continue
        n_rows = len(df)
        n_uniq = df.drop_duplicates(["train_id", "line", "direction", "departure_time"]).shape[0]
        out.append({"date": d.date(), "rows": n_rows, "unique_trips": n_uniq, "dup_share": 1 - n_uniq / n_rows,
                    "supplements": df["supplement_schedule_number"].nunique()})
        log(f"  {d.date()}: строк {n_rows}, уникальных рейсов {n_uniq}")
    return pd.DataFrame(out)


def download_durations(dates, log=print) -> pd.DataFrame:
    """Время рейса от первой до последней остановки (по выбранным датам) — для времени оборота."""
    parts = []
    for d in dates:
        d = pd.Timestamp(d)
        ds = SCHED_DS.get(d.year)
        if not ds:
            continue
        w = f"service_date = '{d:%Y-%m-%dT00:00:00}'"
        df = socrata(ds, "line, direction, train_id, min(departure_time) AS t0, max(arrival_time) AS t1", w,
                     group="line, direction, train_id")
        if len(df):
            df["date"] = d
            parts.append(df)
        log(f"  рейсы {d.date()}: {len(df)}")
    t = pd.concat(parts, ignore_index=True)
    t["t0"], t["t1"] = pd.to_datetime(t["t0"]), pd.to_datetime(t["t1"])
    t["run_h"] = (t["t1"] - t["t0"]).dt.total_seconds() / 3600
    return t[(t["run_h"] > 0.05) & (t["run_h"] < 4)]


def download_ridership(start: pd.Timestamp, end: pd.Timestamp, stations: list[str], log=print) -> pd.DataFrame:
    """Почасовые входы по комплексам (как download_mta.py ВКР) → широкая таблица в порядке станций ВКР."""
    parts = []
    for a in pd.date_range(start, end, freq="7D"):
        b = min(a + pd.Timedelta(days=7), end)
        w = (f"transit_mode = 'subway' AND transit_timestamp >= '{a:%Y-%m-%dT%H:%M:%S}' "
             f"AND transit_timestamp < '{b:%Y-%m-%dT%H:%M:%S}'")
        df = socrata(RIDERSHIP_DS, "transit_timestamp, station_complex_id, sum(ridership) AS ridership", w,
                     group="transit_timestamp, station_complex_id")
        log(f"  входы {a:%Y-%m-%d}…{b:%Y-%m-%d}: {len(df)} строк")
        if len(df):
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=stations)
    t = pd.concat(parts, ignore_index=True)
    t["transit_timestamp"] = pd.to_datetime(t["transit_timestamp"])
    t["ridership"] = pd.to_numeric(t["ridership"])
    w = t.pivot_table(index="transit_timestamp", columns="station_complex_id", values="ridership", aggfunc="sum")
    w.columns = w.columns.astype(str)
    return w.reindex(columns=stations).fillna(0.0)


# =============================================================================
# Сборка массивов среды
# =============================================================================
def full_grid(y: pd.DataFrame, log=print) -> tuple[pd.DataFrame, int]:
    """Полная сетка часов (весенний перевод часов — 02:00 нет; осенний — сдвоенный 01:00 уже сложен MTA)."""
    idx = pd.date_range(y.index.min().normalize(), y.index.max().normalize() + pd.Timedelta(hours=23), freq="h")
    miss = int((~idx.isin(y.index)).sum())
    y = y.reindex(idx).interpolate(limit=6, limit_direction="both").fillna(0.0).clip(lower=0)
    return y, miss


def incidence(lines: list[set]) -> np.ndarray:
    """(S, G): 1, если на станции останавливается маршрут группы."""
    A = np.zeros((len(lines), len(GROUPS)), dtype="float32")
    for s, ls in enumerate(lines):
        for r in ls:
            if r in ROUTE2GROUP:
                A[s, ROUTE2GROUP[r]] = 1.0
    return A


def schedule_hourly(trips: pd.DataFrame, index: pd.DatetimeIndex, log=print):
    """Поездов в час по группам (среднее двух направлений) и число работающих линий; + журнал маршрутов."""
    t = trips.copy()
    t["route"] = t["line"].astype(str).str.strip().str.upper().map(lambda r: ROUTE_MAP.get(r, r))
    unknown = sorted(set(t["route"]) - set(ROUTE2GROUP) - NOT_CONTROLLED)
    t["g"] = t["route"].map(ROUTE2GROUP)
    t = t.dropna(subset=["g"])
    t["g"] = t["g"].astype(int)
    tt = t.groupby(["ts", "g", "route"], as_index=False)["n"].sum()
    tph = tt.groupby(["ts", "g"])["n"].sum().unstack(fill_value=0.0) / 2.0                  # два направления
    nl = (tt[tt["n"] > 0].groupby(["ts", "g"])["route"].nunique().unstack(fill_value=0))
    S = tph.reindex(index=index, columns=range(len(GROUPS)), fill_value=0.0).to_numpy("float32")
    L = nl.reindex(index=index, columns=range(len(GROUPS)), fill_value=0).to_numpy("float32")
    return S, L, unknown


def day_type(days: pd.DatetimeIndex, holiday: np.ndarray) -> np.ndarray:
    """0 — будни, 1 — суббота, 2 — воскресенье или праздник."""
    dt = np.where(days.dayofweek == 5, 1, np.where(days.dayofweek == 6, 2, 0))
    return np.where(holiday > 0, 2, dt)


def fill_schedule_gaps(S, L, days, dtype, is_train, log=print):
    """Сутки, где расписания нет совсем (вне наборов или сбой выгрузки), — типичное расписание по типу дня из train."""
    D = len(days)
    S3, L3 = S.reshape(D, H, -1).copy(), L.reshape(D, H, -1).copy()
    empty = S3.sum(axis=(1, 2)) < 1
    typ_S = np.stack([np.median(S3[(dtype == k) & is_train & ~empty], axis=0) for k in range(3)])
    typ_L = np.stack([np.median(L3[(dtype == k) & is_train & ~empty], axis=0) for k in range(3)])
    S3[empty] = typ_S[dtype[empty]]
    L3[empty] = typ_L[dtype[empty]]
    return S3, L3, empty, typ_S


def group_demand(Y: np.ndarray, A: np.ndarray, S: np.ndarray):
    """Входы станции делятся между группами её маршрутов пропорционально поездам в час (расписание известно заранее).
    Станция, где в этот час не работает ни одна группа, выпадает (челноки, автобусы на время работ) — доля в журнал."""
    den = S @ A.T                                                   # (T, S)
    z = np.where(den > 0, Y / np.maximum(den, 1e-9), 0.0)
    d = S * (z @ A)                                                 # (T, G)
    lost = float(Y[den <= 0].sum() / max(Y.sum(), 1.0))
    no_group = float(Y[:, A.sum(1) == 0].sum() / max(Y.sum(), 1.0))
    return d.astype("float32"), lost, no_group


def weekly_norm(y: np.ndarray) -> np.ndarray:
    """(T, G) медиана того же часа недели за 4 прошлые недели (как norm в ВКР)."""
    T = len(y)
    st = np.full((4, T) + y.shape[1:], np.nan, dtype="float32")
    for k in range(1, 5):
        st[k - 1, 168 * k:] = y[:-168 * k]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(st, axis=0)


def level_feats(y, nm, t):
    out = []
    for w in (3, 24):
        idx = t[:, None] - np.arange(1, w + 1)[None, :]
        ok = (idx >= 0).all(1)
        idx = np.clip(idx, 0, None)
        ys, ns = y[idx].sum(1), np.nan_to_num(nm[idx]).sum(1)
        r = np.log((ys + EPS_G * w) / (ns + EPS_G * w))
        out.append(np.where(ok[:, None], r, 0.0))
    return np.stack(out, -1)                                          # (O, G, 2)


def build_forecasts(y: np.ndarray, D: int, train_end_t: int, start_t: int = 24 * 30):
    """Прогнозы на группу линий из каждой точки (сутки d, час h) на часы h…23: m4w и norm_level.

    m4w — медиана 4 недель; norm — m4w × поправка уровня по последним 3 и 24 ч, коэффициенты на каждый
    горизонт k = 0…23 по train (как norm_level в ВКР, src/baselines.py). Только часы < точки прогноза."""
    nm = weekly_norm(y)
    G = y.shape[1]
    # обучение коэффициентов: точки прогноза в train, каждые 5 часов
    t_fit = np.arange(start_t, train_end_t - H, 5)
    X = level_feats(y, nm, t_fit)
    coef = np.zeros((H, 3))
    for k in range(H):
        tt = t_fit + k
        ok = ~np.isnan(nm[tt]) & (nm[tt] > 5 * EPS_G / 100)
        tgt = np.log((y[tt] + EPS_G) / (np.nan_to_num(nm[tt]) + EPS_G))
        A_ = np.c_[np.ones(ok.sum()), X[ok]]
        coef[k] = np.linalg.lstsq(A_, tgt[ok], rcond=None)[0]
    fc_m4w = np.full((D, H, H, G), np.nan, dtype="float32")
    fc_norm = np.full((D, H, H, G), np.nan, dtype="float32")
    t_all = np.arange(D * H)
    Xa = level_feats(y, nm, t_all).reshape(D, H, G, 2)
    nm3 = nm.reshape(D, H, G)
    for h in range(H):
        for k in range(H - h):
            base = nm3[:, h + k]                                           # (D, G)
            adj = coef[k, 0] + Xa[:, h] @ coef[k, 1:]
            fc_m4w[:, h, k] = base
            fc_norm[:, h, k] = np.clip((base + EPS_G) * np.exp(adj) - EPS_G, 0, None)
    return {"m4w": fc_m4w, "norm": fc_norm}, coef


def aggregate_station_forecast(origins: pd.DatetimeIndex, yhat: np.ndarray, A: np.ndarray, S_hourly: np.ndarray,
                               grid: pd.DatetimeIndex, day0: pd.Timestamp, D: int) -> np.ndarray:
    """Прогноз ВКР по станциям (O, 24, N) → по группам линий в формате (D, 24, 24, G)."""
    pos = pd.Series(np.arange(len(grid)), index=grid)
    out = np.full((D, H, H, A.shape[1]), np.nan, dtype="float32")
    ok = origins.isin(grid)
    origins, yhat = origins[ok], yhat[ok]
    t0 = pos[origins].to_numpy()
    for k in range(H):
        tk = np.clip(t0 + k, 0, len(grid) - 1)
        s = S_hourly[tk]                                         # (O, G)
        den = s @ A.T
        z = np.where(den > 0, yhat[:, k] / np.maximum(den, 1e-9), 0.0)
        dg = s * (z @ A)
        di = (origins.normalize() - day0).days.to_numpy()
        hh = origins.hour.to_numpy()
        keep = (hh + k < H) & (di >= 0) & (di < D)
        out[di[keep], hh[keep], k] = dg[keep]
    return out


def calendar(days: pd.DatetimeIndex, holiday: np.ndarray, school: np.ndarray) -> np.ndarray:
    D = len(days)
    hrs = np.arange(H)
    dow = days.dayofweek.to_numpy()
    cal = np.zeros((D, H, 7), dtype="float32")
    cal[..., 0] = np.sin(2 * np.pi * hrs / H)[None]
    cal[..., 1] = np.cos(2 * np.pi * hrs / H)[None]
    cal[..., 2] = np.sin(2 * np.pi * dow / 7)[:, None]
    cal[..., 3] = np.cos(2 * np.pi * dow / 7)[:, None]
    cal[..., 4] = (dow >= 5)[:, None]
    cal[..., 5] = holiday[:, None]
    cal[..., 6] = school[:, None]
    return cal


def us_holidays_fallback(days: pd.DatetimeIndex) -> np.ndarray:
    """Праздники для дат вне ковариат ВКР (отложенный период): федеральные + выборы (школы закрыты)."""
    try:
        import holidays
        hs = holidays.country_holidays("US", subdiv="NY", years=sorted(set(days.year)))
        return np.array([d.date() in hs for d in days], dtype="float32")
    except Exception:  # noqa: BLE001
        fixed = {"2026-10-12", "2026-11-03", "2026-11-11", "2026-11-26", "2026-12-25"}
        return np.array([d.strftime("%Y-%m-%d") in fixed for d in days], dtype="float32")


def build_env_data(y: pd.DataFrame, cov: pd.DataFrame, static: pd.DataFrame, trips: pd.DataFrame,
                   durations: pd.DataFrame | None, hyb: dict | None = None, P: EnvParams | None = None,
                   split: dict | None = None, log=print) -> tuple[EnvData, dict]:
    """Главная функция подготовки. Возвращает EnvData и паспорт (словарь с цифрами для отчёта)."""
    P = P or EnvParams()
    split = split or SPLIT
    pas: dict = {}
    y, miss = full_grid(y, log)
    grid = y.index
    days = pd.DatetimeIndex(grid[::H])
    D = len(days)
    pas["hours_filled"] = miss
    lines = [station_lines(n) for n in static["station_complex"]]
    A = incidence(lines)
    pas["stations"] = len(lines)
    pas["stations_no_group"] = int((A.sum(1) == 0).sum())
    pas["stations_no_group_names"] = list(static.loc[A.sum(1) == 0, "station_complex"].astype(str))[:30]
    # части
    part = np.full(D, "", dtype=object)
    for name, (a, b) in split.items():
        part[(days >= pd.Timestamp(a)) & (days <= pd.Timestamp(b))] = name
    is_train = part == "train"
    # календарь
    covd = cov.reindex(grid)
    hol_cov = covd["is_holiday"].to_numpy() if "is_holiday" in covd else np.full(len(grid), np.nan)
    hol = pd.Series(hol_cov).groupby(np.arange(len(grid)) // H).max().to_numpy().astype("float64").copy()
    sch_cov = covd["is_school_day"].to_numpy() if "is_school_day" in covd else np.full(len(grid), np.nan)
    sch = pd.Series(sch_cov).groupby(np.arange(len(grid)) // H).max().to_numpy().astype("float64").copy()
    nocov = np.isnan(hol)
    if nocov.any():
        hol[nocov] = us_holidays_fallback(days[nocov])
        wk = days.dayofweek < 5
        sch[nocov] = (wk & (hol == 0) & ~((days.month == 12) & (days.day >= 24)))[nocov]
    pas["days_without_vkr_covariates"] = int(nocov.sum())
    hol = np.nan_to_num(hol).astype("float32")
    sch = np.nan_to_num(sch).astype("float32")
    dtype = day_type(days, hol)
    # расписание
    S_h, L_h, unknown = schedule_hourly(trips, grid, log)
    S3, L3, empty, typ_S = fill_schedule_gaps(S_h, L_h, days, dtype, is_train, log)
    pas["routes_unknown"] = unknown
    pas["days_schedule_filled"] = int(empty.sum())
    pas["days_schedule_filled_list"] = [str(d.date()) for d in days[empty]][:50]
    S_hourly = S3.reshape(D * H, -1)
    # спрос групп
    Y = y.to_numpy("float32")
    dg, lost, no_group = group_demand(Y, A, S_hourly)
    pas["share_entries_no_group"] = no_group
    pas["share_entries_no_service"] = lost
    dem = dg.reshape(D, H, -1)
    dem_prev = np.concatenate([dem[:1, :2] * 0 + dem[:1, -2:], dem[:-1, -2:]], 0)
    # прогнозы
    t_end = int(np.where(is_train)[0].max() + 1) * H
    fc, coef = build_forecasts(dg, D, t_end)
    if hyb:
        for nm_, (orig, yh) in hyb.items():
            fc[nm_] = aggregate_station_forecast(pd.DatetimeIndex(orig), yh, A, S_hourly, grid, days[0], D)
    # время оборота
    rtt = np.full(len(GROUPS), RTT_DEFAULT_H, dtype="float32")
    rtt_src = ["по умолчанию"] * len(GROUPS)
    if durations is not None and len(durations):
        du = durations.copy()
        du["route"] = du["line"].astype(str).str.upper().map(lambda r: ROUTE_MAP.get(r, r))
        du["g"] = du["route"].map(ROUTE2GROUP)
        du = du.dropna(subset=["g"])
        per_route = du.groupby(["g", "route"]).agg(run=("run_h", "median"), n=("run_h", "size")).reset_index()
        for g, grp in per_route.groupby("g"):
            run = float(np.average(grp["run"], weights=grp["n"]))
            rtt[int(g)] = 2 * (run + LAYOVER_H)
            rtt_src[int(g)] = f"расписание: {len(grp)} маршрутов, {int(grp['n'].sum())} рейсов"
    pas["rtt_h"] = dict(zip([g for g, _ in GROUPS], np.round(rtt, 2).tolist()))
    pas["rtt_source"] = dict(zip([g for g, _ in GROUPS], rtt_src))
    # калибровка наполнения: в типичный будний час пик (train) по расписанию ρ = rho_peak
    wk = is_train & (dtype == 0)
    med_d = np.median(dem[wk], axis=0)                                  # (24, G)
    pk = med_d.argmax(0)
    d_pk = med_d[pk, np.arange(len(GROUPS))]
    n_pk = np.median(S3[wk], axis=0)[pk, np.arange(len(GROUPS))]
    kappa = (P.rho_peak * n_pk / np.maximum(d_pk, 1.0)).astype("float32")
    pas["peak_hour"] = dict(zip([g for g, _ in GROUPS], pk.tolist()))
    pas["peak_demand"] = dict(zip([g for g, _ in GROUPS], np.round(d_pk).tolist()))
    pas["peak_tph"] = dict(zip([g for g, _ in GROUPS], np.round(n_pk, 1).tolist()))
    scale_d = dem[is_train].mean(axis=(0, 1)).astype("float32")
    scale_s = S3[is_train].mean(axis=(0, 1)).astype("float32")
    track_cap = S3.max(axis=(0, 1)).astype("float32")
    insvc = (S3 * rtt).sum(-1)
    fleet_max = float(np.max(insvc[is_train]) * (1 + P.fleet_spare))
    pas["fleet_max_trains"] = round(fleet_max, 1)
    pas["budget_trainh_median_weekday"] = float(np.median(insvc[is_train & (dtype == 0)].sum(-1)))
    # режимы дня
    typ = typ_S[dtype]                                                  # (D, 24, G)
    low = (S3 < 0.8 * typ) & (typ >= 2)
    planned = (low.sum(1) >= 3).any(-1)
    ev = covd["is_city_event"].to_numpy() if "is_city_event" in covd else np.zeros(len(grid))
    ev_day = pd.Series(np.nan_to_num(ev)).groupby(np.arange(len(grid)) // H).max().to_numpy() > 0
    flags = {"holiday": hol > 0, "weekend": days.dayofweek.to_numpy() >= 5, "planned_works": planned,
             "city_event": ev_day, "schedule_filled": empty, "daytype": dtype}
    # качество прогнозов (WAPE, группа × час)
    q = {}
    for nm_, f in fc.items():
        for pp in ["train", "val", "test", "holdout"]:
            m = part == pp
            if not m.any():
                continue
            ff = f[m]
            # горизонт k = 0 (ближайший час) по всем точкам прогноза и все горизонты
            k0 = ff[:, :, 0]
            ok = ~np.isnan(k0)
            if not ok.any():
                continue
            q[f"{nm_}|{pp}|h1"] = float(np.abs(k0[ok] - dem[m][ok]).sum() / max(dem[m][ok].sum(), 1))
            allk = []
            for h in range(H):
                fk = ff[:, h, :H - h]
                tk = dem[m][:, h:]
                ok = ~np.isnan(fk)
                allk.append((np.abs(fk[ok] - tk[ok]).sum(), tk[ok].sum()))
            num = sum(a for a, _ in allk)
            den = sum(b for _, b in allk)
            q[f"{nm_}|{pp}|all"] = float(num / max(den, 1))
    pas["forecast_wape"] = q
    pas["norm_coef"] = coef.round(4).tolist()
    pas["days"] = {pp: int((part == pp).sum()) for pp in split}
    pas["period"] = [str(days[0].date()), str(days[-1].date())]
    data = EnvData(day=days.values.astype("datetime64[D]"), part=part, dem=dem, dem_prev=dem_prev.astype("float32"),
                   sched=S3.astype("float32"), nlines=L3.astype("float32"),
                   cal=calendar(days, hol, sch), fc=fc, rtt=rtt, kappa=kappa, scale_d=scale_d, scale_s=scale_s,
                   track_cap=track_cap, fleet_max=fleet_max, flags=flags,
                   meta={"groups": [g for g, _ in GROUPS], "passport": pas, "params": P.__dict__})
    return data, pas
