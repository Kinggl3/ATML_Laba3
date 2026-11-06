# %% [markdown]
# # ЛР3 · lab3_00 — данные и среда: аудит, расписания MTA, свежий отложенный период, паспорт
#
# Задания 2 и 7 (протокол: разд. 2–4, 12). **Accelerator:** нет. **Internet:** включён. Время: ~40–70 мин.
#
# **Inputs:** датасет `VKR_PM232_Pevin_MTA`; *необязательно* — вывод ноутбука 16 ВКР (`16_hybrid_kaggle`)
# с файлами `hyb_blend__val.npz`, `hyb_blend__test.npz` (прогноз гибрида ВКР для лестницы прогнозов, H6).
#
# **Output:** модули `marl_*.py`, `env_data.npz` (+ `_meta.json`), `trips.parquet`, `durations.parquet`,
# `holdout_ridership.parquet`, паспорт данных `data_passport.md`, рисунки, проверки среды.
#
# Что делает:
# 1. Загружает данные ВКР и скачивает свежие почасовые входы после 24.09.2026 (отложенный период).
# 2. Скачивает расписания MTA с поправками 2023–2026: отправления по часам; проверяет дубликаты рейсов;
#    время рейса по датам-образцам → время оборота.
# 3. Собирает массивы среды (спрос групп линий, расписание, календарь, прогнозы m4w / norm_level / гибрид ВКР,
#    калибровка наполнения, парк, бюджет).
# 4. Паспорт данных: описательная статистика, проблемы качества, схема предобработки и разбиения.
# 5. Проверки среды: расписание даёт награду ровно −24; MPC и оракул на val.

# %%
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
sys.path.insert(0, str(HERE))
import marl_data as md
import marl_env as me
from marl_eval import to_md as me_md
OUT = HERE / "results"
OUT.mkdir(parents=True, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)
t0 = time.time()
root = md.find_vkr_root()
y, cov, static = md.load_vkr(root)
log("датасет ВКР:", root, "| входы", y.shape, "|", y.index.min(), "…", y.index.max())

# %% [markdown]
# ## 1. Свежий отложенный период (holdout): входы после конца датасета ВКР

# %%
vkr_end = y.index.max()
start = (vkr_end + pd.Timedelta(hours=1)).normalize()
end = pd.Timestamp.now().normalize()
try:
    fresh = md.download_ridership(start, end, list(y.columns), log=log)
    fresh = fresh[fresh.index >= start]
    if len(fresh):
        per_day = fresh.groupby(fresh.index.normalize()).size()
        full = per_day[per_day >= 23].index
        last_full = full.max() if len(full) else None
        # день считается полным, если есть ≥ 23 часа и сумма входов не меньше 60% медианы тех же дней недели в 2026
        tot = fresh.sum(axis=1).groupby(fresh.index.normalize()).sum()
        ref = y[y.index >= "2026-06-01"].sum(axis=1).groupby(y[y.index >= "2026-06-01"].index.normalize()).sum()
        ref_dow = ref.groupby(ref.index.dayofweek).median()
        ok_days = [d for d in full if tot[d] >= 0.6 * ref_dow[d.dayofweek]]
        # отложенный период — непрерывный отрезок от начала до первого неполного дня
        cont = []
        for d in pd.date_range(start, end, freq="D"):
            if d in ok_days:
                cont.append(d)
            else:
                break
        fresh = fresh[fresh.index.normalize().isin(cont)]
        log(f"свежие входы: {len(cont)} полных суток подряд, {cont[0].date() if cont else '—'} … "
            f"{cont[-1].date() if cont else '—'}; станций с данными {int((fresh.sum() > 0).sum())}")
    fresh.to_parquet(OUT / "holdout_ridership.parquet")
except Exception as e:  # noqa: BLE001
    fresh = pd.DataFrame(columns=y.columns)
    log("ОШИБКА загрузки свежих входов — holdout будет пустым, записать в отступления:", repr(e))
y_all = pd.concat([y, fresh.astype("float32")]) if len(fresh) else y
holdout_end = y_all.index.max().normalize()

# %% [markdown]
# ## 2. Расписания MTA: отправления по часам, проверка дубликатов, время рейса

# %%
t1 = time.time()
trips = md.download_trips(pd.Timestamp("2022-12-31"), holdout_end + pd.Timedelta(days=1), log=log)
trips.to_parquet(OUT / "trips.parquet", index=False)
log(f"расписание: {len(trips):,} строк, форма запроса «{trips.attrs.get('query', '?')}», {time.time() - t1:.0f} c")
tt = trips.groupby("trip_type")["n"].sum()
log("рейсы по trip_type:", tt.to_dict())
# если обычные рейсы (trip_type = 0) — подавляющее большинство, прочие (служебные) не считаем
if "0" in tt and tt["0"] / tt.sum() > 0.9:
    trips = trips[trips["trip_type"] == "0"]
    log("оставлены рейсы trip_type = 0")
samples = pd.to_datetime(["2023-03-08", "2024-07-13", "2025-10-08", "2026-02-11", "2026-05-16", "2026-08-12"])
try:
    dup = md.check_trip_duplicates(samples, log=log)
except Exception as e:  # noqa: BLE001
    dup = pd.DataFrame(columns=["date", "rows", "unique_trips", "dup_share", "supplements"])
    log("проверка дубликатов рейсов не выполнена (записать в паспорт):", repr(e)[:400])
dup.to_csv(OUT / "trip_duplicates_check.csv", index=False)
if len(dup) and dup["dup_share"].max() > 0.01:
    log("ВНИМАНИЕ: в расписании есть дубликаты рейсов > 1% — записать в паспорт как проблему качества")
q_dates = []
for yy in [2023, 2024, 2025, 2026]:
    for mm in [1, 4, 7, 10]:
        first = pd.Timestamp(f"{yy}-{mm:02d}-01")
        if first > holdout_end:
            continue
        wed = first + pd.Timedelta(days=(2 - first.dayofweek) % 7 + 7)      # 2-я среда месяца
        sat = first + pd.Timedelta(days=(5 - first.dayofweek) % 7 + 7)      # 2-я суббота месяца
        q_dates += [wed, sat]
try:
    durations = md.download_durations([d for d in q_dates if d <= holdout_end], log=log)
    durations.to_parquet(OUT / "durations.parquet", index=False)
    log(f"время рейса: {len(durations):,} рейсов, {durations['date'].nunique()} дат")
except Exception as e:  # noqa: BLE001
    durations = None
    log(f"время рейса не скачано — время оборота по умолчанию {md.RTT_DEFAULT_H} ч (записать в паспорт):", repr(e)[:400])

# %% [markdown]
# ## 3. Прогноз гибрида ВКР (если подключён вывод ноутбука 16)

# %%
hyb = {}
parts = []
for part in ["val", "test"]:
    f = md.find_file(f"hyb_blend__{part}.npz")
    if f is None:
        continue
    z = np.load(f, allow_pickle=False)
    assert (z["stations"].astype(str) == np.array(y.columns)).all(), "порядок станций в прогнозе ВКР не совпадает"
    parts.append((pd.DatetimeIndex(z["origins"]), z["yhat"].astype("float32")))
    log(f"прогноз гибрида ВКР {part}: точек {len(z['origins'])}, {f}")
if parts:
    hyb["hyb"] = (pd.DatetimeIndex(np.concatenate([p[0] for p in parts])), np.concatenate([p[1] for p in parts]))
else:
    log("прогноз гибрида ВКР не подключён — MPC «гибрид ВКР» и H6(а) будут без него (записать)")

# %% [markdown]
# ## 4. Массивы среды

# %%
data, pas = md.build_env_data(y_all, cov, static, trips, durations, hyb=hyb, log=log)
data.save(OUT / "env_data.npz")
log("сутки:", pas["days"], "| период", pas["period"])
log("время оборота, ч:", pas["rtt_h"])
log("парк (поездов в работе):", pas["fleet_max_trains"], "| бюджет буднего дня, поездо-ч:",
    round(pas["budget_trainh_median_weekday"]))

# %% [markdown]
# ## 5. Паспорт данных: описательная статистика и проблемы качества

# %%
G = data.G
names = [g for g, _ in me.GROUPS]
desc = []
for p in ["train", "val", "test", "holdout"]:
    m = data.part == p
    if not m.any():
        continue
    for g in range(G):
        dd, ss = data.dem[m][..., g], data.sched[m][..., g]
        desc.append({"часть": p, "группа": names[g], "сутки": int(m.sum()),
                     "входов в сутки (медиана)": int(np.median(dd.sum(1))),
                     "пик часа (95-й перц.)": int(np.percentile(dd, 95)),
                     "поездов в час (среднее)": round(float(ss.mean()), 1),
                     "поездов в час (макс.)": round(float(ss.max()), 1),
                     "κ·спрос/поезд, пик 95%": round(float(np.percentile(data.kappa[g] * dd / np.maximum(ss, 1e-3), 95)), 2)})
desc = pd.DataFrame(desc)
desc.to_csv(OUT / "describe_groups.csv", index=False)
print(desc.to_string(index=False))
fw = pd.Series(pas["forecast_wape"]).rename("WAPE")
fw.index = pd.MultiIndex.from_tuples([tuple(i.split("|")) for i in fw.index], names=["прогноз", "часть", "горизонт"])
fw = fw.unstack("часть")
fw.to_csv(OUT / "forecast_wape.csv")
print(fw.round(4).to_string())
flags = pd.DataFrame({k: v for k, v in data.flags.items() if k != "daytype"}, index=pd.DatetimeIndex(data.day))
flags["part"] = np.where(data.part == "", "история", data.part)
fl = flags.groupby("part").mean(numeric_only=True).round(3)
print(fl.to_string())

# %%
issues = [
    ("часы без данных (заполнены интерполяцией)", pas["hours_filled"], "весенний перевод часов и пропуски выгрузки"),
    ("станций без управляемых маршрутов", pas["stations_no_group"], "только челноки — входы выпадают"),
    ("доля входов на таких станциях", round(pas["share_entries_no_group"], 4), ""),
    ("доля входов в часы без работающих групп станции", round(pas["share_entries_no_service"], 4),
     "станция обслуживается только закрытыми линиями (плановые работы)"),
    ("сутки без расписания (подставлено типичное)", pas["days_schedule_filled"], ", ".join(pas["days_schedule_filled_list"][:10])),
    ("неизвестные маршруты в расписании", len(pas["routes_unknown"]), ", ".join(pas["routes_unknown"])),
    ("сутки без ковариат ВКР (календарь по правилам)", pas["days_without_vkr_covariates"], "holdout после 24.09.2026"),
    ("дубликаты рейсов в образцах расписания", round(float(dup["dup_share"].max()) if len(dup) else float("nan"), 4), ""),
]
lines = ["# ЛР3. Паспорт данных", "",
         f"Сгенерировано ноутбуком lab3_00_data {pd.Timestamp.now():%d.%m.%Y}.", "",
         "## 1. Источники", "",
         "| источник | что берём | период |", "| --- | --- | --- |",
         f"| датасет ВКР `VKR_PM232_Pevin_MTA` (MTA Subway Hourly Ridership `wujg-7c2s`, `5wq4-mkjj`) | почасовые входы 424 комплексов, календарь | {y.index.min():%d.%m.%Y} – {y.index.max():%d.%m.%Y} |",
         f"| MTA Subway Hourly Ridership `5wq4-mkjj` (скачано здесь) | свежие входы для отложенного периода | {start:%d.%m.%Y} – {holdout_end:%d.%m.%Y} |",
         "| MTA Subway Schedules 2023–2026 (`7pnn-mafy`, `ebrw-j62c`, `q9nv-uegs`, `g8es-h7gb`) | отправления с начальной станции по часам, время рейса | 2023–2026 |",
         "| гибрид ВКР `hyb_blend` (вывод ноутбука 16 ВКР) | прогноз на 24 ч по станциям | " + ("val, test" if hyb else "не подключён") + " |",
         "", "## 2. Разбиение (строго по времени)", "",
         "| часть | сутки |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in pas["days"].items()] + [
         "", f"Период массивов: {pas['period'][0]} – {pas['period'][1]}; первые 4 недели — только история для прогноза.",
         "", "## 3. Описательная статистика по группам линий", "", me_md(desc),
         "", "## 4. Калибровка и ресурсы", "",
         f"- Час пик (будни, train) по группам: {pas['peak_hour']}",
         f"- Медианный спрос в час пик: {pas['peak_demand']}",
         f"- Поездов в час пик по расписанию: {pas['peak_tph']}",
         f"- Время оборота, ч: {pas['rtt_h']} ({pas['rtt_source']})",
         f"- Парк (поездов в работе одновременно): {pas['fleet_max_trains']}; бюджет буднего дня: {round(pas['budget_trainh_median_weekday'])} поездо-ч",
         "", "## 5. Качество прогнозов спроса групп (WAPE, группа × час)", "", me_md(fw.round(4).reset_index()),
         "", "## 6. Режимы суток (доля)", "", me_md(fl.reset_index()),
         "", "## 7. Проблемы качества", "", "| проблема | значение | комментарий |", "| --- | --- | --- |"] + \
        [f"| {a} | {b} | {c} |" for a, b, c in issues] + [
         "", "## 8. Смещения выборки (аналог survivorship bias)", "",
         "- Состав станций фиксирован (424 комплекса ВКР) на весь период; открытые позже станции не появляются, "
         "закрытые на ремонт остаются с нулевыми входами — их спрос не перераспределяется на соседние станции.",
         "- Расписание с поправками отражает плановые работы, но не сбои в реальном времени: фактическое число поездов "
         "в день сбоя ниже расписания, а в среде — нет.",
         "- Спрос — входы: пересадки внутри системы не учитываются; нагрузка магистралей с пересадочными потоками занижена "
         "и компенсируется только калибровкой κ.",
         "", "## 9. Схема предобработки", "",
         "1. Полная сетка часов; пропуски — линейная интерполяция (≤ 6 ч).",
         "2. Станция → группы линий по названию комплекса; входы делятся пропорционально поездам по расписанию.",
         "3. Расписание: отправления с начальной станции (stop_order = 1) по часу отправления, среднее двух направлений; "
         "сутки без расписания — типичное расписание типа дня по train.",
         "4. Прогнозы m4w и norm_level по ряду группы (коэффициенты norm_level — только train); гибрид ВКР — сумма прогнозов станций.",
         "5. Калибровка κ, нормировки, парк и пропускная способность — только по train (кроме пропускной способности — максимум расписания).",
         ]
(OUT / "data_passport.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:40]))

# %%
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True)
tr = data.part == "train"
wk = tr & (data.flags["daytype"] == 0)
for g, ax in enumerate(axes.ravel()):
    ax.plot(np.median(data.dem[wk][..., g], 0), label="спрос, будни (медиана)", color="#1f77b4")
    ax2 = ax.twinx()
    ax2.plot(np.median(data.sched[wk][..., g], 0), color="#d62728", ls="--", label="поездов/ч")
    ax.set_title(names[g])
    ax.set_xticks([0, 6, 12, 18, 23])
fig.suptitle("Спрос группы (входов/ч, синий) и расписание (поездов/ч, красный пунктир), будни train")
fig.tight_layout()
fig.savefig(OUT / "fig_demand_vs_schedule.png", dpi=120)

# %% [markdown]
# ## 6. Проверки среды
# Расписание: награда ровно −24 и нулевой перерасход. MPC и оракул на val — ориентир для обучения.

# %%
env = me.MetroEnv(data, me.EnvParams())
va = data.idx("val")
chk = []
for pol in [me.SchedulePolicy(), me.MPCPolicy("m4w"), me.MPCPolicy("norm"), me.MPCPolicy("oracle")]:
    r = me.rollout(env, pol, va)
    g = 1 - r["J"] / r["J_sched"]
    chk.append({"политика": pol.name, "награда": r["reward"].mean(), "выигрыш, %": 100 * g.mean(),
                "перерасход max": r["overrun"].max(), "поездо-ч, % бюджета": 100 * r["used_frac"].mean()})
chk = pd.DataFrame(chk)
print(chk.round(3).to_string(index=False))
assert abs(chk.iloc[0]["награда"] + 24) < 1e-3, "расписание должно давать −24"
chk.to_csv(OUT / "env_checks_val.csv", index=False)
(OUT / "lab3_00.log").write_text("\n".join(LOG), encoding="utf-8")
print(f"готово за {(time.time() - t0) / 60:.1f} мин")
