# %% [markdown]
# # ЛР3 · lab3_05 — сводка: таблицы отчёта, многокритериальная оценка, стресс-тест, режимы, кооперация,
# # ценность прогноза ВКР, гипотезы, журнал нарушений, карточка рисков
#
# Задания 4–7, требования к отчёту 4–9. **Accelerator:** нет. **Internet:** нет. Время: ~30–60 мин.
# **Inputs:** выводы `lab3-00-data`, `lab3-01-tune`, `lab3-02-main-onpolicy`, `lab3-02-main-offpolicy`,
# `lab3-03-select`, `lab3-04-apply`.
# **Output:** `report_tables.md` (все таблицы отчёта), `summary.json`, `hypotheses.csv`, `violations.csv`,
# `risk_card.md`, рисунки `fig_*.png`.

# %%
import os, sys, json, time
from dataclasses import replace
from pathlib import Path


def _find(name):
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if name in fn:
                return dp
    raise FileNotFoundError(name)


sys.path.insert(0, _find("marl_env.py"))
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import marl_algos as ma
import marl_eval as ev
from marl_env import (DIVISION, GROUPS, EnvData, EnvParams, MetroEnv, MPCPolicy, SchedulePolicy, MULTS, rollout)
torch.set_num_threads(os.cpu_count() or 2)
DATA = str(Path(_find("env_data.npz")) / "env_data.npz")
data = EnvData.load(DATA)
SEL = json.loads((Path(_find("selection.json")) / "selection.json").read_text())
BEST = SEL["choice"]
OUT = ev.results_dir()
SEEDS = [int(s) for s in os.environ.get("LAB3_SEEDS", "0,1,2,3,4").split(",")]
T = ma.TITLE
MD = ["# ЛР3. Таблицы отчёта (сгенерировано lab3_05_summary)", "",
      f"Выбранный алгоритм: **{T.get(BEST, BEST)}** — {SEL['reason']} (выбор зафиксирован {SEL['time'][:16]}).", ""]
SUM = {"best": BEST, "selection_reason": SEL["reason"]}
pct = lambda x: f"{100 * x:+.2f}%"
print("выбран:", BEST)

main = ev.load_runs("main__*.daily.parquet")
appl = ev.load_runs("abl__*.daily.parquet")
divs = ev.load_runs("div__*.daily.parquet")
base = ev.baseline_frames(data, EnvParams())
allf = pd.concat([main, base], ignore_index=True)
test, hold = allf[allf["part"] == "test"], allf[allf["part"] == "holdout"]
pax_day = base[(base["policy"] == "schedule") & (base["part"] == "test")]["pax"].mean()
print("суток test:", test["day"].nunique(), "| holdout:", hold["day"].nunique())

# %% [markdown]
# ## 1. Главные таблицы (как в примере задания): качество и риск/ресурсы/устойчивость, test

# %%
ORDER = ma.ALGOS + ["ppo", "schedule", "mpc_m4w", "mpc_norm", "mpc_hyb", "mpc_oracle"]


def full_table(f):
    t = ev.table(f)
    t[["ci_lo", "ci_hi"]] = [ev.boot_ci(f[f["policy"] == p]) for p in t["policy"]]
    t["o"] = t["policy"].map({p: i for i, p in enumerate(ORDER)}).fillna(99)
    t["pax_h_saved"] = t["gain"] * f.groupby("policy")["J_sched"].mean().reindex(t["policy"]).to_numpy() / 60
    return t.sort_values("o").drop(columns="o").reset_index(drop=True)


tt = full_table(test)
tt.to_csv(OUT / "table_test_full.csv", index=False)
t1 = pd.DataFrame({"Алгоритм": tt["policy"].map(lambda p: T.get(p, p)),
                   "Выигрыш к расписанию": [f"{100 * g:+.2f}% [{100 * a:+.2f}; {100 * b:+.2f}]"
                                            for g, a, b in zip(tt["gain"], tt["ci_lo"], tt["ci_hi"])],
                   "IR (аналог Sharpe)": tt["ir"].round(2), "Sortino": tt["sortino"].round(2),
                   "Худшие сутки": (100 * tt["worst"]).round(2).astype(str) + "%",
                   "Потери, мин/пасс.": tt["cost_per_pax"].round(3),
                   "Сэкономлено пасс.-ч/сутки": tt["pax_h_saved"].round(0)})
t2 = pd.DataFrame({"Алгоритм": t1["Алгоритм"], "CVaR 5% суток": (100 * tt["cvar"]).round(2).astype(str) + "%",
                   "Оборот, % бюджета/сут": (100 * tt["turnover"]).round(1),
                   "Перерасход бюджета, % суток": (100 * tt["viol_budget"]).round(1),
                   "Часы выше норматива, %": (100 * tt["breach"]).round(2),
                   "Пасс. в переполненных, %": (100 * tt["crowded"]).round(2),
                   "Вмешательства защиты, %": (100 * tt["clip_share"]).round(1),
                   "Выигрыш по seed, ср. ± ст. откл.": [f"{100 * g:+.2f} ± {100 * s:.2f}" if n > 1 else "—"
                                                        for g, s, n in zip(tt["gain"], tt["seed_sd"], tt["n_seeds"])]})
MD += ["## Таблица 1. Качество на test (01.07–24.09.2026); в скобках 95% ДИ (бутстреп: seed × недели)", "",
       ev.to_md(t1), "", "## Таблица 2. Риск, ресурсы, нарушения, устойчивость по seed (test)", "", ev.to_md(t2), ""]
print(t1.to_string(index=False))
print(t2.to_string(index=False))
SUM["test"] = tt.set_index("policy")[["gain", "ci_lo", "ci_hi", "ir", "cvar", "crowded", "turnover", "viol_budget",
                                      "seed_sd", "pax_h_saved"]].to_dict("index")

# %% [markdown]
# ## 2. Сводный балл и выбор (val) — повтор для отчёта

# %%
tv = pd.read_csv(Path(_find("table_val.csv")) / "table_val.csv")
sv = tv[tv["policy"].isin(ma.ALGOS + ["ppo"])][["policy", "gain", "ir", "cvar", "crowded", "turnover", "seed_sd",
                                                   "viol_budget", "score"]].sort_values("score", ascending=False)
sv_md = sv.assign(policy=sv["policy"].map(T), gain=(100 * sv["gain"]).round(2), cvar=(100 * sv["cvar"]).round(2),
                  crowded=(100 * sv["crowded"]).round(2), turnover=(100 * sv["turnover"]).round(1),
                  seed_sd=(100 * sv["seed_sd"]).round(2), viol_budget=(100 * sv["viol_budget"]).round(1),
                  ir=sv["ir"].round(2), score=sv["score"].round(3))
MD += ["## Таблица 3. Сводный балл на val (35% выигрыш, 20% IR, 20% CVaR, 10% переполнение, 10% оборот, 5% seed)", "",
       ev.to_md(sv_md), ""]

# %% [markdown]
# ## 3. Применение лучшей политики (задание 4): отложенный период и подсети

# %%
rows = []
if len(hold):
    th = full_table(hold)
    th.to_csv(OUT / "table_holdout.csv", index=False)
    for _, r in th[th["policy"].isin([BEST, "ppo", "schedule", "mpc_norm", "mpc_oracle"])].iterrows():
        rows.append({"сеть": "вся (8 групп)", "период": f"holdout ({hold['day'].nunique()} сут)", "политика": T.get(r["policy"]),
                     "выигрыш": f"{100 * r['gain']:+.2f}% [{100 * r['ci_lo']:+.2f}; {100 * r['ci_hi']:+.2f}]",
                     "IR": round(r["ir"], 2), "перерасход, % сут": round(100 * r["viol_budget"], 1)})
    SUM["holdout"] = th.set_index("policy")[["gain", "ci_lo", "ci_hi", "ir"]].to_dict("index")
for dname, gs in [("divA", DIVISION["A"]), ("divB", DIVISION["B"])]:
    P = EnvParams(groups=tuple(gs))
    bd = ev.baseline_frames(data, P, mpc=("norm", "oracle"))
    bd["policy"] = bd["policy"] + f"|{dname}"
    dd = pd.concat([divs[divs["policy"].str.endswith(dname)], bd], ignore_index=True) if len(divs) else bd
    for part in ["test", "holdout"]:
        f = dd[dd["part"] == part]
        if not len(f):
            continue
        t = ev.table(f)
        for _, r in t.iterrows():
            lo, hi = ev.boot_ci(f[f["policy"] == r["policy"]])
            pol = r["policy"].split("|")[0]
            rows.append({"сеть": "дивизион A (1·2·3, 4·5·6, 7)" if dname == "divA" else "дивизион B (A·C·E … G)",
                         "период": part, "политика": T.get(pol, pol),
                         "выигрыш": f"{100 * r['gain']:+.2f}% [{100 * lo:+.2f}; {100 * hi:+.2f}]",
                         "IR": round(r["ir"], 2), "перерасход, % сут": round(100 * r["viol_budget"], 1)})
app = pd.DataFrame(rows)
app.to_csv(OUT / "application.csv", index=False)
MD += ["## Таблица 4. Применение (задание 4): полностью отложенный период и подсети, переобучение без подбора", "",
       ev.to_md(app), ""]
print(app.to_string(index=False))

# %% [markdown]
# ## 4. Кооперация агентов (все 8 MARL × 5 seed, test)

# %%
env_full = MetroEnv(data, EnvParams())
te_days = data.idx("test")
coop = []
for algo in ma.ALGOS:
    for s in SEEDS:
        ck = ev.find_all(f"main__{algo}__s{s}.pt")
        if not ck:
            continue
        pol = ev.load_policy(ck[0], env_full)
        c = ev.cooperation(pol, env_full, te_days, algo)
        c["seed"] = s
        coop.append(c)
coop = pd.DataFrame(coop)
coop.to_csv(OUT / "cooperation.csv", index=False)
ca = coop.groupby("policy")[["gain", "synergy", "realloc", "align"]].mean()
loo = coop.groupby("policy")[[c for c in coop.columns if c.startswith("loo|")]].mean()
ca["loo_sum"] = loo.sum(1)
ca["loo_coord"] = loo.get("loo|координатор", np.nan)
ca = ca.reindex(ma.ALGOS)
print(ca.round(4).to_string())
coop_md = pd.DataFrame({"Алгоритм": [T[a] for a in ca.index], "выигрыш": (100 * ca["gain"]).round(2),
                        "синергия (все − Σ по одному)": (100 * ca["synergy"]).round(2),
                        "Σ вкладов (замена агента расписанием)": (100 * ca["loo_sum"]).round(2),
                        "вклад координатора": (100 * ca["loo_coord"]).round(2),
                        "перераспределено, % бюджета": (100 * ca["realloc"]).round(1),
                        "согласованность (корр.)": ca["align"].round(3)})
MD += ["## Таблица 5. Кооперация агентов (test, среднее по 5 seed; выигрыш и вклады — п.п.)", "", ev.to_md(coop_md), "",
       "Синергия > 0: вместе агенты дают больше, чем сумма их действий по одному (перераспределение общего ресурса). "
       "Согласованность — корреляция «добавленные поезда ↔ ожидаемое наполнение по расписанию».", ""]
best_loo = coop[coop["policy"] == BEST][[c for c in coop.columns if c.startswith("loo|")]].mean()
MD += [f"Вклад агентов {T[BEST]} (п.п. выигрыша, которые теряются при замене агента расписанием): " +
       ", ".join(f"{k[4:]} {100 * v:+.2f}" for k, v in best_loo.items()), ""]

# %% [markdown]
# ## 5. Многокритериальная оценка (задание 5)

# %%
cost = pd.read_csv(Path(_find("cost.csv")) / "cost.csv").set_index("algo")
mc = tt.set_index("policy").reindex(ma.ALGOS + ["ppo"])
mc_md = pd.DataFrame({"Алгоритм": [T[a] for a in mc.index], "выигрыш, %": (100 * mc["gain"]).round(2),
                      "IR": mc["ir"].round(2), "CVaR, %": (100 * mc["cvar"]).round(2),
                      "худшие сутки, %": (100 * mc["worst"]).round(2), "оборот, %": (100 * mc["turnover"]).round(1),
                      "поездо-ч, % бюджета": (100 * mc["used_frac"]).round(1),
                      "перерасход, % сут": (100 * mc["viol_budget"]).round(1),
                      "ст. откл. по seed, п.п.": (100 * mc["seed_sd"]).round(2),
                      "синергия, п.п.": [round(100 * ca["synergy"].get(a, np.nan), 2) for a in mc.index],
                      "обучение, мин": [round(cost["train_min"].get(a, np.nan), 1) for a in mc.index],
                      "мс на решение": [round(cost["infer_ms"].get(a, np.nan), 2) for a in mc.index],
                      "параметров": [int(cost["params"].get(a, 0)) for a in mc.index]})
MD += ["## Таблица 6. Многокритериальная оценка (test)", "", ev.to_md(mc_md), ""]

# %% [markdown]
# ## 6. Абляции выбранного алгоритма (test, парные разницы к основному варианту, те же seed)

# %%
ABL_T = {"A1_no_coord": "A1 без координатора", "A2_no_msg": "A2 без сообщений", "A3_no_switch": "A3 без штрафа за смену частоты",
         "A4_no_fc": "A4 без прогноза в наблюдении", "A5_oracle_fc": "A5 прогноз-оракул"}
ab_rows, pvals = [], {}
mt = test[test["policy"] == BEST]
for key, title in ABL_T.items():
    f = appl[(appl["policy"] == f"{BEST}|{key}") & (appl["part"] == "test")] if len(appl) else pd.DataFrame()
    if not len(f):
        continue
    s = ev.summarize(f)
    pdiff = ev.paired_diff(f, mt)
    pvals[key] = pdiff["p"]
    ab_rows.append({"вариант": title, "выигрыш, %": round(100 * s["gain"], 2),
                    "разница к основному, п.п.": f"{100 * pdiff['diff']:+.2f} [{100 * pdiff['lo']:+.2f}; {100 * pdiff['hi']:+.2f}]",
                    "p": round(pdiff["p"], 4), "IR": round(s["ir"], 2), "CVaR, %": round(100 * s["cvar"], 2),
                    "оборот, %": round(100 * s["turnover"], 1), "перерасход, % сут": round(100 * s["viol_budget"], 1),
                    "переполнение, %": round(100 * s["crowded"], 2), "_key": key, "_diff": pdiff["diff"],
                    "_lo": pdiff["lo"], "_hi": pdiff["hi"], "_viol": s["viol_budget"]})
s0 = ev.summarize(mt)
ab = pd.DataFrame(ab_rows)
MD += [f"## Таблица 7. Абляции {T[BEST]} (test; основной вариант: выигрыш {100 * s0['gain']:.2f}%, IR {s0['ir']:.2f}, "
       f"оборот {100 * s0['turnover']:.1f}%, перерасход {100 * s0['viol_budget']:.1f}% суток)", "",
       ev.to_md(ab.drop(columns=[c for c in ab.columns if c.startswith("_")])) if len(ab) else "нет данных", ""]
print(ab.drop(columns=[c for c in ab.columns if c.startswith("_")]).to_string(index=False) if len(ab) else "нет абляций")

# %% [markdown]
# ## 7. Стресс-тест (без переобучения, test)

# %%
def policies_for(env):
    out = {"schedule": [SchedulePolicy()], "mpc_norm": [MPCPolicy("norm")]}
    for algo in [BEST, "ppo"]:
        out[algo] = [ev.load_policy(ev.find_all(f"main__{algo}__s{s}.pt")[0], env) for s in SEEDS
                     if ev.find_all(f"main__{algo}__s{s}.pt")]
    return out


st_rows = []
for sc, desc in ev.STRESS.items():
    d2, P2 = ev.stress(data, EnvParams(), sc, te_days, seed=0)
    env = MetroEnv(d2, P2)
    for name, pols in policies_for(env).items():
        fr = pd.concat([ev.evaluate(p, env, te_days, name, i, "test") for i, p in enumerate(pols)])
        s = ev.summarize(fr)
        st_rows.append({"сценарий": sc, "описание": desc, "политика": name, "gain": s["gain"], "cvar": s["cvar"],
                        "viol_budget": s["viol_budget"], "crowded": s["crowded"], "cost_per_pax": s["cost_per_pax"]})
stx = pd.DataFrame(st_rows)
stx.to_csv(OUT / "stress.csv", index=False)
piv = stx.pivot_table(index=["сценарий", "описание"], columns="политика", values="gain").mul(100).round(2)
piv = piv[[c for c in ["schedule", "mpc_norm", "ppo", BEST] if c in piv.columns]]
piv.columns = [T.get(c, c) for c in piv.columns]
base_row = tt.set_index("policy")
MD += ["## Таблица 8. Стресс-тест: выигрыш к обычному расписанию без стресса, % (test)", "",
       "В строке «без стресса» — выигрыш из таблицы 1. Расписание в стрессе исполняется как есть (в S3 его урезает лимит парка).", "",
       ev.to_md(piv.reset_index()), ""]
vp = stx.pivot_table(index="сценарий", columns="политика", values="viol_budget").mul(100).round(1)
MD += ["Перерасход бюджета в стрессе, % суток:", "", ev.to_md(vp.reset_index()), ""]
print(piv.to_string())

# %% [markdown]
# ## 8. Режимы: где политика деградирует (test + holdout)

# %%
th_all = allf[allf["part"].isin(["test", "holdout"])]
REG = {"будни": lambda f: (f["flag_daytype"] == 0) & ~f["flag_planned_works"] & ~f["flag_holiday"],
       "выходные": lambda f: f["flag_weekend"] & ~f["flag_holiday"],
       "праздники": lambda f: f["flag_holiday"].astype(bool),
       "плановые работы": lambda f: f["flag_planned_works"].astype(bool),
       "городские события": lambda f: f["flag_city_event"].astype(bool)}
rg = []
for reg, fn in REG.items():
    for pol in [BEST, "ppo", "mpc_norm", "mpc_oracle"]:
        f = th_all[th_all["policy"] == pol]
        f = f[fn(f)]
        if f["day"].nunique() < 2:
            continue
        lo, hi = ev.boot_ci(f)
        rg.append({"режим": reg, "сутки": f["day"].nunique(), "политика": T.get(pol, pol), "gain": f["gain"].mean(),
                   "lo": lo, "hi": hi, "worst": f.groupby("day")["gain"].mean().min()})
rg = pd.DataFrame(rg)
rg.to_csv(OUT / "regimes.csv", index=False)
rg_md = rg.assign(**{"выигрыш": [f"{100 * g:+.2f}% [{100 * a:+.2f}; {100 * b:+.2f}]" for g, a, b in zip(rg["gain"], rg["lo"], rg["hi"])],
                     "худшие сутки, %": (100 * rg["worst"]).round(2)})[["режим", "сутки", "политика", "выигрыш", "худшие сутки, %"]]
MD += ["## Таблица 9. Режимы (test + holdout)", "", ev.to_md(rg_md), ""]
print(rg_md.to_string(index=False))
# худшие сутки выбранной политики — с флагами
wd = th_all[th_all["policy"] == BEST].groupby("day").agg(gain=("gain", "mean"), **{
    k: (k, "first") for k in ["flag_holiday", "flag_weekend", "flag_planned_works", "flag_city_event"]}).nsmallest(10, "gain")
wd.to_csv(OUT / "worst_days.csv")
MD += ["Десять худших суток выбранной политики:", "", ev.to_md(wd.reset_index().assign(gain=lambda x: (100 * x["gain"]).round(2))), ""]

# %% [markdown]
# ## 9. Ценность прогноза ВКР (H6): лестница прогнозов и RL с разными прогнозами

# %%
fw = pd.read_csv(Path(_find("forecast_wape.csv")) / "forecast_wape.csv")
fw = fw[fw["горизонт"] == "all"].set_index("прогноз")
lad = []
for src in ["m4w", "norm", "hyb", "oracle"]:
    pol = f"mpc_{src}"
    f = test[test["policy"] == pol]
    if not len(f):
        continue
    days_ok = f["day"].unique()
    sched_f = test[(test["policy"] == "schedule") & test["day"].isin(days_ok)]
    lo, hi = ev.boot_ci(f)
    w = 0.0 if src == "oracle" else float(fw.loc[src, "test"]) if src in fw.index else np.nan
    lad.append({"прогноз": src, "WAPE (test, все горизонты)": w, "выигрыш MPC": f["gain"].mean(), "lo": lo, "hi": hi,
                "сутки": len(days_ok)})
lad = pd.DataFrame(lad)
if len(lad) >= 2:
    k = np.polyfit(lad["WAPE (test, все горизонты)"], lad["выигрыш MPC"], 1)[0]
    SUM["gain_per_wape_pp"] = float(-k / 100 * 100)   # п.п. выигрыша на 1 п.п. WAPE
# RL: тот же алгоритм с разным прогнозом в наблюдении
rl = [{"вариант": f"{T[BEST]}: прогноз norm_level (основной)", "выигрыш": s0["gain"]}]
for key in ["A4_no_fc", "A5_oracle_fc"]:
    r = [x for x in ab_rows if x["_key"] == key]
    if r:
        rl.append({"вариант": f"{T[BEST]}: {ABL_T[key]}", "выигрыш": s0["gain"] + r[0]["_diff"]})
# подстановка прогноза гибрида ВКР в наблюдение без переобучения (сутки, где он есть)
if "hyb" in data.fc:
    ok = ~np.isnan(data.fc["hyb"][te_days, 0, 0, 0])
    dd = te_days[ok]
    if len(dd):
        envh = MetroEnv(data, EnvParams(forecast="hyb"))
        envn = MetroEnv(data, EnvParams())
        fh, fn_ = [], []
        for s in SEEDS:
            ck = ev.find_all(f"main__{BEST}__s{s}.pt")
            if ck:
                fh.append(ev.evaluate(ev.load_policy(ck[0], envh), envh, dd, "hyb", s, "test"))
                fn_.append(ev.evaluate(ev.load_policy(ck[0], envn), envn, dd, "norm", s, "test"))
        if fh:
            fh, fn_ = pd.concat(fh), pd.concat(fn_)
            pdh = ev.paired_diff(fh, fn_)
            rl.append({"вариант": f"{T[BEST]}: прогноз гибрида ВКР вместо norm_level (без переобучения)",
                       "выигрыш": fh["gain"].mean(), "разница": pdh["diff"], "lo": pdh["lo"], "hi": pdh["hi"]})
rl = pd.DataFrame(rl)
MD += ["## Таблица 10. Ценность прогноза (связь с ВКР): MPC с разными прогнозами (test)", "",
       ev.to_md(lad.assign(**{"WAPE (test, все горизонты)": (100 * lad["WAPE (test, все горизонты)"]).round(2),
                              "выигрыш MPC": (100 * lad["выигрыш MPC"]).round(2), "lo": (100 * lad["lo"]).round(2),
                              "hi": (100 * lad["hi"]).round(2)})), "",
       f"Наклон: {SUM.get('gain_per_wape_pp', float('nan')):.3f} п.п. выигрыша на 1 п.п. WAPE (линейная аппроксимация по лестнице).", "",
       "RL с разными прогнозами:", "", ev.to_md(rl.assign(**{c: (100 * rl[c]).round(2) for c in rl.columns if c != "вариант"})), ""]
print(lad.to_string(index=False))
print(rl.to_string(index=False))
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(100 * lad["WAPE (test, все горизонты)"], 100 * lad["выигрыш MPC"],
            yerr=[100 * (lad["выигрыш MPC"] - lad["lo"]), 100 * (lad["hi"] - lad["выигрыш MPC"])], fmt="o-")
for _, r in lad.iterrows():
    ax.annotate(r["прогноз"], (100 * r["WAPE (test, все горизонты)"], 100 * r["выигрыш MPC"]), textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("WAPE прогноза спроса групп, %")
ax.set_ylabel("выигрыш MPC к расписанию, %")
ax.set_title("Цена точности прогноза (test)")
fig.tight_layout()
fig.savefig(OUT / "fig_forecast_value.png", dpi=120)

# %% [markdown]
# ## 10. Гипотезы H1–H7

# %%
pairs = pd.read_csv(Path(_find("pairs_test.csv")) / "pairs_test.csv").set_index("сравнение")
H = []


def pr(name):
    return pairs.loc[name] if name in pairs.index else None


r = pr("H1: MAPPO − IPPO")
if r is not None:
    H.append(("H1", "MAPPO лучше IPPO (CTDE полезно)", f"{100 * r['diff']:+.2f} п.п. [{100 * r['lo']:+.2f}; {100 * r['hi']:+.2f}]",
              r["p"], "подтверждена" if r["lo"] > 0 else ("опровергнута" if r["hi"] < 0 else "не подтверждена (незначимо)")))
b = SUM["test"][BEST]
hb = SUM.get("holdout", {}).get(BEST)
H.append(("H2", "выбранный лучше расписания на test и holdout",
          f"test {100 * b['gain']:+.2f}% [{100 * b['ci_lo']:+.2f}; {100 * b['ci_hi']:+.2f}]" +
          (f"; holdout {100 * hb['gain']:+.2f}% [{100 * hb['ci_lo']:+.2f}; {100 * hb['ci_hi']:+.2f}]" if hb else "; holdout нет"),
          np.nan, "подтверждена" if b["ci_lo"] > 0 and (hb is None or hb["ci_lo"] > 0) else "не подтверждена"))
r = pr("H3: выбранный − PPO (один агент)")
if r is not None:
    H.append(("H3", "выбранный MARL не хуже одноагентного PPO больше чем на 1 п.п.",
              f"{100 * r['diff']:+.2f} п.п. [{100 * r['lo']:+.2f}; {100 * r['hi']:+.2f}]", r["p"],
              "подтверждена" if r["lo"] > -0.01 else "не подтверждена"))
for key, hk, txt in [("A1_no_coord", "H4", "без координатора хуже"), ("A2_no_msg", "H5", "без сообщений хуже"),
                     ("A4_no_fc", "H6б", "без прогноза хуже")]:
    rr = [x for x in ab_rows if x["_key"] == key]
    if rr:
        x = rr[0]
        worse = x["_hi"] < 0 or (key == "A1_no_coord" and x["_viol"] > s0["viol_budget"] + 0.02)
        H.append((hk, txt, f"{100 * x['_diff']:+.2f} п.п. [{100 * x['_lo']:+.2f}; {100 * x['_hi']:+.2f}]; перерасход "
                  f"{100 * x['_viol']:.1f}% сут", pvals.get(key, np.nan),
                  "подтверждена" if worse else ("опровергнута" if x["_lo"] > 0 else "не подтверждена (незначимо)")))
if len(lad) >= 3:
    mono = lad.sort_values("WAPE (test, все горизонты)", ascending=False)["выигрыш MPC"].is_monotonic_increasing
    H.append(("H6а", "выигрыш MPC растёт с точностью прогноза",
              "; ".join(f"{r['прогноз']}: WAPE {100 * r['WAPE (test, все горизонты)']:.1f}% → {100 * r['выигрыш MPC']:+.2f}%"
                        for _, r in lad.iterrows()), np.nan, "подтверждена" if mono else "не подтверждена"))
    H.append(("H6в", "п.п. выигрыша на 1 п.п. WAPE", f"{SUM.get('gain_per_wape_pp', np.nan):.3f}", np.nan, "оценка"))
r = pr("H7: выбранный − MPC norm_level")
if r is not None:
    H.append(("H7", "RL против «прогноз + оптимизация» (направление не задано)",
              f"{100 * r['diff']:+.2f} п.п. [{100 * r['lo']:+.2f}; {100 * r['hi']:+.2f}]", r["p"],
              "RL лучше" if r["lo"] > 0 else ("MPC лучше" if r["hi"] < 0 else "различие незначимо")))
hyp = pd.DataFrame(H, columns=["гипотеза", "формулировка", "результат", "p", "вывод"])
# поправка Холма по проверкам с p
m = hyp["p"].notna()
ps = hyp.loc[m, "p"].to_numpy()
order = np.argsort(ps)
adj = np.empty_like(ps)
run = 0.0
for i, j in enumerate(order):
    run = max(run, min(1.0, (len(ps) - i) * ps[j]))
    adj[j] = run
hyp.loc[m, "p (Холм)"] = adj
hyp.to_csv(OUT / "hypotheses.csv", index=False)
MD += ["## Таблица 11. Гипотезы", "", ev.to_md(hyp.round(4)), ""]
print(hyp.to_string(index=False))

# %% [markdown]
# ## 11. Статистический и экономический эффект (задание 6), кривая «капитала»

# %%
econ = []
for p in [BEST, "ppo", "mpc_norm", "mpc_oracle"]:
    if p in SUM["test"]:
        x = SUM["test"][p]
        econ.append({"политика": T.get(p, p), "выигрыш, %": round(100 * x["gain"], 2),
                     "статистически": "значим" if x["ci_lo"] > 0 else "незначим",
                     "экономически (≥ 1% при перерасходе ≤ 0,5%)": "значим" if (x["gain"] >= 0.01 and x["viol_budget"] <= 0.02) else "мал",
                     "пасс.-ч в сутки": round(x["pax_h_saved"]), "пасс.-ч за test": round(x["pax_h_saved"] * test["day"].nunique())})
econ = pd.DataFrame(econ)
MD += ["## Таблица 12. Статистический против экономического эффекта (test)", "", ev.to_md(econ), ""]
fig, ax = plt.subplots(figsize=(11, 4))
for p in [BEST, "ppo", "mpc_norm", "mpc_oracle"]:
    f = th_all[th_all["policy"] == p]
    if not len(f):
        continue
    per = f.assign(saved=(f["J_sched"] - f["J"]) / 60).pivot_table(index="day", columns="seed", values="saved").sort_index()
    cum = per.cumsum() / 1e3
    ax.plot(cum.index, cum.mean(1), label=T.get(p, p))
    if cum.shape[1] > 1:
        ax.fill_between(cum.index, cum.min(1), cum.max(1), alpha=0.2)
if len(hold):
    ax.axvline(pd.Timestamp(hold["day"].min()), color="grey", ls="--", lw=1)
    ax.text(pd.Timestamp(hold["day"].min()), ax.get_ylim()[1] * 0.9, " holdout", color="grey")
ax.set_ylabel("сэкономлено, тыс. пасс.-ч (накопл.)")
ax.set_title("Накопленный выигрыш к расписанию (аналог кривой капитала); полоса — разброс по seed")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig_cumulative.png", dpi=120)

# %% [markdown]
# ## 12. Что делает политика: средний множитель по часам и группам (интерпретация)

# %%
ck = ev.find_all(f"main__{BEST}__s0.pt")
if ck:
    pol = ev.load_policy(ck[0], env_full)
    r = rollout(env_full, pol, te_days, record=True)
    s = env_full.sched[torch.as_tensor(te_days)].numpy()
    mult = (r["trace"]["n"] / np.maximum(s, 1e-3))
    wk = data.flags["daytype"][te_days] == 0
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    for a, msk, ttl in [(ax[0], wk, "будни"), (ax[1], ~wk, "выходные и праздники")]:
        im = a.imshow(mult[msk].mean(0).T, aspect="auto", cmap="RdBu_r", vmin=0.7, vmax=1.3)
        a.set_yticks(range(data.G))
        a.set_yticklabels([g for g, _ in GROUPS])
        a.set_xlabel("час")
        a.set_title(f"{T[BEST]}, seed 0: поездов / расписание — {ttl}")
    fig.colorbar(im, ax=ax)
    fig.savefig(OUT / "fig_policy_heatmap.png", dpi=120, bbox_inches="tight")

# %% [markdown]
# ## 13. Журнал нарушений и карточка рисков (задание 7)

# %%
J = pd.concat([allf, appl, divs], ignore_index=True)
J = J[J["part"].isin(["test", "holdout"])]
sch_breach = J[J["policy"] == "schedule"].groupby(["part", "day"])["breach_share"].first()
J["breach_vs_sched"] = J["breach_share"].to_numpy() - sch_breach.reindex(pd.MultiIndex.from_frame(J[["part", "day"]])).to_numpy()
viol = J[(J["overrun"] > ev.TOL_OVERRUN) | (J["breach_vs_sched"] > 0.01) | (J["cap_share"] > 0)]
viol = viol[["policy", "seed", "part", "day", "overrun", "used_frac", "breach_share", "breach_vs_sched", "clip_share", "cap_share"]]
viol.to_csv(OUT / "violations.csv", index=False)
vsum = J.groupby("policy").agg(суток=("day", "size"), перерасход=("overrun", lambda x: (x > ev.TOL_OVERRUN).mean()),
                               max_перерасход=("overrun", "max"), выше_норматива_чем_расписание=("breach_vs_sched", lambda x: (x > 0.01).mean()),
                               лимит_парка=("cap_share", lambda x: (x > 0).mean())).reset_index()
vsum.to_csv(OUT / "violations_summary.csv", index=False)
MD += ["## Таблица 13. Журнал нарушений: сводка (test + holdout, все прогоны). Полный журнал — violations.csv", "",
       ev.to_md(vsum.assign(policy=vsum["policy"].map(lambda p: T.get(p.split("|")[0], p.split("|")[0]) + ("|" + p.split("|")[1] if "|" in p else "")))
                .round(4)), ""]

# %%
P0 = EnvParams()
pas = data.meta.get("passport", {})
bt = SUM["test"].get(BEST, {})
worst_reg = rg[rg["политика"] == T[BEST]].sort_values("gain").head(2) if len(rg) else pd.DataFrame()
worst_st = stx[stx["политика"] == BEST].sort_values("gain").head(3) if len(stx) else pd.DataFrame()
card = f"""# ЛР3. Карточка рисков политики {T[BEST]}

Сгенерировано lab3_05_summary {pd.Timestamp.now():%d.%m.%Y}. Это учебная модель; она не утверждена MTA и не может
управлять движением без диспетчера.

## Назначение
Почасовая рекомендация диспетчерскому центру: множитель к расписанию для 8 магистралей и лимит выпуска поездов
при неизменном суточном бюджете поездо-часов. Решение — рекомендация, исполнение — после проверки человеком.

## Что показано
- test: выигрыш {pct(bt.get('gain', np.nan))} [{pct(bt.get('ci_lo', np.nan))}; {pct(bt.get('ci_hi', np.nan))}] потерь времени пассажиров, ≈ {bt.get('pax_h_saved', np.nan):.0f} пасс.-ч в сутки;
  перерасход бюджета в {100 * bt.get('viol_budget', np.nan):.1f}% суток; разброс по seed ± {100 * bt.get('seed_sd', np.nan):.2f} п.п.
- holdout: {('выигрыш ' + pct(SUM['holdout'][BEST]['gain'])) if 'holdout' in SUM and BEST in SUM['holdout'] else 'нет данных'}.

## Допущения модели (главные источники риска)
1. Спрос не зависит от частоты в пределах часа (нет перераспределения пассажиров между линиями и часами).
2. Изменение выпуска ±30% внутри суток выполнимо (резерв составов и бригад; перестановки не моделируются).
3. Наполнение — по калибровке: в типичный будний пик ρ = {P0.rho_peak} норматива по расписанию; замеров загрузки нет.
   При ошибке калибровки на 25% (S5) выигрыш меняется — см. таблицу 8.
4. Агрегация до магистрали: внутри группы (например, 4 и 6) поезда считаются взаимозаменяемыми; направления усреднены.
5. Пересадки не моделируются; входы на пересадочных станциях делятся пропорционально поездам.
6. Расписание с поправками — это план; реальные сбои в среде не воспроизводятся.

## Где политика деградирует
{chr(10).join(f'- режим «{r.режим}»: выигрыш {pct(r.gain)} [{pct(r.lo)}; {pct(r.hi)}]' for r in worst_reg.itertuples()) or '- см. таблицу 9'}
{chr(10).join(f'- стресс {r.сценарий} ({r.описание}): выигрыш {pct(r.gain)}, перерасход {100 * r.viol_budget:.1f}% суток' for r in worst_st.itertuples()) or '- см. таблицу 8'}

## Ограничения переноса в реальную систему
- Нужны онлайн-данные о входах с задержкой не больше часа (в открытом наборе MTA задержка — дни).
- Решение «добавить 15% поездов через час» должно быть исполнимо: резервные составы на конечных, бригады в резерве.
- Нормативы MTA (loading guidelines) — плановые, а не оперативные; оперативные решения требуют согласования.
- Политика обучена на 2023–2026; структурные сдвиги (новые тарифы, новые линии) требуют переобучения и
  мониторинга дрейфа (сравнение выигрыша с MPC на тех же сутках).

## Контроль
- Защитный слой: нормативный интервал, пропускная способность, лимит парка — нарушения невозможны, вмешательства
  записываются; бюджет — мягкое ограничение, журнал нарушений `violations.csv`.
- Запасной режим: если выигрыш политики к MPC norm_level за неделю отрицателен — вернуться к MPC или расписанию.
- Вычислительная стоимость: обучение ≈ {cost['train_min'].get(BEST, np.nan):.0f} мин CPU, решение ≈ {cost['infer_ms'].get(BEST, np.nan):.2f} мс.
"""
(OUT / "risk_card.md").write_text(card, encoding="utf-8")
(OUT / "report_tables.md").write_text("\n".join(MD), encoding="utf-8")
(OUT / "summary.json").write_text(json.dumps(SUM, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print(card)
