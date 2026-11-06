# %% [markdown]
# # ЛР3 · lab3_04 — абляции и перенос выбранного алгоритма на подсети (задания 4 и 6)
#
# Протокол: разд. 9–10. **Accelerator:** нет (CPU). **Internet:** нет. Время: ~1–3 ч (зависит от алгоритма).
# **Inputs:** выводы `lab3-00-data`, `lab3-01-tune`, `lab3-03-select`.
#
# Абляции (выбранный алгоритм, 5 seed, тот же бюджет и гиперпараметры):
# A1 без координатора · A2 без сообщений · A3 без штрафа за смену частоты · A4 без прогноза · A5 прогноз-оракул.
# Перенос: дивизион A (3 группы + координатор) и дивизион B (5 групп + координатор) — выбранный алгоритм и PPO,
# гиперпараметры без изменений. **Output:** `apply/*.pt`, `*.curve.csv`, `*.daily.parquet`, `runs.csv`.

# %%
import os, sys, json, time
from pathlib import Path


def _find(name):
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if name in fn:
                return dp
    raise FileNotFoundError(name)


sys.path.insert(0, _find("marl_env.py"))
import pandas as pd
import marl_algos as ma
import marl_eval as ev
from marl_env import DIVISION
DATA = str(Path(_find("env_data.npz")) / "env_data.npz")
TUNED = json.loads((Path(_find("tuned.json")) / "tuned.json").read_text())
SEL = json.loads((Path(_find("selection.json")) / "selection.json").read_text())
BEST = SEL["choice"]
OUT = ev.results_dir()
RUNS = OUT / "apply"
WORKERS = os.cpu_count() or 2
BUDGET = int(os.environ.get("LAB3_BUDGET", 400_000))   # переменная — только для отладки
SEEDS = [int(s) for s in os.environ.get("LAB3_SEEDS", "0,1,2,3,4").split(",")]
print("выбранный алгоритм:", BEST, "—", SEL["reason"])
ABL = {"A1_no_coord": {"use_coord": False}, "A2_no_msg": {"comm": False}, "A3_no_switch": {"lam_switch": 0.0},
       "A4_no_fc": {"forecast": "none"}, "A5_oracle_fc": {"forecast": "oracle"}}
DIVS = {"divA": {"groups": tuple(DIVISION["A"])}, "divB": {"groups": tuple(DIVISION["B"])}}

# %%
jobs = []
for name, P in ABL.items():
    for s in SEEDS:
        jobs.append({"algo": BEST, "seed": s, "budget": BUDGET, "data_path": DATA, "P": P, "cfg": TUNED.get(BEST, {}),
                     "tag": f"abl__{name}__{BEST}__s{s}", "name": f"{BEST}|{name}", "parts": ["val", "test", "holdout"],
                     "out": str(RUNS)})
for dname, P in DIVS.items():
    for algo in [BEST, "ppo"]:
        for s in SEEDS:
            jobs.append({"algo": algo, "seed": s, "budget": BUDGET, "data_path": DATA, "P": P,
                         "cfg": TUNED.get(algo, {}), "tag": f"div__{dname}__{algo}__s{s}", "name": f"{algo}|{dname}",
                         "parts": ["val", "test", "holdout"], "out": str(RUNS)})
print("прогонов:", len(jobs))
t0 = time.time()
res = ev.run_many(jobs, workers=WORKERS)
print(f"готово за {(time.time() - t0) / 60:.1f} мин; ошибок: {int((~res['ok'].astype(bool)).sum())}")
for _, r in res[~res["ok"].astype(bool)].iterrows():
    print(r["tag"], r.get("error"), r.get("trace"))

# %%
d = ev.load_runs("*.daily.parquet", roots=(str(RUNS),))
t = ev.table(d[d["part"] == "test"])
print(t[["policy", "gain", "ir", "cvar", "crowded", "turnover", "viol_budget", "seed_sd"]].round(4).to_string(index=False))
