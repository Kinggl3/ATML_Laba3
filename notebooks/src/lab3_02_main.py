# %% [markdown]
# # ЛР3 · lab3_02 — сравнение алгоритмов: 5 seed × одинаковый бюджет (часть GROUP)
#
# Задание 3 (протокол: разд. 5). **Accelerator:** нет (CPU). **Internet:** нет.
# **Inputs:** выводы `lab3-00-data`, `lab3-01-tune`.
#
# Ноутбук собирается в двух вариантах, которые можно запускать параллельно:
# `lab3_02_main_onpolicy` — IPPO, MAPPO, HAPPO, MAT, COMA, PPO (~1–1,5 ч);
# `lab3_02_main_offpolicy` — MADDPG, VDN, QMIX (~1–2 ч).
# Каждый прогон: 400 000 шагов, 12 оценок на val, лучший чекпойнт → оценка на val, test, holdout.
# **Output:** `main/*.pt`, `*.curve.csv`, `*.daily.parquet`, `runs.csv`.

# %%
import os, sys, json, time
from pathlib import Path


def _find(name):
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if name in fn:
                return dp
    raise FileNotFoundError(f"{name}: подключите выводы lab3-00-data и lab3-01-tune")


sys.path.insert(0, _find("marl_env.py"))
import pandas as pd
import marl_algos as ma
import marl_eval as ev
GROUP = "onpolicy"
ALGOS = {"onpolicy": ["mat", "happo", "ippo", "mappo", "coma", "ppo"],
         "offpolicy": ["maddpg", "qmix", "vdn"]}[GROUP]
DATA = str(Path(_find("env_data.npz")) / "env_data.npz")
TUNED = json.loads((Path(_find("tuned.json")) / "tuned.json").read_text())
OUT = ev.results_dir()
RUNS = OUT / "main"
WORKERS = os.cpu_count() or 2
BUDGET = int(os.environ.get("LAB3_BUDGET", 400_000))   # переменная — только для отладки
SEEDS = [int(s) for s in os.environ.get("LAB3_SEEDS", "0,1,2,3,4").split(",")]
print(GROUP, ALGOS, "| процессов:", WORKERS)

# %%
jobs = [{"algo": a, "seed": s, "budget": BUDGET, "data_path": DATA, "P": {}, "cfg": TUNED.get(a, {}),
         "tag": f"main__{a}__s{s}", "parts": ["val", "test", "holdout"], "out": str(RUNS)}
        for a in ALGOS for s in SEEDS]
t0 = time.time()
res = ev.run_many(jobs, workers=WORKERS)
print(f"готово за {(time.time() - t0) / 60:.1f} мин; ошибок: {int((~res['ok'].astype(bool)).sum())}")
for _, r in res[~res["ok"].astype(bool)].iterrows():
    print(r["tag"], r.get("error"), r.get("trace"))

# %%
d = ev.load_runs("main__*.daily.parquet", roots=(str(RUNS),))
t = ev.table(d[d["part"] == "val"])
print("val (для контроля; выбор — в lab3_03):")
print(t[["policy", "gain", "ir", "cvar", "crowded", "turnover", "viol_budget", "seed_sd"]].round(4).to_string(index=False))
print(res[["tag", "train_sec", "best_step", "n_params"]].to_string(index=False))
