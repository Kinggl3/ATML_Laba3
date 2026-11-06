# %% [markdown]
# # ЛР3 · lab3_01 — подбор скорости обучения (одинаковый бюджет подбора для всех)
#
# Задание 3 (протокол: разд. 5). **Accelerator:** нет (CPU, процессы по числу ядер). **Internet:** нет.
# Время: ~30–60 мин. **Inputs:** вывод `lab3-00-data`.
#
# Для каждого из 8 MARL и одноагентного PPO — 2 варианта скорости обучения (по умолчанию и /3), seed 100,
# 200 000 шагов. Выбор по лучшей средней награде на val. Остальные гиперпараметры — по умолчанию
# (`marl_algos.DEFAULTS`). **Output:** `tuned.json`, `tune_table.csv`, кривые обучения, журнал `runs.csv`.

# %%
import os, sys, json, time
from pathlib import Path


def _find(name):
    for root in ["/kaggle/input", "/kaggle/working", "."]:
        for dp, dn, fn in os.walk(root, followlinks=True):
            if name in fn:
                return dp
    raise FileNotFoundError(f"{name}: подключите вывод lab3-00-data")


sys.path.insert(0, _find("marl_env.py"))
import numpy as np, pandas as pd
import marl_algos as ma
import marl_eval as ev
DATA = str(Path(_find("env_data.npz")) / "env_data.npz")
OUT = ev.results_dir()
RUNS = OUT / "tune"
WORKERS = os.cpu_count() or 2
BUDGET_TUNE = int(os.environ.get("LAB3_BUDGET_TUNE", 200_000))   # переменная — только для отладки
SEED_TUNE = 100
LR_MULTS = {"lr1": 1.0, "lr3": 1 / 3}
print("данные:", DATA, "| процессов:", WORKERS)

# %%
jobs = []
for algo in ma.ALGOS + ma.BASELINES_LEARNED:
    base = ma.config(algo)
    for key, m in LR_MULTS.items():
        cfg = {k: base[k] * m for k in ("lr", "lr_critic") if k in base}
        jobs.append({"algo": algo, "seed": SEED_TUNE, "budget": BUDGET_TUNE, "data_path": DATA, "P": {},
                     "cfg": cfg, "tag": f"tune__{algo}__{key}", "parts": ["val"], "out": str(RUNS),
                     "n_evals": 8})
# сначала долгие — так процессы загружены равномернее
slow = {"maddpg": 0, "qmix": 1, "mat": 2, "vdn": 3, "happo": 4}
jobs.sort(key=lambda j: slow.get(j["algo"], 9))
t0 = time.time()
res = ev.run_many(jobs, workers=WORKERS)
print(f"готово за {(time.time() - t0) / 60:.1f} мин; ошибок: {int((~res['ok'].astype(bool)).sum())}")
bad = res[~res["ok"].astype(bool)]
for _, r in bad.iterrows():
    print(r["tag"], r.get("error"), r.get("trace"))

# %%
res["lr_key"] = res["tag"].str.split("__").str[-1]
tab = res.pivot_table(index="algo", columns="lr_key", values="best_val_reward")
tab["выбрано"] = tab.idxmax(axis=1)
tab.to_csv(OUT / "tune_table.csv")
print(tab.round(4).to_string())
tuned = {}
for algo, row in tab.iterrows():
    base = ma.config(algo)
    m = LR_MULTS[row["выбрано"]]
    tuned[algo] = {k: base[k] * m for k in ("lr", "lr_critic") if k in base}
(OUT / "tuned.json").write_text(json.dumps(tuned, indent=1), encoding="utf-8")
print(json.dumps(tuned, indent=1))

# %%
curves = pd.concat([pd.read_csv(f) for f in sorted(RUNS.glob("*.curve.csv"))], ignore_index=True)
curves.to_csv(OUT / "tune_curves.csv", index=False)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=True)
for ax, (algo, g) in zip(axes.ravel(), curves.groupby("algo")):
    for tag, c in g.groupby("tag"):
        ax.plot(c["step"], c["val_score"], marker="o", label=tag.split("__")[-1])
    ax.axhline(-24, color="grey", ls=":", lw=1)
    ax.set_title(ma.TITLE.get(algo, algo))
    ax.legend()
fig.suptitle("Подбор: средняя награда на val (−24 — расписание)")
fig.tight_layout()
fig.savefig(OUT / "fig_tune_curves.png", dpi=110)
