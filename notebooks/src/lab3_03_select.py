# %% [markdown]
# # ЛР3 · lab3_03 — выбор алгоритма по val, затем главная таблица на test (один раз)
#
# Задания 3 и 5 (протокол: разд. 6–8, 11). **Accelerator:** нет. **Internet:** нет. Время: ~10 мин.
# **Inputs:** выводы `lab3-00-data`, `lab3-02-main-onpolicy`, `lab3-02-main-offpolicy`.
#
# Порядок важен: сначала правило выбора на val → `selection.json` (с временем), и только потом test.
# **Output:** `selection.json`, `table_val.csv`, `table_test.csv`, `table_holdout.csv`, `pairs_test.csv`,
# `cost.csv`, `fig_curves.png`.

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
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import marl_algos as ma
import marl_eval as ev
from marl_env import EnvData, EnvParams
DATA = str(Path(_find("env_data.npz")) / "env_data.npz")
data = EnvData.load(DATA)
OUT = ev.results_dir()
daily = ev.load_runs("main__*.daily.parquet")
print("прогонов:", daily["tag"].nunique(), "| алгоритмы:", sorted(daily["policy"].unique()))
base = ev.baseline_frames(data, EnvParams())
base.to_parquet(OUT / "baselines.daily.parquet", index=False)
allf = pd.concat([daily, base], ignore_index=True)
COLS = ["policy", "gain", "ci_lo", "ci_hi", "ir", "sortino", "worst", "cvar", "cost_per_pax", "wait_per_pax",
        "crowded", "breach", "turnover", "used_frac", "viol_budget", "clip_share", "cap_share", "seed_sd", "n_seeds"]

# %% [markdown]
# ## 1. Выбор по val (правило разд. 7)

# %%
val = allf[allf["part"] == "val"]
tv = ev.table(val)
tv[["ci_lo", "ci_hi"]] = [ev.boot_ci(val[val["policy"] == p]) for p in tv["policy"]]
cand = tv[tv["policy"].isin(ma.ALGOS + ma.BASELINES_LEARNED)].reset_index(drop=True)
cand["score"] = ev.composite(cand)
sel = ev.select(tv[tv["policy"].isin(ma.ALGOS)].reset_index(drop=True), val[val["policy"].isin(ma.ALGOS)],
                ma.SIMPLICITY)
best = sel["choice"]
selection = {"choice": best, "reason": sel["reason"], "time": pd.Timestamp.now().isoformat(),
             "ranking": sel["table"][["policy", "score", "gain", "ci_lo", "admissible"]].to_dict("records"),
             "ppo_score_vs_marl": cand[["policy", "score"]].to_dict("records")}
(OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("ВЫБРАН:", best, "—", sel["reason"])
print(sel["table"][["policy", "score", "gain", "ci_lo", "ir", "cvar", "crowded", "turnover", "viol_budget",
                    "seed_sd", "admissible"]].round(4).to_string(index=False))
tv.merge(cand[["policy", "score"]], how="left").to_csv(OUT / "table_val.csv", index=False)

# %% [markdown]
# ## 2. Главная таблица на test (после выбора)

# %%
def full_table(part):
    f = allf[allf["part"] == part]
    t = ev.table(f)
    t[["ci_lo", "ci_hi"]] = [ev.boot_ci(f[f["policy"] == p]) for p in t["policy"]]
    order = ma.ALGOS + ma.BASELINES_LEARNED + ["schedule", "mpc_m4w", "mpc_norm", "mpc_hyb", "mpc_oracle"]
    t["o"] = t["policy"].map({p: i for i, p in enumerate(order)})
    return t.sort_values("o").drop(columns="o")


tt = full_table("test")
tt.to_csv(OUT / "table_test.csv", index=False)
show = tt[COLS].copy()
for c in ["gain", "ci_lo", "ci_hi", "worst", "cvar", "crowded", "breach", "turnover", "viol_budget", "clip_share",
          "cap_share", "seed_sd"]:
    show[c] = (show[c] * 100).round(2)
print(show.round(3).to_string(index=False))
th = full_table("holdout") if (allf["part"] == "holdout").any() else pd.DataFrame()
if len(th):
    th.to_csv(OUT / "table_holdout.csv", index=False)

# %% [markdown]
# ## 3. Парные сравнения на test (H1, H2, H3, H7)

# %%
te = allf[allf["part"] == "test"]
P_ = lambda p: te[te["policy"] == p]
pairs = []
for name, a, b in [("H1: MAPPO − IPPO", "mappo", "ippo"), ("H2: выбранный − расписание", best, "schedule"),
                   ("H3: выбранный − PPO (один агент)", best, "ppo"), ("H7: выбранный − MPC norm_level", best, "mpc_norm"),
                   ("выбранный − оракул", best, "mpc_oracle"), ("PPO − MPC norm_level", "ppo", "mpc_norm")]:
    if len(P_(a)) and len(P_(b)):
        pairs.append({"сравнение": name, "A": a, "B": b, **ev.paired_diff(P_(a), P_(b))})
pairs = pd.DataFrame(pairs)
pairs.to_csv(OUT / "pairs_test.csv", index=False)
print(pairs.assign(diff=lambda x: (x["diff"] * 100).round(2), lo=lambda x: (x["lo"] * 100).round(2),
                   hi=lambda x: (x["hi"] * 100).round(2)).to_string(index=False))

# %% [markdown]
# ## 4. Вычислительная стоимость и кривые обучения

# %%
runs = pd.concat([pd.read_csv(f) for f in ev.find_all("runs.csv")], ignore_index=True)
runs = runs[runs["tag"].astype(str).str.startswith("main__") & runs["ok"].astype(bool)].drop_duplicates("tag")
cost = runs.groupby("algo").agg(train_min=("train_sec", lambda x: x.mean() / 60), params=("n_params", "first"),
                                best_step=("best_step", "mean")).reset_index()
inf = te.groupby("policy")["infer_ms"].mean().rename("infer_ms")
cost = cost.merge(inf, left_on="algo", right_index=True, how="left")
cost.to_csv(OUT / "cost.csv", index=False)
print(cost.round(2).to_string(index=False))
curves = pd.concat([pd.read_csv(f) for f in ev.find_all("main__*.curve.csv")], ignore_index=True)
curves.to_csv(OUT / "curves_main.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
for algo, g in curves.groupby("algo"):
    m = g.groupby("step")["val_score"].agg(["mean", "std"]).reset_index()
    ax[0].plot(m["step"], m["mean"], label=ma.TITLE.get(algo, algo))
    ax[0].fill_between(m["step"], m["mean"] - m["std"], m["mean"] + m["std"], alpha=0.15)
    m2 = g.groupby("step")["train_reward"].mean().reset_index()
    ax[1].plot(m2["step"], m2["train_reward"], label=ma.TITLE.get(algo, algo))
for a, t in zip(ax, ["val: средняя награда (детерминированно), ± ст. откл. по seed", "train: награда эпизода (с исследованием)"]):
    a.axhline(-24, color="grey", ls=":", lw=1)
    a.set_title(t)
    a.set_xlabel("шагов среды")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_curves.png", dpi=120)
