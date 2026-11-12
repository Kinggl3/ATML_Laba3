# %% [markdown]
# # ЛР3 · lab3_06 — дополнительная оценка «жёсткий бюджет» (без переобучения)
#
# Отступление №4 в PROTOCOL.md (дополнительный анализ, основные результаты не меняются).
# **Accelerator:** нет. **Internet:** нет. Время: ~20–40 мин.
# **Inputs:** выводы `lab3-00-data`, `lab3-02-main-onpolicy`, `lab3-02-main-offpolicy`, `lab3-03-select`,
# `lab3-04-apply`.
#
# Зачем. В основной оценке бюджет поездо-часов — мягкое ограничение (штраф), и все RL-политики в части суток
# немного его превышают, поэтому ни одна не прошла допуск. Здесь те же обученные политики оцениваются с
# защитным слоем, который не даёт выйти за бюджет: каждый час остаток бюджета минус нормативный минимум на
# оставшиеся часы — это предел текущего часа; надбавки сверх нижних границ урезаются в одной пропорции
# (так же, как лимит парка). Политики этого слоя не видели при обучении — это защита на этапе применения
# (shielding). Расписание и MPC бюджет и так не превышают, слой их не меняет.
#
# **Output:** `hb_table_test.csv`, `hb_table_holdout.csv`, `hb_table_val.csv`, `hb_selection.json`,
# `hb_pairs_test.csv`, `hb_ablations.csv`, `hb_compare.csv`, `hb_report.md`, `hb_*.daily.parquet`.

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
import numpy as np, pandas as pd, torch
import marl_algos as ma
import marl_eval as ev
from marl_env import EnvData, EnvParams, MetroEnv, MPCPolicy, SchedulePolicy
torch.set_num_threads(os.cpu_count() or 2)
DATA = str(Path(_find("env_data.npz")) / "env_data.npz")
data = EnvData.load(DATA)
OUT = ev.results_dir()
T = ma.TITLE
t0 = time.time()


class HardBudgetEnv(MetroEnv):
    """MetroEnv + защитный слой бюджета: за сутки нельзя потратить больше бюджета поездо-часов."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        fl = (self.lo * self.rtt).sum(-1)                                        # (D, 24) минимум часа
        self.reserve_after = torch.flip(torch.cumsum(torch.flip(fl, [1]), 1), [1]) - fl   # минимум часов > h

    def apply_actions(self, actions):
        n, c, clip_flag, cap_flag, cap = super().apply_actions(actions)
        b, h = self.b, self.h
        lo = self.lo[b, h]
        allowed = self.budget[b] - self.used - self.reserve_after[b, h]
        use = (n * self.rtt).sum(-1)
        floor_use = (lo * self.rtt).sum(-1)
        cut = use > allowed + 1e-4
        alpha = ((allowed - floor_use) / (use - floor_use).clamp(min=1e-6)).clamp(0.0, 1.0)
        n = torch.where(cut[:, None], lo + (n - lo) * alpha[:, None], n)
        return n, c, clip_flag, cap_flag | cut, cap


# проверка слоя: расписание и MPC не меняются, перерасхода нет
chk = []
for env_cls in [MetroEnv, HardBudgetEnv]:
    env = env_cls(data, EnvParams())
    for pol in [SchedulePolicy(), MPCPolicy("norm")]:
        f = ev.evaluate(pol, env, data.idx("val"), pol.name, 0, "val")
        chk.append({"среда": env_cls.__name__, "политика": pol.name, "выигрыш": f["gain"].mean(),
                    "перерасход max": f["overrun"].max()})
chk = pd.DataFrame(chk)
print(chk.round(5).to_string(index=False))
assert chk.groupby("политика")["выигрыш"].agg(lambda x: abs(x.max() - x.min())).max() < 1e-4, "слой не должен менять MPC и расписание"

# %% [markdown]
# ## 1. Все обученные политики (основные 9 × 5 seed и абляции) с жёстким бюджетом

# %%
ckpts = ev.find_all("main__*.pt") + ev.find_all("abl__*.pt")
print("чекпойнтов:", len(ckpts))
frames = []
envs = {}
for i, ck in enumerate(ckpts):
    meta = torch.load(ck, weights_only=False)
    P = EnvParams(**{k: (tuple(v) if k == "groups" else v) for k, v in meta["P"].items()})
    key = P.tag()
    if key not in envs:
        envs[key] = HardBudgetEnv(data, P)
    env = envs[key]
    pol = ev.load_policy(ck, env)
    tag = ck.name[:-3]
    name = meta["algo"] if tag.startswith("main__") else f"{meta['algo']}|{tag.split('__')[1]}"
    for part in ["val", "test", "holdout"]:
        days = data.idx(part)
        if len(days):
            f = ev.evaluate(pol, env, days, name, meta["seed"], part)
            f["tag"] = tag
            frames.append(f)
    if i % 10 == 0:
        print(f"  {i + 1}/{len(ckpts)} {tag} {time.time() - t0:.0f} c")
hb = pd.concat(frames, ignore_index=True)
base = ev.baseline_frames(data, EnvParams())                   # расписание и MPC: слой их не меняет
hb_all = pd.concat([hb, base], ignore_index=True)
hb.to_parquet(OUT / "hb_learned.daily.parquet", index=False)

# %% [markdown]
# ## 2. Таблицы: val (выбор по тому же правилу), test, holdout

# %%
ORDER = ma.ALGOS + ["ppo", "schedule", "mpc_m4w", "mpc_norm", "mpc_hyb", "mpc_oracle"]


def full_table(f):
    t = ev.table(f)
    t[["ci_lo", "ci_hi"]] = [ev.boot_ci(f[f["policy"] == p]) for p in t["policy"]]
    t["pax_h_saved"] = t["gain"] * f.groupby("policy")["J_sched"].mean().reindex(t["policy"]).to_numpy() / 60
    t["o"] = t["policy"].map({p: i for i, p in enumerate(ORDER)}).fillna(99)
    return t.sort_values("o").drop(columns="o").reset_index(drop=True)


main_pols = ORDER
tabs = {}
for part in ["val", "test", "holdout"]:
    f = hb_all[(hb_all["part"] == part) & hb_all["policy"].isin(main_pols)]
    if len(f):
        tabs[part] = full_table(f)
        tabs[part].to_csv(OUT / f"hb_table_{part}.csv", index=False)
val = hb_all[(hb_all["part"] == "val") & hb_all["policy"].isin(ma.ALGOS)]
sel = ev.select(tabs["val"][tabs["val"]["policy"].isin(ma.ALGOS)].reset_index(drop=True), val, ma.SIMPLICITY)
best = sel["choice"]
(OUT / "hb_selection.json").write_text(json.dumps(
    {"choice": best, "reason": sel["reason"], "ranking": sel["table"][["policy", "score", "gain", "ci_lo", "viol_budget",
                                                                       "admissible"]].to_dict("records")},
    ensure_ascii=False, indent=1, default=float), encoding="utf-8")
print("выбор при жёстком бюджете:", best, "—", sel["reason"])
print(sel["table"][["policy", "score", "gain", "ci_lo", "viol_budget", "admissible"]].round(4).to_string(index=False))

# %%
te = hb_all[hb_all["part"] == "test"]
P_ = lambda p: te[te["policy"] == p]
sel_soft = json.loads((Path(_find("selection.json")) / "selection.json").read_text())["choice"]
pairs = []
for name, a, b in [("H1: MAPPO − IPPO", "mappo", "ippo"), ("H3: выбранный − PPO", best, "ppo"),
                   ("H7: выбранный − MPC norm_level", best, "mpc_norm"), ("H7: IPPO − MPC norm_level", "ippo", "mpc_norm"),
                   ("выбранный − оракул", best, "mpc_oracle"), ("PPO − MPC norm_level", "ppo", "mpc_norm")]:
    if len(P_(a)) and len(P_(b)):
        pairs.append({"сравнение": name, "A": a, "B": b, **ev.paired_diff(P_(a), P_(b))})
pairs = pd.DataFrame(pairs).drop_duplicates(["A", "B"])
pairs.to_csv(OUT / "hb_pairs_test.csv", index=False)
print(pairs.assign(**{c: (pairs[c] * 100).round(2) for c in ["diff", "lo", "hi"]}).to_string(index=False))

# %% [markdown]
# ## 3. Абляции выбранного в основной части алгоритма — при жёстком бюджете

# %%
mt = te[te["policy"] == sel_soft]
ab = []
for key in ["A1_no_coord", "A2_no_msg", "A3_no_switch", "A4_no_fc", "A5_oracle_fc"]:
    f = te[te["policy"] == f"{sel_soft}|{key}"]
    if not len(f):
        continue
    s = ev.summarize(f)
    pdiff = ev.paired_diff(f, mt)
    ab.append({"вариант": key, "выигрыш, %": round(100 * s["gain"], 2),
               "разница, п.п.": f"{100 * pdiff['diff']:+.2f} [{100 * pdiff['lo']:+.2f}; {100 * pdiff['hi']:+.2f}]",
               "p": round(pdiff["p"], 4), "урезано лимитом парка или бюджета, % ч": round(100 * s["cap_share"], 1)})
ab = pd.DataFrame(ab)
ab.to_csv(OUT / "hb_ablations.csv", index=False)
print(ab.to_string(index=False))

# %% [markdown]
# ## 4. Сравнение с основной оценкой (мягкий бюджет) и сводка в markdown

# %%
soft = ev.load_runs("main__*.daily.parquet")
soft_t = ev.table(soft[soft["part"] == "test"]).set_index("policy")
hard_t = tabs["test"].set_index("policy")
cmp = pd.DataFrame({
    "выигрыш, мягкий бюджет, %": (100 * soft_t["gain"]).reindex(hard_t.index),
    "выигрыш, жёсткий бюджет, %": 100 * hard_t["gain"],
    "перерасход > 0,5%, мягкий, % сут": (100 * soft_t["viol_budget"]).reindex(hard_t.index),
    "перерасход > 0,5%, жёсткий, % сут": 100 * hard_t["viol_budget"],
    "поездо-ч, % бюджета (жёсткий)": 100 * hard_t["used_frac"],
}).round(2)
cmp.index = [T.get(p, p) for p in cmp.index]
cmp.to_csv(OUT / "hb_compare.csv")
print(cmp.to_string())
t = tabs["test"]
t1 = pd.DataFrame({"Алгоритм": t["policy"].map(lambda p: T.get(p, p)),
                   "Выигрыш (95% ДИ)": [f"{100 * g:+.2f}% [{100 * a:+.2f}; {100 * b:+.2f}]" for g, a, b in zip(t["gain"], t["ci_lo"], t["ci_hi"])],
                   "IR": t["ir"].round(2), "CVaR 5%": (100 * t["cvar"]).round(2), "Худшие сутки, %": (100 * t["worst"]).round(2),
                   "Переполнение, %": (100 * t["crowded"]).round(2), "Оборот, %": (100 * t["turnover"]).round(1),
                   "Перерасход, % сут": (100 * t["viol_budget"]).round(1), "Урезано лимитом парка или бюджета, % ч": (100 * t["cap_share"]).round(1),
                   "Пасс.-ч/сутки": t["pax_h_saved"].round(0).astype(int),
                   "По seed": [f"{100 * g:+.2f} ± {100 * s:.2f}" if n > 1 else "—" for g, s, n in zip(t["gain"], t["seed_sd"], t["n_seeds"])]})
econ = t[t["policy"].isin([best, sel_soft, "ppo", "mpc_norm", "mpc_hyb", "mpc_oracle"])].drop_duplicates("policy")
econ_md = pd.DataFrame({"политика": econ["policy"].map(lambda p: T.get(p, p)), "выигрыш, %": (100 * econ["gain"]).round(2),
                        "статистически": np.where(econ["ci_lo"] > 0, "значим", "незначим"),
                        "экономически (≥ 1%, перерасход ≤ 0,5%)": np.where((econ["gain"] >= 0.01) & (econ["viol_budget"] <= 0.02), "значим", "мал"),
                        "пасс.-ч в сутки": econ["pax_h_saved"].round(0).astype(int)})
MD = ["# ЛР3. Дополнительная оценка: жёсткий бюджет (lab3_06)", "",
      "Те же чекпойнты, что в основной части; защитный слой не даёт выйти за суточный бюджет поездо-часов. "
      "Расписание и MPC слой не меняет.", "",
      f"Выбор по тому же правилу (val): **{T.get(best, best)}** — {sel['reason']}. В основной части: {T.get(sel_soft, sel_soft)}.", "",
      "## Таблица Б1. Test при жёстком бюджете", "", ev.to_md(t1), "",
      "## Таблица Б2. Мягкий против жёсткого бюджета (test)", "", ev.to_md(cmp.reset_index().rename(columns={"index": "Алгоритм"})), "",
      "## Таблица Б3. Парные сравнения (test, жёсткий бюджет)", "",
      ev.to_md(pairs.assign(**{c: (pairs[c] * 100).round(2) for c in ["diff", "lo", "hi"]})), "",
      f"## Таблица Б4. Абляции {T.get(sel_soft, sel_soft)} при жёстком бюджете (test)", "", ev.to_md(ab) if len(ab) else "нет", "",
      "## Таблица Б5. Статистический против экономического эффекта (test, жёсткий бюджет)", "", ev.to_md(econ_md), ""]
if "holdout" in tabs:
    h = tabs["holdout"]
    h = h[h["policy"].isin([best, sel_soft, "ppo", "schedule", "mpc_norm", "mpc_oracle"])].drop_duplicates("policy")
    MD += ["## Таблица Б6. Holdout при жёстком бюджете", "", ev.to_md(pd.DataFrame({
        "политика": h["policy"].map(lambda p: T.get(p, p)),
        "выигрыш (95% ДИ)": [f"{100 * g:+.2f}% [{100 * a:+.2f}; {100 * b:+.2f}]" for g, a, b in zip(h["gain"], h["ci_lo"], h["ci_hi"])],
        "IR": h["ir"].round(2), "перерасход, % сут": (100 * h["viol_budget"]).round(1)})), ""]
(OUT / "hb_report.md").write_text("\n".join(MD), encoding="utf-8")
print("\n".join(MD))
print(f"готово за {(time.time() - t0) / 60:.1f} мин")
