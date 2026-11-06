"""Сквозной прогон всех ноутбуков ЛР3 на синтетике (малый бюджет, 2 seed). Загрузки из сети подменены.

python tests/e2e.py /tmp/e2e
Проверяет только, что код работает от начала до конца; цифры синтетики смысла не имеют.
"""
import os
import runpy
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WORK = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/e2e")
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
for m in ["marl_env.py", "marl_data.py", "marl_algos.py", "marl_eval.py"]:
    shutil.copy(REPO / m, WORK / m)
os.chdir(WORK)
sys.path.insert(0, str(WORK))
sys.path.insert(0, str(REPO / "tests"))
os.environ.update({"LAB3_BUDGET_TUNE": "6000", "LAB3_BUDGET": "12000", "LAB3_SEEDS": "0,1"})

import make_synthetic as ms  # noqa: E402
import marl_data as md  # noqa: E402

y, cov, static, trips, durations = ms.make(n_st=120)
y_vkr = y[y.index <= "2026-09-24 23:00"]
y_new = y[y.index > "2026-09-24 23:00"]
# прогноз «гибрида ВКР» в формате ноутбука 16 (шумный факт по станциям)
pd_dir = WORK / "vkr16" / "results" / "preds"
pd_dir.mkdir(parents=True)
rng = np.random.default_rng(3)
Yv = y_vkr.to_numpy()
for part, (a, b) in {"val": ("2026-04-01", "2026-06-30 23:00"), "test": ("2026-07-01", "2026-09-24 23:00")}.items():
    o = y_vkr.index[(y_vkr.index >= a) & (y_vkr.index <= b)]
    pos = y_vkr.index.get_indexer(o)
    yh = np.stack([Yv[np.clip(pos + k, 0, len(Yv) - 1)] for k in range(24)], 1) * rng.lognormal(0, 0.1, (len(o), 24, 1))
    np.savez_compressed(pd_dir / f"hyb_blend__{part}.npz", origins=o.values.astype("datetime64[ns]"),
                        stations=np.array(y.columns, dtype="U"), yhat=yh.astype("float32"))
md.find_vkr_root = lambda: Path("synthetic")
md.load_vkr = lambda root: (y_vkr, cov, static)
md.download_ridership = lambda a, b, st, log=print: y_new.reindex(columns=st)
md.download_trips = lambda a, b, log=print: trips.assign(ts=pd.to_datetime(trips["ts"]))
md.check_trip_duplicates = lambda dates, log=print: pd.DataFrame(
    [{"date": d, "rows": 100, "unique_trips": 100, "dup_share": 0.0, "supplements": 1} for d in dates])
md.download_durations = lambda dates, log=print: durations.assign(date=pd.Timestamp("2024-01-10"))
orig_find = md.find_file
md.find_file = lambda name: next(iter(sorted(WORK.rglob(name))), None)

NB = REPO / "notebooks" / "src"
for nb, extra in [("lab3_00_data", {}), ("lab3_01_tune", {}), ("lab3_02_main", {"GROUP": "onpolicy"}),
                  ("lab3_02_main", {"GROUP": "offpolicy"}), ("lab3_03_select", {}), ("lab3_04_apply", {}),
                  ("lab3_05_summary", {})]:
    t = time.time()
    src = (NB / f"{nb}.py").read_text(encoding="utf-8")
    if "GROUP" in extra:
        src = src.replace('GROUP = "onpolicy"', f'GROUP = "{extra["GROUP"]}"')
    print(f"\n========== {nb} {extra} ==========")
    exec(compile(src, nb, "exec"), {"__name__": "__main__"})
    print(f"========== {nb}: {time.time() - t:.0f} c")
    if os.environ.get("E2E_ONLY") == nb:
        break
print("\nВСЁ ПРОШЛО")
