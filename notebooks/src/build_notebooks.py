"""Собирает .ipynb из notebooks/src/*.py (ячейки разделены «# %%»).

Запуск из корня репозитория: python notebooks/src/build_notebooks.py
lab3_00_data получает модули marl_*.py ячейками %%writefile — дальше они едут в выводе ноутбука.
lab3_02_main собирается в двух вариантах: onpolicy и offpolicy (можно запускать параллельно).
"""
import json
import re
from pathlib import Path

SRC = Path(__file__).parent
DST = SRC.parent
ROOT = DST.parent
MODULES = ["marl_env.py", "marl_data.py", "marl_algos.py", "marl_eval.py"]


def cells_from(text: str):
    cells = []
    for chunk in re.split(r"^# %%", text, flags=re.M):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            body = chunk[len(" [markdown]"):].strip("\n")
            lines = [re.sub(r"^# ?", "", ln) for ln in body.splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            code = chunk.strip("\n")
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code})
    for c in cells:
        src = c["source"]
        c["source"] = [ln + "\n" for ln in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    return cells


def write(name: str, text: str):
    nb = {"cells": cells_from(text),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (DST / name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", DST / name)


for f in sorted(SRC.glob("lab3_*.py")):
    text = f.read_text(encoding="utf-8")
    if f.stem == "lab3_00_data":
        first, rest = text.split("\n# %%\n", 1)
        kit = "".join(f"\n# %%\n%%writefile /kaggle/working/{m}\n" + (ROOT / m).read_text(encoding="utf-8").rstrip("\n") + "\n"
                      for m in MODULES)
        text = first + kit + "\n# %%\n" + rest
    if f.stem == "lab3_02_main":
        for g in ["onpolicy", "offpolicy"]:
            write(f"lab3_02_main_{g}.ipynb", text.replace('GROUP = "onpolicy"', f'GROUP = "{g}"', 1))
    else:
        write(f"{f.stem}.ipynb", text)
