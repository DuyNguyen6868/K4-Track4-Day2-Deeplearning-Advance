"""evalref.py - nạp `eval.py` GỐC của repo (không sửa, không chép) dưới tên module `eval`.

Bài nộp nằm trong `submissions/<mssv>_<ten>/code/`, còn `eval.py` nằm ở gốc repo. Module này đi ngược lên
các thư mục cha để tìm nó, hoặc dùng biến môi trường LAB_EVAL_PY nếu bạn đặt `eval.py` chỗ khác.

    from evalref import ev
    ev.compute_metrics(...); ev.save_predictions(...); ev.load_group(...)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def find_eval_py() -> Path:
    env = os.environ.get("LAB_EVAL_PY")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        cand = parent / "eval.py"
        if cand.is_file() and "def compute_metrics" in cand.read_text(encoding="utf-8"):
            return cand
    raise FileNotFoundError("Không tìm thấy eval.py gốc của repo. Đặt biến môi trường LAB_EVAL_PY=<đường dẫn>.")


def _load():
    if "eval" in sys.modules and hasattr(sys.modules["eval"], "compute_metrics"):
        return sys.modules["eval"]
    path = find_eval_py()
    spec = importlib.util.spec_from_file_location("eval", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval"] = mod
    spec.loader.exec_module(mod)
    return mod


ev = _load()
EVAL_PY_PATH = find_eval_py()
