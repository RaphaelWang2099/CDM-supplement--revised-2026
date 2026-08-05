# -*- coding: utf-8 -*-
"""无头运行 small pipe 5（不启动 Tkinter 界面）。

用途：以 2026-02-16 主实验（202602162033，歸一化版 32 篇）相同配置重跑
TF 通道（TF-cos / TF-IDF / Jaccard / 網絡 / PCA / Strength / Ward），
其中 Ward 聚類採用 small pipe 5 的歐氏距離修正。

语义与重排关闭（use_semantic=False）：本机（M1 16GB）无 torch 环境，且
Ward 修正不涉及语义通道；語義列數值沿用 2026-02 原实验，不重算。
结果缓存读写均关闭，避免与旧版缓存交叉污染。
"""
import importlib.util
import os
import sys
import time
from datetime import datetime
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "small pipe 5.4.py")
EXCEL = os.path.join(HERE, "小樣本實驗-孔子衣鏡（歸一化版32篇）.xlsx")
FONT = os.path.join(HERE, "fonts", "simhei.ttf")
OUT_BASE = os.path.normpath(os.path.join(HERE, "..", "run output"))

# tkinter 兜底：若运行环境无 tkinter，用假模块顶替（仅模块导入需要）
try:
    import tkinter  # noqa: F401
except Exception:
    for name in ("tkinter", "tkinter.ttk", "tkinter.filedialog",
                 "tkinter.messagebox", "tkinter.font", "tkinter.scrolledtext"):
        sys.modules[name] = MagicMock()

# 预先正常导入 matplotlib（Agg 后端），使 small pipe 5 内部的
# _import_matplotlib_safe 二次导入成为空操作，绕开其 system_profiler
# 补丁在受限环境下返回空 plist 导致的 font_manager 解析崩溃。
import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)
import matplotlib.font_manager  # noqa: E402,F401
import matplotlib.pyplot  # noqa: E402,F401

spec = importlib.util.spec_from_file_location("small_pipe54", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class HeadlessApp(mod.NgramApp):
    """跳过 __init__（不建 Tk 窗口），UI 相关调用一律打桩。"""

    def __init__(self):  # noqa: D401  不调用父类 __init__
        self.output_paths = []
        self.auto_output_paths = {}
        self._progress_total = None
        self._run_started_at = time.perf_counter()
        self._run_started_wall = datetime.now()

    # —— UI 桩 ——
    def _append_log(self, msg="", *a, **k):
        print(f"[log] {msg}", flush=True)

    def _status_set(self, msg="", *a, **k):
        print(f"[status] {msg}", flush=True)

    def _ui_progress(self, done, total, msg="", *a, **k):
        print(f"[progress] {done}/{total} {msg}", flush=True)

    def _post_error(self, title, msg, *a, **k):
        raise RuntimeError(f"{title}: {msg}")

    def _post_info(self, *a, **k):
        pass

    def _set_run_enabled(self, *a, **k):
        pass

    def _clear_run_log(self, *a, **k):
        pass

    def _reset_progress(self, *a, **k):
        pass

    def __getattr__(self, name):  # 仅在常规查找失败时触发（如 Tk 变量）
        return MagicMock()


config = {
    "excel_path": EXCEL,
    "out_dir": OUT_BASE,
    "n": 3,
    "topk_edge": 10,
    "granularity": "h2",
    "show_examples": True,
    "strip_label_digits": False,
    "heatmap_font_setting": FONT,
    "heatmap_title": "《論語》《史記》《家語》《衣鏡》（長度歸一化）關係網絡圖",
    "use_semantic": False,
    "embed_model_size": "8B",
    "use_reranker": False,
    "reranker_model_size": "8B",
    "normal_candidate_topk": 20,
    "rerank_top_percent_non_h1": 3.0,
    "use_clustering": True,
    "result_cache_reuse": False,
    "result_cache_write": False,
}

if __name__ == "__main__":
    mod._ensure_ml_stack()  # GUI 流程在启动线程前调用，无头模式需手动补上
    app = HeadlessApp()
    app.run_analysis(config)
    print("== DONE ==")
    for p in getattr(app, "output_paths", []):
        print("OUT:", p)
