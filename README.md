# CDM Supplement — 孔子衣鏡與早期孔子書寫 補充材料

論文《孔子衣鏡與早期孔子書寫——基於文本相似性的比較》，《嶺南學報》復刊第二十五輯（數字人文專號）。

## 版本對照（請先讀）

本倉庫並存兩代材料，**舊材料一律保留、未經刪改**：

| 版本 | 路徑 | 內容 | 對應 |
|---|---|---|---|
| **20260226 原版** | 倉庫根目錄各處（`text data`、`CDM data`、`code pipe`、`experiment data`、`data analysis report`、`figure` 等） | 完整材料包：文本語料、元數據（照片／摹本／釋文／字典）、`pipeline20.py`／`small pipe 4.py`、2026-02 各次實驗運行、刊面圖表 | **刊面所據**——已刊論文 的數值與圖表出自此版。快照見 tag `v20260226` |
| **20260805 刊後更新** | [`revised 2026-08/`](./revised%202026-08/) | 受刊後勘誤影響而更新者：`pipeline21.2.py`／`small pipe 5.2–5.4.py`、5.4 正典運行 202608041925 全套輸出、替換圖版、四份修訂報告、`CDM_DICT_3.1` | **更正後**——Ward 距離定義、PCA 中心化與確定性復現、Reranker 加載修復、校勘閾值 norm_edit ≤ 0.35 等 |

> **取用原則：若更新包與刊面數據不符，以 [`revised 2026-08/`](./revised%202026-08/) 及根目錄[《更新説明_20260805.md》](./更新説明_20260805.md)爲準。**
> 《更新説明》以「改了什麼、爲什麼、影響邊界」爲綱逐條記録本次全部更正。

**取用提示**：本倉庫 20260226 舊包的 `*.xlsx`／`*.json`／`*.npy` 由 Git LFS 託管，
用「Download ZIP」下載會得到指針檔而非真實內容；請 `git lfs install` 後 clone，
或按需 `git lfs pull --include="<path>"`。`revised 2026-08/` 內各檔不走 LFS，可直接下載。

---

## 以下爲 20260226 原版 README（原文保留）

# CDM-revised-supplement-20260226

## 孔子衣鏡與早期孔子書寫——基於文本相似性的比較（修訂版）

**Lingnan Journal of Chinese Studies — Supplementary Revised Materials**

**嶺南學報 — 數字人文論文補充材料**

---

## Text Sources 電子文本來源

本文數字實驗所用電子文本來自：

1. 學衡數據 — 中文核心典籍 CCT (Chinese Core Texts)：http://core.xueheng.net/
2. 中國哲學書電子計劃 Chinese Text Project：https://ctext.org

---

## Materials Overview

本材料集為論文的完整開源附錄，包含文本語料、實驗代碼、實驗原始數據、數據分析報告、論文圖表，以及《孔子衣鏡》元數據。

| Directory | Contents | Size |
|-----------|----------|------|
| `text data` (文本語料) | Experiment corpora (Excel / Word) | ~1.5 MB |
| `code pipe` (實驗代碼) | Detection pipeline + visualization scripts | ~596 KB |
| `experiment data` (實驗數據) | Main + control + ablation experiment outputs | ~1.7 GB |
| `data analysis report` (數據分析與統計報告) | Experiment reports + statistical tables | ~21 MB |
| `figure` (論文圖表) | Paper figures and tables (DPI=600) | ~15 MB |
| `CDM data` (《孔子衣鏡》元數據) | Photographs, facsimiles, transcriptions, dictionary, phonetic-loan annotations | ~124 MB |

For detailed file descriptions, see `Supplementary Materials Guide.docx`.

---

## Citation

```
孔子衣鏡與早期孔子書寫——基於文本相似性的比較.《嶺南學報》數字人文專號.
```

## License

This material set is intended solely for academic research. Please cite the paper when using these materials.

本材料集僅供學術研究使用，使用時請引用本論文。
