# CDM-revised-supplement-20260805

## 孔子衣鏡與早期孔子書寫——基於文本相似性的比較（刊後在線更新版）

**Lingnan Journal of Chinese Studies — Supplementary Revised Materials (Post-publication Update)**

**嶺南學報 — 數字人文論文補充材料（刊後更新）**

論文刊於《嶺南學報》復刊第二十五輯。

---

## 本包性質

本包爲 **20260226 版補充材料包的刊後在線更新**（2026-08-03 初修；08-04 續修；08-05 定稿），僅收錄受本次勘誤影響而更新的文件：

1. **算法表述修正**：Ward 層次聚類距離定義、PCA 中心化與確定性復現（詳見《更新説明_20260805.md》第一節）；
2. **其他勘誤**：idf 平滑式、Coverage 定義、robust_soft 仿射步、校勘閾值 norm_edit ≤ 0.35、表 1 Reranker 列刪除、表 3 兩表併一（解釋文字移作腳注）、校勘記重複條目改爲逐字位原始條目（合併爲可選開關）（詳見《更新説明_20260805.md》第 3–6 條）；
3. **5.2→5.4 沿革與圖版替換**：`small pipe 5.4.py` 爲最終版（5.3＋PCA 標籤避讓，數值零改動）（5.2 基礎上圖版文字修訂＋PCA 説明框與右緣長標籤修復＋校勘記「凡N見」合併——後改爲**可選開關、默認逐字位原始條目**，算法與數值零改動）；圖 3 升至 v5（標籤避讓）、圖 4／5 爲 v4（08-04 續修：PCA 説明框刪投影質量套語，解釋方差比仍標於座標軸）；圖 2 網絡圖**保留原 ctext 圖版（作者裁定）**，刊面節點文字一仍其舊，本次不作更正，不入本包（詳見《更新説明_20260805.md》第 6–8 條）。

**本包爲瘦身包，不保留與舊包同構的空鏡像目錄。其餘一切材料（text data 文本語料、
CDM data 元數據、data analysis report 之數據統計、其餘實驗報告與 20260226 版實驗數據）
沿用 20260226 包，本包不重複收錄——請取用本倉庫 20260226 對應 tag／發佈中的舊包。**

---

## Text Sources 電子文本來源

本文數字實驗所用電子文本來自：

1. 學衡數據 — 中文核心典籍 CCT (Chinese Core Texts)：http://core.xueheng.net/
2. 中國哲學書電子計劃 Chinese Text Project：https://ctext.org

---

## Materials Overview（本包實收內容）

| Directory | Contents |
|-----------|----------|
| `code pipe(實驗代碼）` | **只收與定稿對應之最終版**：`small pipe 5.4.py`（小樣本代碼終版；校勘記默認逐字位原始條目，「凡N見」合併爲可選開關 `merge_repeated_variants`）、`run_small_pipe54_headless.py`（無頭運行器，本包正典運行 202608041925 即由此產出）、`pipeline21.2.py`（Reranker 正確實現版主管線；校勘記開關同上）、`requirements_py313_freeze.txt`、`环境配置说明_20260802.md`。中間工作版 5／5.1／5.2／5.3 及 pipeline21／21.1 不隨包發佈，沿革見《更新説明》第 8 條 |
| `figure (論文圖表）` | 圖 3 替換 v5 版（5.4 生成，取自本包正典運行 202608041925 之 `_h2_PCA_TF-IDF.png`，逐位元相同）、圖 4／圖 5 替換 v4 版（5.3 生成；與 5.4 正典運行同名輸出逐位元相同——5.4 僅改 PCA 標籤位置，對此二圖零改動）；圖 2 保留原 ctext 圖版（作者裁定）、表 3 改排爲刊面 Word 內置表格（解釋文字移作腳注），二者均不出圖版 |
| `experiment data(實驗數據)` | `202608041925＋small pipe 5.4＋h2＋小樣本實驗-孔子衣鏡（歸一化版32篇）＋Lexical`：5.4 終版正典運行全套輸出，共 15 檔（Excel 六表、9 張圖、主結果報告 docx、聚類與中心性分析報告、成對相似度權重表、預處理監測、運行日誌）；Excel 六表與 5.3 之 202608041704、5.2 之 202608021346 兩次運行逐值零差異，5.4 僅改 PCA 標籤位置 |
| `data analysis report (數據分析與統計報告）/實驗報告` | 受勘誤影響之四份報告的修訂版（改動標紅）：文本層次最優策略分析報告（融合H2）、大規模實驗報告、自動校勘图表 總、CDM 自动校勘图表_14对 |
| 根目錄 | 本 README、《更新説明_20260805.md》、Supplementary Materials Guide(材料說明) 2026-08-05 修訂版（改動標紅）、`CDM_DICT_3.1(《孔子衣鏡》字典映射）.docx / .pdf`（**嚴式判據 v3.1**：嚴式異文判定所依據之字典映射） |
| （未收錄目錄） | text data（文本語料）、CDM data（元數據）、data analysis report 之數據統計：沿用 20260226 包（見倉庫對應 tag／發佈），本包不設空目錄 |

運行環境：python3.13（依賴凍結見 `requirements_py313_freeze.txt`，配置步驟見 `环境配置说明_20260802.md`）。

---

## Citation

```
孔子衣鏡與早期孔子書寫——基於文本相似性的比較.《嶺南學報》復刊第二十五輯（數字人文專號）.
```

## License

This material set is intended solely for academic research. Please cite the paper when using these materials.

本材料集僅供學術研究使用，使用時請引用本論文。
