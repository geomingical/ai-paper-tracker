# Reading Notes

The paper develops the Brook environment, which aims to enable general-purpose computation on GPUs without requiring programmers to directly use graphics APIs, by expressing computation using streams and kernels. The authors use multiple benchmarks (SAXPY, Segment [image segmentation], SGEMV, FFT, and ray tracing) to compare the performance of GPU and CPU implementations. Through these experiments, they analyze what kinds of computation run well on GPUs and what the limitations are.

From prior work and the paper's introduction, GPUs were originally designed to handle repeated tasks for graphics. However, using GPUs for general-purpose computation required programmers to understand many graphics-specific details and APIs, which created a high barrier to entry. Therefore, Brook, as a programming environment, aims to reduce this burden and make GPU computing more accessible.

This also makes me think about systems like TPU and compiler stacks such as MLIR: specialized hardware often benefits from a programming model and compiler/runtime support that hides low-level details. (Note: this is my own extension beyond the Brook paper.) Regarding MLIR, I think compiler infrastructure can help integrate multiple sources and targets, but there can be tension between openness and the need for private, application-specific optimizations.

**Reference**: https://www.cnblogs.com/wujianming-110117/p/18845720

---

# Learning From AI (GPT5.2)

## 1️⃣ 問題的本質：為什麼一直出現新語言 / 新系統？

隨著計算場景越來越多（圖形、AI、資料分析、機器人），「計算型態」開始分化：
有些計算需要複雜邏輯（控制、決策），有些計算是大量重複、結構固定（矩陣、影像、神經網路）。
單一 CPU 已無法在效能與能效上同時滿足所有需求，因此出現 GPU、TPU 等專用或半專用硬體。

真正的問題不是「算不算得出來」，而是：
👉 如何用「人能理解的方式」正確地驅動「結構高度專用的硬體」？

## 2️⃣ C 與 Python：控制 vs 意圖

**C 的角色：**
C 是「最低階控制語言」，你必須明確寫出 怎麼做。
它適合表達硬體導向、效能敏感的細節，但對人類不友善。

**Python 的角色：**
Python 是「高階意圖描述語言」，你主要表達 要做什麼。
它本身慢，但能快速組合各種 domain library。
實際效能幾乎永遠來自 底層的 C / C++ / GPU / TPU。

👉 Python 的成功，不是因為快，而是因為它成功成為所有計算場景的共同入口。

## 3️⃣ Library vs Compiler：為什麼 Brook / CUDA 不是「套件」？

**Library（套件）：**
- 你呼叫它
- 它不改變你的程式結構
- 編譯器不知道你在做什麼

**Compiler / 語言系統：**
- 它理解你的「計算語意」
- 它會重寫、重排、融合你的程式
- 它決定「怎麼映射到硬體」

👉 Brook、CUDA、XLA、MLIR 的共同點：
它們不是幫你算，而是幫你決定「怎麼算」

## 4️⃣ Brook vs CUDA：理想化抽象 vs 工程化控制

**Brook：**
- 提供「理想化的資料平行模型」（stream / kernel）
- 使用者不必理解 GPU 細節
- 由 compiler / runtime 負責 mapping
- 是「語意驅動」的設計

**CUDA：**
- 提供直接但結構化的 GPU 控制方式
- 使用者需理解 thread / block / memory
- 效能極高，但門檻高
- 是「機器驅動」的設計

👉 Brook 想解的是「使用方式」，CUDA 解的是「工程可行性」。

## 5️⃣ Compiler 與 Runtime 的分工（關鍵分界）

**Compiler（事前規劃）**
- 分析整個計算
- 重排運算順序
- 決定 kernel 形狀、資料流
- 產生適合硬體的低階表示

**Runtime（執行當下調度）**
- 配置記憶體
- 管理資料搬移
- 啟動 kernel
- 處理同步、錯誤、資源競爭

👉 Compiler 決定「怎麼跑」，runtime 決定「現在怎麼跑得起來」

## 6️⃣ XLA / MLIR：為什麼你看不到，但它每次都在動？

在現代系統（如 PyTorch / TensorFlow）中：
1. 使用者用 Python 描述高階運算
2. 系統建立「計算圖」（不是立刻算）
3. XLA / MLIR 取得整個計算結構
4. 進行融合（fusion）、重排、硬體對映
5. 產生 GPU / TPU 專用程式
6. Runtime 執行

👉 XLA / MLIR 的價值在於：
它們看的是「整塊計算」，而不是一行一行程式碼

## 7️⃣ 為什麼新語言看起來很亂，但仍然必要？

**短期看：**
新語言、新抽象 → 學習成本高 → 混亂

**長期看：**
- 舊語言無法表達新硬體的語意
- Compiler 需要新的資訊才能最佳化
- 成功的抽象，最後都會「被藏起來」

👉 Brook 沒成為主流語言，但它的思想已被吸收到 CUDA / MLIR / AI 系統中。

## 8️⃣ Brook 在今天還有機會嗎？

- 作為「一個新語言」：不太可能
- 作為「一個抽象思想」：已全面成功

在今天，Brook 會以以下形式存在：
- Python decorator
- 中介 IR
- Compiler pass
- Runtime policy

👉 現代系統不再要求使用者學新語言，而是把新抽象藏在系統底層

## 9️⃣ 一句總結（可當你的核心心法）

> 現代計算系統的關鍵，不是再發明一個語言，而是把正確的計算抽象放在對的層級，讓使用者不用知道它存在，也能驅動高度專用的硬體。

**ChatGPT 對話連結**: https://chatgpt.com/share/697b0e95-6880-800a-8313-2099f7c7a90f
