# Reading Notes

This paper addresses a fundamental limitation of the encoder–decoder architecture used in neural machine translation. In the original sequence-to-sequence framework, the encoder compresses the entire source sentence into a single fixed-length hidden vector. Empirically, translation accuracy decreases as sentence length increases. The core issue is the fixed-length bottleneck: all information from the source sentence must be compressed into one vector regardless of sentence length. The authors hypothesize that this structural limitation, rather than merely insufficient model capacity, is responsible for the degradation in performance on long sentences.

To address this, the authors introduce the attention mechanism. Instead of encoding the entire sentence into a single vector, the encoder produces a sequence of hidden states. During decoding, the model dynamically computes a context vector for each target word by weighting these annotations. The encoder uses a bidirectional RNN with GRU cells, producing forward and backward hidden states that are concatenated into annotations. The alignment model is implemented as a small MLP that learns a compatibility function between the current decoder state and each encoder annotation. Softmax is then applied to obtain normalized attention weights, and the context vector is computed as a weighted sum of annotations. The entire model is differentiable and trained end-to-end.

The alignment model uses a learnable MLP because the compatibility between a decoder query and an encoder key is not fixed. An MLP with a non-linear activation can capture asymmetric and context-dependent relationships that a simple dot product cannot. GRU cells in both the encoder and decoder provide sequential memory through update and reset gates without maintaining a separate cell state, making them computationally lighter than LSTM while still capable of modeling long-range dependencies. The bidirectional encoder ensures that each annotation captures context from both directions of the source sentence.

The contribution of this paper is not the invention of GRU or BiRNN individually, but their integration with a learnable alignment mechanism that solves the fixed-length bottleneck. The key insight is that attention changes the information access pattern from a single static compressed vector to a dynamic retrieval over all source positions. This is a qualitative change in architecture, not merely a quantitative increase in capacity.

---

# Learning From AI (GPT5.2)

### 1️⃣ Bottleneck 是結構問題，不是容量問題

原始 encoder–decoder 把整句壓縮進一個 fixed-length vector `c`。
句子越長，壓縮越激進，這是 **representation bottleneck**，不是 hidden size 不夠大的問題。

👉 增大 hidden size 不能解決結構性瓶頸。

---

### 2️⃣ Attention 改變的是資訊存取機制，不是容量

原始模型：decoder 每一步都 condition 在同一個靜態向量。
Attention：decoder 每一步動態計算新的 context vector。

👉 這是機制的改變，不是參數量的改變。

---

### 3️⃣ Encoder 輸出從單一向量變成記憶庫

原始：

$$c = h_T$$

Attention：

$$h_1, h_2, ..., h_T$$

每個 encoder hidden state 都成為可讀取的記憶單元。

👉 Decoder 不再只能讀一個壓縮記憶，而是動態查詢整個序列。

---

### 4️⃣ Alignment Model 是可學習的匹配函數

對齊分數：

$$e_{ij} = v^\top \tanh(W s_{i-1} + U h_j)$$

這是一個小型 MLP。
它學習 decoder state（query）與 encoder state（key）之間的相容性。

👉 Attention 的「對齊」是從資料中學出來的，不是人工設計的規則。

---

### 5️⃣ Softmax 讓對齊可微分、可訓練

對齊權重：

$$\alpha_{ij} = \text{softmax}(e_{ij})$$

這使得 attention：

* 可微分
* 可端對端訓練
* 能同時關注多個 source 位置（soft alignment）

👉 Softmax 是讓 attention 從「潛在概念」變成「可訓練機制」的關鍵。

---

### 6️⃣ Context Vector 是加權檢索

$$c_i = \sum_j \alpha_{ij} h_j$$

不是選一個詞，而是對所有 source 位置做加權平均。
每個 decoding step 得到不同的 context。

👉 這讓 decoder 在翻譯每個詞時，都能「看」到最相關的 source 位置。

---

### 7️⃣ GRU 解決時間記憶，Attention 解決空間存取

GRU：處理序列內的長期依賴（時間方向）。
Attention：選擇性存取 source 位置（空間方向）。

👉 兩者解決不同的瓶頸，可以組合。

---

### 8️⃣ 雙向 Encoder 讓每個 annotation 包含全局上下文

前向 RNN：$\overrightarrow{h_j}$（讀取左側上下文）
後向 RNN：$\overleftarrow{h_j}$（讀取右側上下文）

串接：$h_j = [\overrightarrow{h_j}; \overleftarrow{h_j}]$

👉 每個 source 位置的 annotation 都同時捕捉了左右文脈。

---

### 9️⃣ 整個模型是端對端可訓練的

梯度從：

$$\text{Loss} \rightarrow \text{Output} \rightarrow \text{Decoder} \rightarrow \text{Attention} \rightarrow \text{Encoder} \rightarrow \text{Embedding}$$

沒有任何固定或啟發式的部分。
對齊與翻譯是聯合學習的。

👉 Attention weights 不是人工設計，而是為了最大化翻譯品質而學到的。

---

### 🔟 研究邏輯：從失敗模式出發，重新設計功能

這篇論文不是從數學出發，而是從一個觀察出發：長句子的翻譯效果下降。

提問：「缺少什麼功能？」

回答：解碼時需要動態存取 source 的任意位置。

然後用可微分元件實作這個功能。

👉 好的架構創新往往從識別資訊流瓶頸開始，而不是從增加參數開始。

---

### 總結一句話

Attention 的本質是：

> 把固定的靜態記憶讀取，改為動態的加權查詢。

這是 capacity 的量變，更是 mechanism 的質變。

**ChatGPT 對話連結**: https://chatgpt.com/share/699d6a0c-afbc-800a-88d6-a342ccb369a5
