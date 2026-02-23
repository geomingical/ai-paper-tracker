# Reading Notes

The 2014 Sequence to Sequence Learning with Neural Networks paper proposes an end-to-end neural approach for machine translation. The model consists of two separate LSTMs: an encoder and a decoder. The encoder reads the source sentence (English) sequentially and compresses the entire sentence into a fixed-dimensional vector representation, defined as the final hidden state of the top LSTM layer. In the experiments, the authors used a deep LSTM with four stacked layers, and each layer contained 1000 hidden units. This means that at each time step, the representation passes vertically through four LSTM layers before moving to the next word in the sequence.

The resulting fixed-dimensional vector (often denoted as 𝑣) is then passed to a second deep LSTM, which serves as the decoder. The decoder generates the translated sentence one word at a time. At each decoding step, a softmax layer is applied to produce a probability distribution over the target vocabulary. Training is performed using supervised learning on parallel corpora, where each English sentence is paired with its correct French translation. The objective is to maximize the conditional probability of the correct target sentence given the source sentence.

Inside each LSTM unit, there are three gates: the forget gate, input gate, and output gate. These gates are computed using sigmoid activation functions (not softmax), which produce values between 0 and 1. These values control how much past information is retained, how much new information is written into memory, and how much internal memory is exposed as output. All gate parameters, along with other weight matrices in the model, are learned through backpropagation.

The overall architecture allows the model to capture sequential dependencies in language by maintaining a controlled memory state across time. The encoder transforms a variable-length sentence into a continuous representation, and the decoder conditions on this representation to generate a grammatically coherent translation. This design was significant at the time because it demonstrated that a purely neural network–based system could outperform traditional phrase-based statistical machine translation systems.

---

# Learning From AI (GPT)

### 1️⃣ LSTM 的核心更新公式（最重要）

記憶槽更新：

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

輸出狀態：

$$h_t = o_t \odot \tanh(c_t)$$

* $f_t$：忘記門（保留多少舊記憶）
* $i_t$：輸入門（寫入多少新資訊）
* $o_t$：輸出門（對外暴露多少）

---

### 2️⃣ 三個 Gate 的數學形式

每個 gate 都是：

$$\sigma(Wx_t + Uh_{t-1} + b)$$

* sigmoid 產生 0~1 連續開關
* 每一維都是獨立控制的比例因子

---

### 3️⃣ 長期記憶的數學來源

關鍵在於加法路徑：

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

若 $f_t \approx 1$：

* 記憶幾乎完整傳遞
* 梯度幾乎不衰減

這就是 LSTM 抗梯度消失的核心原因。

---

### 4️⃣ 極端情況理解

* 若 $f_t = 1, i_t = 0$

  * 記憶完全保留
  * 梯度完全通過

* 若 $f_t = 0, i_t = 1$

  * 舊記憶清空
  * 梯度斷裂

這表示 LSTM 可以「保留」或「重設」狀態。

---

### 5️⃣ Hidden Size 是什麼？

$$h_t, c_t \in \mathbb{R}^h$$

* h = hidden dimension
* 表示同時存在多少「記憶通道」
* 每一維學習不同時間尺度

本論文：每層 1000 cells。

---

### 6️⃣ Hidden Dimension 的真正意義

每個維度近似：

$$c_t^{(k)} \approx \lambda_k c_{t-1}^{(k)} + ...$$

不同 $\lambda_k$ = 不同時間常數。

hidden size = 可同時存在多少種時間尺度。

---

### 7️⃣ 為什麼 h_t 不等於 c_t？

$$h_t = o_t \odot \tanh(c_t)$$

原因：

* c_t 是內部記憶
* h_t 是對外投影
* tanh 保持數值穩定（限制在 -1~1）
* output gate 控制暴露比例

---

### 8️⃣ 多層 LSTM 是什麼？

若為 4-layer LSTM：

$$h_t^{(l)} = LSTM^{(l)}(h_t^{(l-1)}, h_{t-1}^{(l)})$$

* 時間方向：sequence depth
* 垂直方向：representation depth

深層增加表示抽象能力。

---

### 9️⃣ Seq2Seq 架構核心

Encoder：

$$x_1,...,x_T \rightarrow v = h_T$$

Decoder：

$$p(y_1,...,y_{T'}|x) = \prod_{t=1}^{T'} p(y_t|y_{<t}, v)$$

* 最後 hidden state 作為句向量 v
* Decoder 是條件語言模型

---

### 🔟 Capacity 與退化問題

若所有維度的 forget gate 相同：

$$f_t^{(1)} \approx ... \approx f_t^{(h)}$$

則模型退化成多個相似時間尺度。

真正 capacity 來自：

* 不同維度學到不同 $f_t$
* 梯度壓力使維度分工

---

### 總結一句話

LSTM 本質是：

> 一組可學習時間常數的動態記憶系統

Seq2Seq 則是：

> 用這個動態記憶把整句壓縮成向量，再解碼成另一句話。

**ChatGPT 對話連結**: https://chatgpt.com/share/699c1332-5178-800a-b1e2-f4c7714d85b9
