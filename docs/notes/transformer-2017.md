# Reading Notes

The Transformer is a foundational architecture that underlies many modern large language models. Compared with earlier sequence approaches that depended on recurrence (RNNs) or convolution-based sequence models, the Transformer relies primarily on attention and can be trained in a highly parallel manner. In its original form, it uses an encoder–decoder design for sequence-to-sequence tasks.

At the input stage, each token is represented as a discrete ID and mapped to a continuous vector through a learned embedding matrix. Because attention alone does not encode order, positional information (positional encoding) is added to the embeddings to represent token positions. The resulting sequence of vectors becomes the input to the encoder.

A single encoder layer contains two main submodules. First, multi-head self-attention mixes information across tokens: the encoder projects the current token representations through three learned linear transformations to obtain Q, K, and V. Attention scores are computed by comparing queries to keys (via dot products), then normalized with softmax to form weights. These weights are used to take a weighted sum of values, producing context-dependent token representations. Second, the layer applies a position-wise feed-forward network (FFN/MLP) to each token independently, introducing nonlinearity and increasing representational capacity. Both submodules are wrapped with residual connections and layer normalization, which help stabilize optimization and enable deeper stacks. The original Transformer stacks multiple encoder layers (commonly six) to build increasingly abstract representations.

After encoding, the decoder generates outputs autoregressively. Each decoder layer includes masked self-attention (so the model cannot "peek" at future tokens) and encoder–decoder attention, which allows the decoder to attend to the encoder's outputs when producing each next token. The decoder's final representation is mapped through a linear layer and softmax to yield a probability distribution over the vocabulary for the next token.

Training is performed with backpropagation and gradient-based optimization to minimize a prediction objective such as cross-entropy loss. The key point is that correctness during training is defined by ground-truth tokens from the dataset (e.g., the true next token in language modeling, or the reference translation token in sequence-to-sequence training). Gradients flow from the output loss back through the decoder, encoder–decoder attention, encoder layers, and embeddings, updating all parameters.

My prompt experiments suggest that the model's outputs are strongly shaped by context and instruction. When prompts include domain-specific scientific terms and formal constraints, the model is pushed toward a low-ambiguity ("low-entropy") technical style that resembles textbooks or academic writing. When prompts explicitly request "simple terms," the model tends to shift toward a more general, blog-like explanatory mode. This supports the idea that prompts do not merely add information; they steer the model toward different regions of its learned conditional distribution.

---

# Learning From AI (GPT5.2)

### 1️⃣ Transformer 不是在學「語言規則」，而是在學「條件機率分佈」

重點：
模型唯一目標是 p(next token ∣ context)，不是理解，不是記憶，不是邏輯。
所有能力都是低 loss 副產品。

### 2️⃣ ID → Embedding 不是數學運算，而是「索引選行」

重點：
xᵢ = E[i]

- ID 沒有數值意義
- embedding matrix 是可訓練參數
- lookup = one-hot × linear 的等價實作

👉 真正的連續運算從 embedding 開始。

### 3️⃣ Self-Attention 本質上只是兩步

重點：
softmax(QKᵀ / √dₖ) V

- QKᵀ = 關聯分數
- softmax = 轉成比例
- ×V = 加權平均

👉 沒有語法規則，只有線性代數。

### 4️⃣ Attention 負責「資訊混合」，FFN 負責「非線性轉換」

重點：
- Attention：跨 token
- FFN：逐 token

如果沒有 FFN，整個模型會退化成近似線性系統。

### 5️⃣ LayerNorm 修的是「數值動態」，不是「語義」

重點：
它解的是梯度穩定，不是修 ID，不是修語言。

👉 它讓深層模型可以訓練，而不是讓模型變聰明。

### 6️⃣ 「低熵語境」= 只有極少數續寫能維持低期望 loss

重點：
這是你整段對話的核心洞察。

- 教科書式 prompt → 低熵
- simple terms → 混合模態
- 污染發生在語境衝突，不是主題轉換

👉 熵 = 條件分佈的集中程度。

### 7️⃣ LLM 沒有「記憶文本」，只有「壓縮後的機率地形」

重點：
沒有錯誤日誌，沒有 episodic memory，沒有成長歷史。
它只是把「高 loss 路徑」壓到低機率。

### 8️⃣ 摘要犧牲的不是字數，而是「重建推理所需的條件」

重點：
摘要通常會丟掉：

- 推理鏈
- 例子
- 邊界條件
- 不確定性語氣

👉 你不能用摘要取代理解。

### 9️⃣ 上下文污染 ≠ 話題變多

重點：
污染發生在：

- 同一生成任務中
- 混入衝突模態
- 目標不一致

我們這串對話沒有污染，因為目標一致（機制理解）。

### 🔟 Prompt 的真正功能不是「給資訊」，而是「限制生成軌道」

重點：
好的 prompt：

- 鎖定語料群集
- 指定結構型動詞
- 避免高熵詞

👉 Prompt 是在調整條件分佈，不是在教模型。

### 🌱 加碼：你個人特別要記住的一句話

你在這整段對話中，其實已經完成了一個轉變：

從「模型為什麼會這樣？」
變成「我怎麼從機率分佈角度預測模型會怎樣？」

這個轉變比你學會 Q/K/V 還重要。

**ChatGPT 對話連結**: https://chatgpt.com/share/698d5e95-10c0-800a-9f10-ee2d54b06ce5