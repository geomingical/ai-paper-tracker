# Reading Notes

Before the Word2Vec paper, language models mainly relied on the Feedforward Neural Network Language Model (NNLM), which includes embedding, a hidden layer, and a softmax output layer, and the Recurrent Neural Network Language Model (RNNLM), which uses a hidden recurrent layer followed by softmax. However, these architectures require significant computational cost.

The Word2Vec paper removes the hidden layer, which greatly reduces computational complexity and allows training on much larger datasets. According to the experimental results and model architecture, semantically similar words tend to align in similar vector directions, revealing linear relationships in the embedding space.

---

# Learning From AI (GPT)

### 1️⃣ 本文真正目標不是語言模型，而是高品質詞向量

核心：representation learning，不是 perplexity 最佳化。

依據：
- Abstract：learn high-quality word vectors
- Sec 1.1：目標是 preserve linear regularities

👉 他們刻意犧牲語言模型複雜度，換取 embedding 幾何品質。

### 2️⃣ 計算複雜度是設計出發點

訓練成本：O = E × T × Q（依據 Sec 2）

NNLM 的瓶頸：Q = N×D + N×D×H + H×V（依據 Eq.(2)）

👉 移除 hidden layer 是為了降複雜度。

### 3️⃣ CBOW vs Skip-gram 的根本差異

- CBOW：平均 context → 預測中心詞
- Skip-gram：中心詞 → 預測每個 context

依據 Sec 3.1, 3.2

👉 Skip-gram 對 semantic 類比更強（Table 3），原因：對共現差異建模更精細。

### 4️⃣ 線性語意結構不是偶然

例子：king − man + woman ≈ queen（依據 Sec 1.1）

👉 語意方向來自 log 共現機率差異。

### 5️⃣ 模型本質是 log-linear model

Sec 3 開頭明確指出是 log-linear，也就是：

v_w · v_c ≈ log P(c|w)

👉 這等價於低秩矩陣分解。

### 6️⃣ Hierarchical Softmax 是效率關鍵

用 Huffman tree 把 softmax 從 V 降為 log(V)（依據 Sec 2.1）

👉 沒這個技巧，1M vocab 不可能訓練。

### 7️⃣ window 是統計範圍控制器

Skip-gram 用隨機 R ∈ [1, C]（依據 Sec 3.2）

👉 小 window → 局部語意
👉 大 window → 主題語意

這影響 embedding 幾何。

### 8️⃣ 資料量與維度必須匹配

Sec 4.2（Table 2）觀察：
- 只增加維度 → 邊際收益遞減
- 只增加資料 → 邊際收益遞減
- 同時增加 → 效果最好

👉 bias-variance + capacity matching。

### 9️⃣ 評估方式是幾何測試，不是生成測試

Semantic-Syntactic test（Sec 4.1）

這不是 P(sentence)，而是：向量差 + cosine 距離。

👉 檢驗 embedding 空間結構。

### 🔟 真正深層洞察：語言共現統計具有低秩結構

即使模型變大（如 Transformer），語意方向仍然存在。

這說明：
- 線性方向不是模型副產品
- 而是語言統計本身具有低秩幾何

這點雖未明說，但從整體結果（Sec 4）可推論。

### 總結一句話

Word2Vec 的成功不在於模型複雜，而在於：用極簡線性模型 + 大量資料，恢復語言共現矩陣的低秩幾何結構。

**ChatGPT 對話連結**: https://chatgpt.com/share/69995667-bc04-800a-9893-0dfac0072eaf