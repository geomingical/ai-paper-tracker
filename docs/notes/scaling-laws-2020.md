# Reading Notes

The scaling law paper serves as an important foundation for the later GPT-3 work. The main research question is whether there is a predictable relationship between language model scale and language modeling performance. In particular, the paper investigates how model parameters, dataset size, and training compute affect cross-entropy loss.

The authors empirically show that loss follows power-law relationships with respect to model size, dataset size, and compute. These results are not presented as a strong theoretical derivation. They are primarily empirical observations supported by regression fits on log-log plots.

One important conclusion is that the compute-efficient strategy is not to train a small model until full convergence. Instead, for a fixed compute budget, it can be better to train a much larger model for fewer steps and stop early. Larger models are more sample-efficient, meaning they require fewer training tokens and fewer optimization steps to reach the same loss compared with smaller models.

However, Chinchilla et al. (2022) later argued that the Kaplan scaling law overemphasized model size and underestimated dataset size. They showed that many large language models were undertrained because they used too few tokens. Their revised scaling rule suggests that training tokens should scale roughly linearly with parameters, commonly summarized as:

$$
\text{tokens} \approx 20 \times \text{parameters}
$$

This later result does not make the Kaplan paper unimportant. Instead, it clarifies that the central question is not simply how to make models larger, but how to balance model size, dataset size, and compute.

---

# Learning From AI (GPT5.4)

### 1️⃣ 核心問題

這篇 paper 想回答：

> Language model 的 performance 如何隨著 model size、dataset size、training compute 一起 scaling？

核心目標是找到可預測的 scaling law。

📍Section 1

### 2️⃣ 最重要發現：Loss Obeys Power Law

作者發現 loss 會隨著 model size $N$、dataset size $D$、compute $C$ 呈現 power-law 下降：

$$
L \propto N^{-\alpha}
$$

$$
L \propto D^{-\alpha}
$$

$$
L \propto C^{-\alpha}
$$

也就是模型越大、資料越多、算力越高，loss 會平滑下降，而且在 log-log plot 上接近直線。

📍Eq.(1.1)-(1.3)

### 3️⃣ 真正重要的是總參數量 $N$

Paper 發現 performance 對 deeper、wider、attention heads 等架構細節相對不敏感。

真正主要的 controlling factor 是：

$$
N = \text{total non-embedding parameters}
$$

📍Section 3.1, Figure 5

### 4️⃣ 大模型比較 Sample Efficient

大模型不是比較便宜，而是在達到相同 loss 時，所需 data 和 training steps 較少。

換句話說，大模型的 training efficiency 更高。

📍Summary, Figure 2

### 5️⃣ Overfitting 的本質是 $N$ 和 $D$ 的比例

Section 4 的核心是：performance 同時受到 model capacity 和 dataset size 限制。

作者用：

$$
L(N,D)
$$

統一描述 model size 和 dataset size 對 loss 的影響。

📍Eq.(4.1)

### 6️⃣ Figure 9：Data Bottleneck

Figure 9 左圖顯示：固定 dataset size $D$ 時，一開始增加 model size $N$ 會有效，但後來 loss 會 plateau。

原因是 data 不夠，model 開始 overfit。

📍Figure 9

### 7️⃣ Kaplan Scaling

作者發現 overfitting 主要由下列比例控制：

$$
\frac{N^{0.74}}{D}
$$

因此，若想維持固定 overfitting：

$$
D \propto N^{0.74}
$$

意思是 model 增大 10x 時，data 不需要增大 10x。

📍Eq.(4.4)

### 8️⃣ Critical Batch Size

Section 5 引入 critical batch size：

$$
B_{\text{crit}}
$$

它代表 batch size 再增大就開始 diminishing returns 的點。

重要的是，$B_{\text{crit}}$ 不是固定值，而是 training 狀態的函數：

$$
B_{\text{crit}}(L)
$$

📍Eq.(5.2)-(5.3)

### 9️⃣ 模型越強，越能使用超大 Batch

Figure 10 發現，當 loss 降低時，$B_{\text{crit}}$ 會快速增加。

直覺上：

* training early：gradient noise 很大
* training late：gradient 更一致

因此，訓練後期可以有效使用更大的 batch。

📍Figure 10

### 🔟 最革命發現：Compute-Efficient Training

Kaplan paper 最重要的 insight 是：最佳策略不是小模型 train 到收斂，而是大模型加 early stopping。

原因是大模型更 sample efficient。

📍Section 6

### 1️⃣1️⃣ Kaplan vs Chinchilla

Kaplan (2020) 的結論偏向超大模型，認為 data 增長可以比 model size 慢。

Chinchilla (2022) 後來指出 Kaplan scaling 低估了 data 的重要性，許多大型模型其實是 undertrained giant models。

Chinchilla 的經典比例是：

$$
\text{tokens} \approx 20 \times \text{parameters}
$$

這表示 model size 和 data size 應該更同步地 scaling。

### 1️⃣2️⃣ 個人理解

這篇 paper 的本質不是理論推導，而是 empirical phenomenology。

作者先觀察 scaling 現象，接著 fit power law，再建立 unified framework。這很像 statistical physics 裡的 data collapse。

---

## 一句話記憶

Scaling Law 的核心不是 bigger is always better，而是 performance bottleneck 可以被 model、data、compute 的 scaling law 預測。

**ChatGPT 對話連結**: https://chatgpt.com/share/6a041364-6d60-83ab-a676-4a106a3a83d6
