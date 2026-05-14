# Reading Notes

Knowledge distillation is useful for tasks where ensemble models achieve high prediction accuracy, but their computational cost and latency are too expensive for practical deployment. In these situations, a smaller model with similar performance is desirable. The outputs of the teacher model, or multiple teacher models, together with their probability distributions, serve as the supervision targets for the student model. The student model is trained through backpropagation so that its output distribution becomes close to that of the teacher model.

Temperature is a key parameter in the distillation process. The teacher model generates output classes or tokens together with a probability distribution over all possible outputs. Increasing the temperature smooths the probability distribution and reveals more information about relationships among candidate outputs. As a result, the prediction becomes more informative for the student model. This temperature parameter is mathematically identical to the temperature used in normal large language model inference, although the purpose is different. In inference, temperature controls sampling randomness during text generation. A higher temperature increases the chance of selecting lower-probability tokens, while a lower temperature makes the model output more deterministic.

If this concept is extended to earthquake science, the process can be viewed as an ensemble of multiple earthquake prediction or seismic picking models acting as the teacher system. The student model then learns from the combined probability outputs of these models. In this framework, the upper limit of the student model performance is strongly constrained by the quality and capability of the teacher ensemble.

---

# Learning From AI (GPT5.5)

### 1️⃣ Distillation 的核心不是壓縮參數，而是壓縮「行為」

這篇 paper 最重要的哲學是：knowledge 並不真正存在於 model weights，而存在於 input 到 output 的 mapping。

也就是說，student model 並不是複製 teacher 的參數，而是在學 teacher 如何看待世界、如何排列答案、如何進行判斷。

因此 distillation 的本質是 behavior transfer，而不是 parameter copy。

### 2️⃣ Soft Targets 比 Hard Labels 含有更多資訊

一般 supervised learning 只有：

```text
正確答案 = 1
其他答案 = 0
```

但 teacher model 的 soft probabilities 包含：

* 哪些類別彼此接近
* 哪些答案容易混淆
* teacher 的 uncertainty
* latent semantic structure

例如：

```text
cat 0.52
dog 0.31
car 0.001
```

這比單純的 one-hot label 更接近模型真正學到的 decision geometry。

### 3️⃣ Temperature 的目的不是增加 Randomness，而是揭露 Hidden Structure

Temperature 是整篇 paper 最核心的方法。

正常 softmax 通常太尖銳：

```text
cat 0.999
dog 0.001
```

student 幾乎學不到資訊。

提高 temperature 後：

```text
cat 0.52
dog 0.31
fox 0.14
```

原本隱藏的 similarity structure 開始浮現。

因此 distillation 真正學的是：

```text
哪些東西彼此像
```

而不是：

```text
唯一正確答案是什麼
```

### 4️⃣ Distillation 的 Loss 是 Distribution Matching

普通 supervised learning：

```text
student output vs ground truth
```

Distillation：

```text
student distribution vs teacher distribution
```

student 的目標是模仿 teacher 的完整 token probability landscape。

這通常透過 cross entropy 或 KL divergence 來完成。

因此 teacher probabilities 並不是 input，而是 supervision signal。

### 5️⃣ Distillation 幾乎總是建立在 Pretrained Model 上

現代 LLM world 中，student model 幾乎不會是隨機初始化。

真正流程通常是：

```text
Pretraining
→ base model
→ distillation / finetuning
```

因此 distillation 並不是重新學習語言，而是在已經具備語言能力的模型上重新塑造 behavior。

Pretraining 在建立 world model。

Distillation 在 transfer reasoning policy。

### 6️⃣ Distillation 最擅長 Transfer Capability，而不是 Inject Knowledge

這是現代 AI engineering 很重要的 distinction。

Distillation 很適合 transfer：

* reasoning
* instruction following
* coding style
* chain of thought
* planning
* alignment behavior

但不適合新增：

* 公司內部知識
* 最新論文
* 法律條文
* 即時資訊

因此 domain knowledge 問題通常更適合 RAG、domain finetuning、retrieval systems，而不是純 distillation。

### 7️⃣ Ensemble Distillation 的本質是壓縮 Decision Boundary

Paper 的 speech recognition 實驗中，teacher 其實是 10 個 model 的 ensemble。

Ensemble 的優勢包括 variance reduction、更平滑的 decision boundary，以及更強的 generalization。

Distillation 的目的不是重建 10 個模型，而是：

```text
用單一小模型近似 ensemble 的判斷方式
```

因此 student model 學到的是 ensemble 的 collective behavior。

### 8️⃣ Generalization Ability 可以被 Transfer

這是 paper 最革命的觀念之一。

以前大家認為：

```text
generalization 是模型自己學到的
```

但這篇 paper 發現：

```text
generalization pattern 也能被 teacher 傳遞
```

例如哪些資料彼此接近、哪些 decision boundary 比較合理、哪些錯誤比較自然，都能透過 soft targets 傳遞給 student。

這是後來 reasoning distillation 與 synthetic supervision 的基礎。

### 9️⃣ Specialist Models 與 Mixture of Experts 的核心是「分工」

Paper 後半開始討論 specialists。

核心概念是：

```text
不是每個模型都要學全部東西
```

而是：

* generalist 處理大方向
* specialists 處理容易混淆的小子空間

例如車種 specialist、橋樑 specialist、mushroom specialist。

這其實是現代 MoE 的早期思想。

真正大型 system 可以先建立很多 experts，再透過 distillation 壓縮回單一 student model。

### 🔟 Distillation 的真正價值是 Performance / Cost Tradeoff

Distillation 很少讓 student 完全超越 teacher。

真正目標通常是：

```text
接近 teacher 的能力
+
更低 inference cost
+
更低 latency
+
更容易 deployment
```

因此 distillation 本質上是：

```text
用有限 capacity 逼近更大的 decision landscape
```

這也是為什麼現代 frontier models，例如 GPT 系列、Claude 系列、DeepSeek 系列、Gemma 系列，都大量使用 distillation。

因為大型 teacher 很強，但真正能大量部署的，往往是較小的 distilled models。

---

## 一句話記憶

普通 training 在學：

```text
哪個答案是正確的
```

Distillation 在學：

```text
強模型如何理解整個答案空間
```

這就是 knowledge distillation 的真正核心。

**ChatGPT 對話連結**: https://chatgpt.com/share/6a05b13c-5298-83a7-84ed-3ebb92e862f3
