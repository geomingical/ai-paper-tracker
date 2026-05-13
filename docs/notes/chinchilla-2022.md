# Reading Notes

The original Kaplan scaling law suggested that larger models improve LLM performance, but this conclusion came from settings where the number of training tokens and the learning rate schedule were not fully varied. Chinchilla asks a slightly different question: if the total training compute budget is fixed, how should we balance model size and the number of training tokens?

The authors argue that previous large language models may have been undertrained. They had many parameters, but they were not trained on enough tokens. To study this, the paper trains over 400 models with different model sizes $N$ and numbers of training tokens $D$, then analyzes the trade-off in three ways. First, it fixes model size and varies the number of training tokens. Second, it fixes compute budget, measured in FLOPs, and studies how model size affects final loss. Third, it fits an empirical loss function that separates natural uncertainty, model-size limitation, and data limitation.

The main conclusion is that model size and training tokens should scale in roughly equal proportions as compute increases. Based on this finding, the authors train Chinchilla, a 70B-parameter model on 1.4T tokens. This gives an approximate parameter-to-token ratio of 1:20. Even though Chinchilla is much smaller than Gopher, it performs better under a similar compute budget, suggesting that training with more data can be more effective than simply increasing model size.

---

# Learning From AI (GPT5.4)

### 1️⃣ 這篇不是單純問「模型越大越好嗎？」

Chinchilla paper 的核心問題是：在固定訓練 compute budget 下，模型參數量 $N$ 和訓練 token 數 $D$ 要怎麼分配，才能讓 loss 最低？

它不是否定 scaling law，而是修正「compute 增加時應該主要拿去把模型做大」這個早期直覺。

📍Introduction, Eq.(1), Section 3

### 2️⃣ FLOPs 可以理解成訓練的總計算預算

FLOPs 是 floating-point operations，也就是訓練過程中大約做了多少次浮點運算。本文把它當成訓練 compute budget $C$。

本文主文使用的近似是：

$$
\text{FLOPs}(N,D) \approx 6ND
$$

其中 $N$ 是模型參數量，$D$ 是訓練 tokens。直覺上，模型越大，每看一個 token 越貴；tokens 越多，總訓練成本也越高。

📍Eq.(1), Eq.(4) 前文, Section 3

### 3️⃣ Kaplan Scaling Law 和 Chinchilla 的差異在「compute 要花在哪裡」

Kaplan et al. 的結論偏向：如果 compute 增加 10 倍，模型大小應增加約 5.5 倍，訓練 tokens 只增加約 1.8 倍。

Chinchilla 的結論是：模型大小和訓練 tokens 應該近似等比例一起增加。

用 power law 表示：

$$
N_{\text{opt}} \propto C^a, \quad D_{\text{opt}} \propto C^b
$$

Kaplan 約是 $a=0.73, b=0.27$，Chinchilla 三種方法約是 $a \approx 0.46\sim0.50, b \approx 0.50\sim0.54$。

📍Introduction, Table 2

### 4️⃣ 「Equal Proportions」不是說 $D = kN$ 的固定線性關係

這裡的 equal proportions 比較準確的意思是：當 compute budget 增加時，最佳模型大小和最佳訓練 tokens 的成長速度差不多。

也就是說，compute 多了，不該大部分都拿去增加參數量，也要同步增加資料量。

📍Table 2, Section 3.4

### 5️⃣ Gopher 是本文中「舊 Scaling 直覺」的代表

Gopher 是 280B parameters、約 300B training tokens 的大型 dense Transformer LM。

Chinchilla 是 70B parameters、1.4T training tokens。兩者訓練 compute 大致相近，但 Chinchilla 比 Gopher 小約 4 倍、看過的 tokens 多約 4 倍以上。

本文用這個對照驗證：同樣 compute 下，較小模型加更多資料，可能比超大模型加較少資料更好。

📍Abstract, Table 1, Table 4, Section 4

### 6️⃣ 400 多個模型訓練實驗不是 400 種大架構

本文所說的 over 400 language models，指的是作者訓練了許多不同模型大小、不同訓練 token 數、不同 FLOPs budget 的實驗點。

模型大小範圍約從 70M 到超過 16B parameters；訓練 tokens 約從 5B 到 500B tokens。

可以把它想成在 $(N,D)$ 平面上做很多「試鑽點」，觀察固定 compute 下哪個組合 loss 最低。

📍Abstract, Introduction, Section 3

### 7️⃣ 三個方法其實都在找 Compute-Optimal Frontier

三個方法的共同目標都是估計：

$$
N_{\text{opt}}(C), D_{\text{opt}}(C)
$$

也就是在固定 compute $C$ 下的最佳模型大小與最佳訓練 token 數。

三個方法分別是：

* **Approach 1: training curve envelope**  
  固定一批模型大小，改變訓練長度，從所有 training curves 中找每個 FLOPs 下 loss 最低的包絡線。
* **Approach 2: IsoFLOP profiles**  
  固定 FLOPs，掃不同模型大小；每條 IsoFLOP curve 的 loss 谷底就是該 compute 下的最佳模型大小。
* **Approach 3: parametric loss fitting**  
  用經驗式擬合所有實驗點，再在 $\text{FLOPs} \approx 6ND$ 限制下求最佳點：

$$
\hat{L}(N,D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}
$$

📍Section 3.1, Figure 2; Section 3.2, Figure 3; Section 3.3, Eq.(2), Eq.(4), Figure 4

### 8️⃣ Chinchilla 的地質直覺：不要只增加模型自由度，也要增加觀測約束

如果把模型參數量 $N$ 類比成地質反演模型的自由度，把 tokens $D$ 類比成觀測資料量，那 Gopher 像是自由度非常高、但觀測資料相對不足的模型。

Chinchilla 的策略像是：自由度少一些，但用更多觀測資料來約束模型。

這個類比的失效點是：Transformer 參數不是可直接解釋的地質參數，tokens 也不是均質觀測點；本文結論仍來自 loss/FLOPs 實驗 fitting。

📍Eq.(1), Eq.(2), Table 1, Table 2

### 9️⃣ 「少於一個 Epoch」表示模型還沒完整看完整個資料庫一次

本文結論提到 scaling analysis 中的 training runs 都少於一個 epoch。意思是：模型訓練時看到的 tokens 少於整個 corpus 的 tokens，還沒有把整個訓練資料完整掃過一遍。

這很重要，因為在這個 regime 下，增加 training tokens 比較像是「看更多新資料」，而不是「重複背同一批資料」。

作者也明確把 multiple epoch regime 留給 future work，因此 Chinchilla scaling law 的結論主要適用於資料量足夠大、模型還沒大量重複資料的情境。

📍Page 2 footnote 2, Discussion & Conclusion

### 🔟 Chinchilla 把後續 LLM 發展的瓶頸推向「資料」

Chinchilla 的結果暗示：未來如果 compute 繼續增加，不能只把模型做大，也需要等比例增加高品質資料。

但更大的 web-scale dataset 會帶來毒性語言、偏見、私人資訊、train-test overlap、資料重複與清洗困難等問題。

後續 LLM 因此越來越重視資料策展、清洗、去重，以及 synthetic data。要注意的是：synthetic data 是一種資料來源；distillation 是一種訓練方法。用大模型產生資料來教小模型時，兩者可能重疊，但概念上不完全相同。

📍Discussion & Conclusion

---

## 一句話記憶

Chinchilla 的重點不是「小模型一定比較好」，而是「在固定 compute 下，模型大小和訓練資料量要平衡；當時許多大模型相對於它們的資料量來說太大、訓練不夠。」

**ChatGPT 對話連結**: https://chatgpt.com/share/6a0437bb-9404-83a7-ae55-d0cbeac9e8dd
