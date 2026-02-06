# Reading Notes

The ResNet paper introduces the concept of residual learning, which has strongly influenced later architectures such as Transformers. Before ResNet, widely used deep learning models included AlexNet and VGG. However, the authors observed that as network depth increases, the training error also increases, which is known as the degradation problem. To address potential optimization issues, the authors adopted Batch Normalization, where the activations in each batch are normalized to have zero mean and unit variance. This helps stabilize training and ensures that gradients can be propagated effectively. Nevertheless, the degradation problem still remained even with Batch Normalization. To solve this issue, the authors proposed residual learning. Instead of directly learning the target mapping 𝐻 ( 𝑥 ) H(x), the network is encouraged to learn a residual function 𝐹 ( 𝑥 ) = 𝐻 ( 𝑥 ) − 𝑥 F(x)=H(x)−x, and the output becomes 𝑦 = 𝐹 ( 𝑥 ) + 𝑥 y=F(x)+x. The shortcut connection delivers the input feature map 𝑥 x directly to the output of the block.

When the shortcut connection operates within the same stage (i.e., the channel number and spatial dimensions are the same), an identity shortcut is used. When the shortcut crosses different stages, where dimensions change, a projection or zero-padding shortcut is required to match the shapes. With this design, each residual block can either modify the input features by learning a non-zero residual or simply preserve the input through identity mapping. If the optimal behavior is to keep the representation unchanged, the residual function approaches zero; otherwise, the network learns meaningful residuals to refine the features.

---

# Learning From AI (GPT5.2)
## ResNet 論文 Reading Notes（10 點版）

### 1️⃣ ResNet 真正解決的是「優化問題」，不是資訊流動
重點：
ResNet 的核心不是讓資訊「傳得更遠」，而是讓 深度網路不再因為 optimization 困難而變差（degradation problem）。
實戰意義：
如果你加深模型後 training error 反而上升，你該想的是 結構性重參數化（residual），不是換 optimizer。

### 2️⃣ Residual 的本質：讓「什麼都不做」成為合法選項

重點：
每一個 residual block 都可以選擇：
學一點修正（F(x) ≠ 0）
或完全不動（F(x) = 0 → identity）
實戰意義：
深度變深時，不是每一層都必須有貢獻，重要的是「需要時能動，不需要時能退場」。

### 3️⃣ Shortcut 要「每個 block 都有」，不是只在 stage

重點：
Shortcut 的密度很重要。
如果只在 stage 有 shortcut，stage 內的 block 仍然被迫「一定要動」，會退化成 plain net。

實戰意義：
Residual 是 block-level 的設計原則，不是 stage-level 的裝飾。

### 4️⃣ Stage 切換時 shortcut 一定要動，原因只有一個：shape

重點：
只要 (C, H, W) 有任何一個改變，shortcut 就必須對齊（projection 或 padding），否則數學上不能相加。

實戰意義：
你設計新架構時，shortcut 的第一檢查項不是「效能」，而是：

最後能不能加？

### 5️⃣ Stride、channel、kernel 是三件完全獨立的事

重點：
stride → 決定空間尺寸（H, W）
filter 數 → 決定 channel（C）
kernel size → 決定感受野

實戰意義：
不要把「stride=2 所以 channel 變多」混在一起；
channel 永遠是你設計的，不是自動發生的。

### 6️⃣ Projection shortcut 不是核心能力，只是對齊工具

重點：
Table 3 顯示 A/B/C 差異很小 →
ResNet 能成功 不是因為 projection 厲害，而是 residual formulation 本身。

實戰意義：
能用 identity 就用 identity；
projection 只在「shape 不合」時出現。

### 7️⃣ Bottleneck + identity shortcut 是一組「省錢又穩定」的搭配

重點：
Bottleneck 把計算集中在低維空間，
identity shortcut 保證高維端不再增加額外成本。

實戰意義：
當模型要「很深」時，
你應該先想 怎麼省 shortcut 的成本，而不是只看主分支。

### 8️⃣ 「模型變深比較好」只對 representation 階段成立

重點：
深度的價值主要在 convolution / representation learning，
不是在 classification head（FC）。

實戰意義：
如果你想提升表現，優先加深 backbone，
不是在 top 加一堆 fully connected。

### 9️⃣ Fully Connected 的使用判斷，不是看圖片，而是看「決策性質」

重點：

自然影像、資料多 → 幾乎不需要多 FC
結構化特徵、小資料、複雜規則 → 少量 FC 有時有幫助
實戰意義：
FC 解的是 decision boundary，不是視覺理解。

### 🔟 新手選架構的黃金流程（最重要）

重點流程：
先用標準 backbone（ResNet-18/34/50）+ GAP + 1 FC
看 training / validation 行為，而不是只看 accuracy
有證據才調整：
學不到 → backbone 不夠
overfit → FC 太多
決策太僵硬 → 小 FC

實戰意義：
👉 架構不是一開始「想出來的」，
👉 是被訓練行為「逼出來的」。

### 一句總結（給你以後反覆看的）

好的網路結構不是「什麼都學」，
而是「在正確的位置給模型正確的自由度」。

**ChatGPT 對話連結**: https://chatgpt.com/share/6985dd07-21a0-800a-8546-7ef7b1b1bc15