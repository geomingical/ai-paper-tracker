# Reading Notes

The paper *Large-scale Video Classification with Convolutional Neural Networks* explores convolution-based methods for video recognition. Historically, fusing spatio-temporal features for video recognition has been a central challenge. To address this, the authors evaluate three temporal fusion strategies: early fusion, late fusion, and slow fusion. These are tested on the newly introduced Sports-1M dataset and compared against a single-frame baseline that processes inputs of $170 \times 170 \times 3$ pixels.

In early fusion, the network stacks 10 continuous frames as input, creating a $170 \times 170 \times 30$ spatio-temporal volume. In late fusion, the model inputs two frames sampled 15 frames apart into shared-parameter single-frame networks. These streams run in parallel until their final convolutional layers are flattened into 1D arrays of 12,544 dimensions. The two arrays are then concatenated into one long feature vector in the first fully connected layer to compute global motion characteristics before mapping to the 487 sports categories.

In slow fusion over a 10-frame clip, Conv 1 applies filters with temporal extent $T=4$ and stride 2, producing four sequential temporal responses:

```text
[1,2,3,4], [3,4,5,6], [5,6,7,8], [7,8,9,10]
```

Conv 2 and Conv 3 then gradually merge these components with temporal extent $T=2$, constructing a global spatio-temporal representation.

In the spatial domain, the authors introduce a multiresolution architecture with a context stream and a fovea stream to improve runtime performance. The context stream takes the original $178 \times 178$ frame and downsamples it by half to $89 \times 89$, capturing global context. The fovea stream extracts a high-resolution $89 \times 89$ crop from the center of the frame, leveraging camera bias. This design reduces total input dimensionality by half and allows the authors to evaluate multiple temporal configurations on the large Sports-1M dataset.

Experimental results show that slow fusion with the multiresolution architecture achieves the best balance of speed and accuracy, reaching 60.9% overall video accuracy. However, the single-frame baseline surprisingly reaches 59.3%, showing that the complex temporal designs provide only a modest improvement over static appearance. Qualitative analysis shows that common errors occur in fine-grained categories, such as deer hunting vs. hunting, hiking vs. backpacking, powered paragliding vs. paragliding, sledding vs. toboggan, and bujinkan vs. ninjutsu.

The authors hypothesize that motion-aware networks underperform when videos contain significant camera motion, such as translation and zoom. This suggests that explicit modeling of camera motion or tracking may be necessary to unlock the full value of temporal cues. Finally, the authors test transfer learning on UCF-101. They compare training from scratch, fine-tuning only the top layer, fine-tuning the top three layers, and fine-tuning all layers. Fine-tuning the top three layers performs best, reaching 65.4%, suggesting that low-level spatio-temporal features learned from Sports-1M can generalize to other action recognition tasks.

---

# Learning From AI (Gemini 3.5 Flash)

### 1️⃣ 研究目標與大規模資料集：Sports-1M

本研究針對大規模影片分類進行卷積神經網路的實證評估。

為了支持後續研究，作者收集並釋出 Sports-1M，包含 100 萬部 YouTube 影片與 487 個運動類別。

### 2️⃣ Video CNN 面臨的三大核心挑戰

第一，影片收集、標註與儲存都比影像資料更困難，當時缺乏能與影像領域相比的大規模基準資料集。

第二，建模上必須決定如何設計時域連接模式，才能有效利用影片中的局部運動資訊。

第三，影片需要同時處理多幀影像，導致數百萬參數的優化過程需要很長的訓練時間。

### 3️⃣ 影片輸入機制：Bag of Clips

影片在時間長度上差異很大，無法像影像一樣直接縮放到固定尺寸。

因此，本文將每部影片視為一袋短小且固定尺寸的 clips。每個 clip 包含時間上連續的多個 frames，用來延伸 CNN 在時間維度上的連接性，讓模型學習 spatio-temporal features。

### 4️⃣ 三種時域特徵融合策略

論文研究三種將時域資訊融合進 CNN 的策略：early fusion、late fusion、slow fusion。

融合可以發生在網路底層，例如修改第一層卷積核，也可以發生在網路頂層，例如雙塔特徵最後才合併。

### 5️⃣ Early Fusion：像素級的動態感測

Early fusion 在 pixel level 立即結合整個 temporal window 的資訊。

它將第一層卷積核在時間維度上延伸，尺寸可表示為：

$$
11 \times 11 \times 3 \times T
$$

這種直接連接 pixel data 的方式，使網路能偵測局部 motion direction 與 velocity。

### 6️⃣ Late Fusion：前後狀態的盲檢對比

Late fusion 使用兩個共享參數的 single-frame CNN towers，兩個輸入 frames 在時間上相隔 15 frames。

這兩個 streams 直到第一個 fully connected layer 才合併。單看任何一個單影格 tower 都無法偵測 motion，但 fully connected layer 可以透過比較雙塔輸出，計算較全局的 motion features。

### 7️⃣ Slow Fusion：最優的層級式時空金字塔

Slow fusion 在整個網路中緩慢融合 temporal information，使高層 neurons 能逐步接觸到在空間與時間上更全局的資訊。

它透過在時間維度上延伸所有 convolutional layers 的連接性，並額外執行 temporal convolution 來計算 activations。

實驗結果中，Slow Fusion 在單一網路架構中表現最好，Video Hit@1 達 60.9%。

### 8️⃣ 多解析度雙流架構：Multiresolution CNN

為了解決訓練耗時過長的問題，作者提出 multiresolution CNN，在不大幅犧牲準確度的前提下改善 runtime。

Context stream 接收空間解析度減半的整張影格，也就是 $89 \times 89$，負責建模低頻與色彩資訊。

Fovea stream 則接收原始高解析度影格中央的 $89 \times 89$ 區域，負責建模高頻細節。

這種設計將總輸入維度減半，並利用網路影片主角常出現在中央的 camera bias，帶來約 2 到 4 倍的速度提升。

### 9️⃣ 跌破眼鏡的「單影格悖論」

Single-frame model 只看單張靜態照片，但 Video Hit@1 仍達到 59.3%。

相比之下，精心設計並引入 temporal features 的 slow fusion 只有 60.9%，領先幅度不大。

這暗示在許多 Sports-1M 分類任務中，局部 motion cues 並非絕對關鍵；背景環境與靜態物件特徵，例如球場、服裝、器材，往往已經足以提供強分類訊號。

### 🔟 Transfer Learning 的黃金微調公式

Sports-1M 學到的 spatio-temporal features 具有一定泛化能力，可以遷移到較小資料集如 UCF-101。

實驗顯示，從小資料集直接從頭訓練會嚴重 overfit，準確率只有 41.3%；fine-tune all layers 也可能因過擬合而表現不佳，約 62.2%。

最佳折衷是 fine-tune top 3 layers，準確率達 65.4%。這保留了底部較通用的 low-level features，同時更新頂部與特定資料集相關的 high-level features。

---

## 一句話記憶

這篇 paper 的核心是：在 Sports-1M 大規模影片資料上比較 early、late、slow fusion，發現 temporal modeling 有幫助但提升有限，single-frame appearance 本身已經非常強，而 slow fusion 加 multiresolution CNN 是當時最有效的折衷。

**Gemini 對話連結**: https://gemini.google.com/share/7a3bcac23eb7
