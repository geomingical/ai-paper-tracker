# Reading Notes

AlexNet is widely regarded as a turning point that demonstrated the practical power of deep convolutional neural networks (DNNs) at large scale. Before AlexNet, many image recognition systems relied on hand-crafted features designed for specific image categories or tasks, and then applied shallow classifiers such as SVMs or logistic regression for classification.

In contrast, AlexNet used a deep architecture with five convolutional layers and three fully connected layers. The earlier layers tend to learn low-level visual patterns such as edges and textures, while deeper layers combine these patterns into more abstract, high-level representations. Trained end-to-end with backpropagation, along with techniques such as ReLU and regularization methods, the model achieved a substantial performance improvement on ImageNet-style benchmarks.

In addition, the book《深度學習革命：從歷史到未來》addresses the history of deep learning, including how major technology companies pursued and invested in the field about a decade ago.

**Reference**: 《深度學習革命：從歷史到未來》 https://www.tenlong.com.tw/products/9787521747553

---

# Learning From AI (GPT5.2)

## AlexNet（2012）— 10 點必記 Learning Notes

### 1️⃣ 問題規模本身就是突破的一半

ImageNet（100 萬張、1000 類）不是背景，而是核心挑戰。在這個規模下，手工特徵 + 傳統分類器全面失效。

依據：Section 1, Section 2

👉 記住：AlexNet 解的不是「更準的模型」，而是「在這個規模下還能學得動」。

### 2️⃣ CNN 的核心不是 convolution，而是「端到端學特徵」

卷積不是新東西，新的是：從 pixel 到 class，所有表示都為同一個 loss 服務。

依據：Abstract, Section 1

👉 這是 AlexNet 與 SIFT/HOG 時代的根本分水嶺。

### 3️⃣ Channel（厚度）≠ 顏色，而是「特徵種類數」

- C1 的 96
- C2 的 256
- C3/C4 的 384

都是 kernel 數量 = 作者設定的超參數。

依據：Section 3.5, Figure 2

👉 空間尺寸回答「在哪裡」，channel 回答「用幾種方式看」。

### 4️⃣ 滑動（stride）只影響 2D 尺寸，不會產生厚度

- 55×55 來自 stride = 4
- 96 來自 96 個 kernel

依據：Section 3.5

👉 厚度不是疊位置，是疊「不同的回應」。

### 5️⃣ ReLU 的意義是「讓深網真的學得動」

ReLU 不是生物模擬，而是數值工程：
- 不飽和
- 梯度不容易消失
- 訓練速度大幅提升

依據：Section 3.1, Figure 1

👉 沒有 ReLU，AlexNet 的深度在當年幾乎不可行。

### 6️⃣ Data Augmentation 是「學不變性」，不是資料作弊

AlexNet 只做兩件事：
- 隨機裁切 + 翻轉（空間不變性）
- PCA-based 顏色擾動（光照不變性）

依據：Section 4.1

👉 模型不是記住圖片，而是被逼忽略不重要的變化。

### 7️⃣ Dropout 不是關資料，也不是關類別

Dropout = 訓練時，隨機讓「中間 neuron 暫時失聲」
- 用在 FC6、FC7
- 不用在最後 softmax

依據：Section 4.2

👉 防止 co-adaptation，而不是降低維度。

### 8️⃣ Flatten 是「資訊重排」，不是運算

13×13×256 → 43264

只是把三維 tensor 拉成一條向量。沒有學習、沒有權重、沒有保留空間關係。

依據：Section 3.5

👉 從這一刻開始，表示不再對齊影像座標。

### 9️⃣ 1000 個輸出之所以對應 1000 個類別，是 loss 在「點名」

- forward：模型不知道哪個 index 是哪個類
- loss：只懲罰正確 label 對應的那一個位置
- backprop：反覆把語義壓到固定 index

依據：Section 3.5

👉 語義不是算出來的，是被反向傳播「固定住的」。

### 🔟 AlexNet 的成功是「工程折衷」，不是理論最優

- 大 kernel（11×11）
- GPU 分組
- 手動設計 channel 數

都是 2012 年硬體與經驗的結果。

依據：Section 3.2, Section 7

👉 你要學的是「設計原則」，不是照抄數字。

**ChatGPT 對話連結**: https://chatgpt.com/share/697f3a95-687c-800a-9bfa-2f067b4d7200
