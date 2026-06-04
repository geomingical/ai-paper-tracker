# Reading Notes

Video action recognition involves processing complex spatio-temporal information, making effective feature learning a significant challenge. To address this issue, the paper proposes a Two-Stream ConvNet architecture comprising two parts: the Spatial Stream, which operates on still video frames to capture appearance, and the Temporal Stream, which operates on dense optical flow to capture motion. The outputs of both streams are later combined through late fusion. Both streams share the same underlying CNN-M-2048 backbone; their primary difference lies in their input configurations.

During training, each video clip in the dataset is assigned a single ground-truth action label. For the Spatial Stream, the data pipeline randomly samples a single frame $\tau$ from a video. For the Temporal Stream, a stack of $L=10$ consecutive optical-flow frames starting from $\tau$ is extracted.

The computation of dense optical flow relies on two classical computer-vision assumptions: brightness constancy and spatial smoothness, where neighboring pixels are expected to share similar velocities and directions. If pixels become occluded or move outside the frame boundary, the energy-minimization framework uses interpolation or extrapolation to estimate displacement vectors.

Motion between frame $t$ and $t+1$ is decomposed into horizontal displacement $d^x$ and vertical displacement $d^y$. Stacking 10 optical-flow frames therefore produces 20 channels, resulting in a temporal tensor with shape:

$$
224 \times 224 \times 20
$$

Finally, a linear SVM performs late fusion by combining the normalized softmax scores from both streams. The experimental results show that the two-stream model achieves 87.0% classification accuracy on the first split of UCF-101. However, the model still makes specific errors, notably confusing the Hammering class with HeadMassage and BrushingTeeth.

---

# Learning From AI (Gemini 3.5 Flash)

### 1️⃣ 雙流核心架構：Core Two-Stream Architecture

Two-Stream ConvNet 將影片拆解為空間 (Spatial) 與時間 (Temporal) 兩個獨立的卷積網路，最後在 softmax scores 後進行 late fusion。

空間流捕捉靜態外觀 (appearance)，時間流捕捉幀間運動 (motion)。

### 2️⃣ 空間流配置：Spatial Stream

空間流的輸入是單張靜態影片幀，維度為：

$$
224 \times 224 \times 3
$$

因為它本質上接近 image classification，所以可以利用大規模 ImageNet 資料集進行 pre-training，減少影片資料量不足造成的 overfitting。

### 3️⃣ 時間流配置：Temporal Stream

時間流的輸入是連續多幀的 dense optical-flow displacement fields。

它顯式地向網路提供運動資訊，使神經網路不需要完全依賴底層特徵自行學習 motion representation，降低模型的優化負擔。

### 4️⃣ 光流堆疊機制：Optical Flow Stacking

系統將連續 $L$ 幀的 optical flow 疊加，每幀包含水平位移 $d^x$ 與垂直位移 $d^y$，因此總通道數為 $2L$。

實驗顯示，使用 $L=10$，也就是 20 個通道，可以提供比單幀光流 $L=1$ 更有效的運動資訊。

### 5️⃣ 關鍵前處理：Motion Compensation 與 Compression

Mean-flow subtraction 會從每個 displacement field 中減去平均向量，作為簡單且有效的 camera-motion compensation。

光流浮點數也會被線性縮放到 $[0,255]$ 並儲存成 grayscale JPEG。這項工程優化將 UCF-101 的 optical-flow 資料量從約 1.5 TB 降低到 27 GB，大幅減少儲存和訓練 I/O 成本。

### 6️⃣ 網路架構的對稱性：Structural Symmetry

兩條 stream 的 convolutional backbone 幾乎完全相同，皆基於 CNN-M-2048，包含 5 個 convolutional layers 與 2 個 fully connected layers。

時間流為了減少 GPU memory 使用，移除了第二個 local response normalization (LRN) layer，其餘結構與空間流對齊。

### 7️⃣ 克服小資料集：Multi-Task Learning

影片資料集如 HMDB-51 規模較小，直接訓練容易 overfit。

論文修改 network head，使模型擁有兩個獨立的 softmax classification layers，同時對 UCF-101 與 HMDB-51 進行分類。這種 multi-task learning 發揮 regularization 效果，讓兩個資料集共享影片特徵表徵。

### 8️⃣ 分數融合技術：Fusion Methods

論文比較了 averaging prediction scores 與訓練 multiclass linear SVM 兩種融合方法。

實驗結果顯示，在串接並進行 L2 normalization 的 softmax scores 上訓練 SVM，能取得較好的 late-fusion accuracy。

### 9️⃣ 測試期的群體投票機制：Testing Strategy

測試時，系統會從整部影片等間隔抽取 25 個 frames。

每個 frame 會裁剪四個角落與一個中心區域，並加入水平翻轉，共產生：

$$
25 \times 10 = 250
$$

個 network inputs。最後將 250 個 prediction scores 平均，作為影片的最終預測結果。

### 🔟 模型的盲點與改進方向

雖然輸入端使用 optical flow 捕捉運動，但網路內部的 spatial pooling 並未沿著 motion trajectories 進行，因此可能忽略 spatio-temporal tubes 的局部特徵。

此外，mean-flow subtraction 只能處理較簡單的 camera motion；面對複雜相機移動時，仍需要更明確的 global-motion compensation。

---

## 一句話記憶

Two-Stream ConvNet 的核心是分別用 spatial stream 學外觀、temporal stream 學 optical-flow motion，再透過 late fusion 組合兩種互補訊息進行影片動作辨識。

**Gemini 對話連結**: https://gemini.google.com/share/246c8b4f410d
