# Reading Notes

The paper *MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs* was presented by ByteDance and Peking University. As scaling laws push model capability upward, moving from billion-parameter to trillion-parameter model training becomes a complex systems engineering challenge rather than only a modeling problem.

At this extreme scale, several bottlenecks become critical. First, model-state communication creates severe overhead when states are sharded and synchronized across massive GPU clusters. Second, pipeline parallelism introduces pipeline bubbles during warm-up and cool-down phases, leaving GPUs idle. Third, the data pipeline can become CPU-to-GPU or disk-to-memory bound when many workers redundantly read the same data. Fourth, when more than 10,000 GPUs run continuously for weeks, software errors, hardware failures, silent stragglers, and transient network problems become inevitable.

MegaScale addresses these challenges through full-stack algorithm-system co-design. For 3D parallelism, it overlaps computation and communication by breaking GEMM kernels into small chunks, computing chunk $N$ while communicating chunk $N-1$. This hides tensor-parallel communication behind computation. To reduce pipeline bubbles, the system uses large-batch training with LAMB instead of Adam, scaling batch size by 4x while maintaining convergence. This extends the steady phase of pipeline training and eliminates 87.5% of pipeline bubbles.

For data loading, MegaScale uses a tree-based data loader. Since GPUs within the same server share identical inputs, a single dedicated data loader per machine reads data into shared memory, and individual GPUs copy data locally. This removes redundant disk reads. For network optimization, the system uses port splitting, such as dividing one 400G downlink into two 200G downlinks, and schedules communication-intensive nodes under the same rack to reduce ECMP hashing conflicts.

The paper also emphasizes robust training and observability. A heartbeat daemon monitors stdout/stderr logs and RDMA traffic metrics in real time. Because LLM training traffic has highly periodic patterns, abnormal traffic fluctuations can trigger automatic fault localization, node isolation, and fast recovery.

The main lesson is that at 10K+ GPU scale, faults are normal. The goal is not to build a flawless cluster, but to build an automated and resilient fault-tolerant architecture. Model FLOPs Utilization (MFU) becomes more important than raw GPU count, because utilization directly determines end-to-end training speed and cost efficiency. MegaScale shows that successful large-scale training requires algorithm-system co-design and deep observability across CUDA events, data dependencies, communication paths, and hardware behavior.

---

# Learning From AI (Gemini 3.5 Flash)

### 1️⃣ MegaScale 是面向 10,000+ GPU 的 LLM 預訓練生產系統

MegaScale 是一個專為擴展大語言模型預訓練而設計的生產系統，成功將訓練規模擴展至超過 10,000 顆 GPU。

它建立在 Megatron-LM 的開源框架基礎之上，但重點不是單一模型技巧，而是完整的大規模訓練系統工程。

### 2️⃣ MFU 是衡量超大規模訓練效率的核心指標

在硬體效能利用率上，MegaScale 在 12,288 顆 GPU 上訓練 175B 模型時，達成了 55.2% 的 Model FLOPs Utilization (MFU)。

相較於開源基準 Megatron-LM，這帶來約 1.34 倍提升。

### 3️⃣ 核心架構原則：Algorithm-System Co-Design 與 In-Depth Observability

MegaScale 的核心設計有兩個原則。

第一是跨越全棧組件的 algorithm-system co-design，也就是模型演算法、平行策略、網路拓撲、資料載入與 fault recovery 必須一起設計。

第二是深入系統底層的 in-depth observability，也就是不能只看表層 GPU utilization，而要追蹤 CUDA events、通訊模式、資料依賴與 straggler 行為。

### 4️⃣ 3D 平行策略依賴通訊與計算重疊

在 3D parallelism 中，MegaScale 透過 prefetching 隱藏 data-parallel communication，並將 pipeline-parallel 的 send/receive 操作解耦。

在 tensor parallelism 中，它將大型 GEMM 切塊流水線化，使通訊與計算可以時間交疊。

### 5️⃣ LAMB 讓大 Batch Training 降低 Pipeline Bubbles

系統引入 LAMB optimizer，在不損失模型精度的情況下，將 training batch size 放大至 4 倍。

Batch size 放大後，pipeline parallelism 的穩定高載計算時間變長，因此成功消除了 87.5% 的 pipeline bubble 閒置代價。

### 6️⃣ Tree-Based Dataloader 解決資料讀取瓶頸

MegaScale 消除了每顆 GPU 各自讀取資料的冗餘設計。

它改用雙層樹狀架構，由每台主機唯一的專屬 dataloader 將資料讀入 shared memory，再由各 GPU 複製，避免集體爭奪 disk bandwidth。

### 7️⃣ 初始化優化把萬卡啟動時間從 1,047 秒降到 30 秒內

在 collective communication initialization 上，MegaScale 將 PyTorch 預設的單執行緒阻塞式 TCPStore 替換為非阻塞 Redis。

同時，它仔細調整 communication group 的初始化順序以消除 global barriers，將萬卡規模的初始化時間從 1,047 秒縮短到 30 秒以內。

### 8️⃣ Network Topology Optimization 降低 ECMP Hashing Conflicts

為了降低等價多路徑路由 (ECMP) 的 hashing conflicts，MegaScale 在物理拓撲上將 top-of-rack switch 的 400G downlink 拆分為兩個 200G links。

系統也會把 communication-intensive compute nodes 儘量調度在同一個 switch 下，減少跨 rack 通訊衝突。

### 9️⃣ Fast Checkpointing 與 Fault Tolerance 降低訓練中斷成本

MegaScale 設計了兩階段 fast checkpointing。

第一階段由 GPU 在數秒內將狀態寫入 host memory，讓訓練可以快速恢復。第二階段再由 background process 非同步上傳到 HDFS。

復原時，系統由 group representative 讀取 checkpoint 並廣播，降低 I/O 壓力與計算時間損耗。

### 🔟 毫秒級監控與 CUDA Event Analysis 用來找出隱形 Stragglers

MegaScale 建立了毫秒級監控與 CUDA event analysis tools，即時生成分散式視角的效能熱點圖與 timeline traces。

這幫助工程師找出只佔少數比例、但會拖慢整體訓練的 silent stragglers，並移除干擾程式碼，確保數週級別的生產訓練能持續穩定收斂。

---

## 一句話記憶

MegaScale 的核心不是單純堆更多 GPU，而是用 algorithm-system co-design、deep observability、fault tolerance 和 communication/data pipeline optimization，讓 10,000+ GPU 的 LLM 預訓練真正可用、可監控、可恢復。

**Gemini 對話連結**: https://gemini.google.com/share/61cffb28982f
