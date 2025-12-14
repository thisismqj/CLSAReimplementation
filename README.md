#### 卷积神经网络 以VGG13为例
###### 简介
![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765683505327-d1ff9224-4604-4920-9f3d-aa972f3453a0.png)

VGG13是牛津大学提出的经典卷积神经网络模型，其核心特征是通过堆叠多个3×3小卷积核来构建深度网络。该模型共13层，包括10个卷积层和3个全连接层。每个卷积层后使用ReLU激活函数，并穿插最大池化层进行下采样。

###### 感受野
**感受野**定义了卷积神经网络中某一层特征图上的一个点，在原始输入图像上所能“看到”的区域大小。

+ 计算公式
    - 输入: 输出特征图上的矩形 ![image](https://cdn.nlark.com/yuque/__latex/5ca8ff6086743949febdb5e39db8a388.svg), 卷积核尺寸![image](https://cdn.nlark.com/yuque/__latex/e765bb428a84bf2f8a1250999c53c9e7.svg), 步长![image](https://cdn.nlark.com/yuque/__latex/8695897a4bd8ad2a2ae852477a252671.svg)
    - 输出: 输入特征图上的矩形 ![image](https://cdn.nlark.com/yuque/__latex/cf562b9aba7890719766b926a72b0084.svg)
    - ![image](https://cdn.nlark.com/yuque/__latex/b467fcf0bcba74750a73a453b5af90ba.svg)
    - ![image](https://cdn.nlark.com/yuque/__latex/a737c8a34319f0827576b61c759d337c.svg)

#### 存算一体加速器
###### 原理
+ 硬件
    - RRAM可通过配置为阵列结构，用作向量-矩阵乘法器。输入电压施加于行线[字线WL与位线BL]，每个RRAM单元中的电流等于所加电压乘以RRAM器件的电导值。根据基尔霍夫定律，列线上检测到的总电流向量，即为输入电压向量与电导矩阵的乘法运算结果。由于RRAM阵列中的计算在模拟域进行，需要借助模数/数模转换器或感应放大器等接口电路来实现模拟信号与数字信号之间的转换。
+ 卷积运算 im2col
    - ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765684110865-fdf47564-3953-451a-919a-b755ccd13616.png)
    - 单个卷积核的不同通道沿同一列展开，每列对应一个卷积核。在每个卷积计算周期中，卷积核权重与对应位置的输入特征图块（称为子特征图）执行乘积累加运算，以生成输出特征图中的一个像素点。

###### 建模
+ 处理单元(PE)
    - 数量
    - 维度(256x256)
+ MVM延迟 (1400ns)
+ 每个处理单元有预先写入的权重，可以并行地执行MVM运算

###### 权重复制
+ 原理
    - 复制权重，提高并行性，提高吞吐量，代价是芯片面积
+ array-duplication
    - 在阵列复制方案中，每个RRAM阵列副本接收不同的待卷积子特征图。由于所有RRAM阵列副本同时执行乘积累加运算，卷积操作的总计算延迟得以降低。
+ SDK
    - ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765685329902-5f9b405c-bdb0-4f16-a053-b9f344583154.png)
    - 在每个计算周期内，对多个子特征图而非单一子特征图同时执行乘积累加运算，从而减少总体计算周期。如图(a)所示，在每个周期中，输入特征图内3×3区域（称为并行窗口）的2×2卷积操作被并行执行。这可通过图(b)配置的RRAM阵列实现：将并行窗口内的子特征图（而非卷积核）展开作为RRAM阵列的输入以执行乘积累加运算。卷积核（例如核1）被复制多次，形成一组复制核群，并分布在多个相邻列上。同一组内的复制核（即图(b)中的四列）共享相同权重（核1a至核1d）。每个复制核沿行方向下移若干行（定义为偏移量），该偏移量由并行窗口尺寸、卷积步长及卷积核尺寸共同决定。
    - 优势
        * 减少DAC，显著减少能耗开销
        * 通过权重复制方式减少阻值误差，增加准确率

#### 跨层调度 CLSA
+ ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765685796193-aee87024-779f-4817-bedf-6fc7c58d3b4a.png)
+ 原理
    - 流水线，一部分计算好的特征图直接参与下一步运算，不用等待该层计算完成，提高PE利用率
+ 流程
    - 确定特征图的分块大小
    - 通过层内依赖和层间依赖建立分块的有向无环图
    - 跑拓扑排序
    - 输出调度表
+ 具体实现
    - 层间依赖
        * 对OFM的每个块b，通过感受野计算公式算出IFM的对应矩形Rect
        * 找到IFM的最小块集合S使得S覆盖Rect
        * S中的所有块向b连一条有向边
    - 层内依赖(权重复制)
        * array duplication
            + 计算一个小块需要![image](https://cdn.nlark.com/yuque/__latex/e73c0226bca1d4f77538288c9803883e.svg)(D是小块尺寸) 个周期，一个小块计算完后才能进行下一个小块的计算，因此将该小块向之后计算的小块连边。
            + 考虑权重复制![image](https://cdn.nlark.com/yuque/__latex/4ae81ab31a25e2fb87fd92fd8c9b056d.svg)则可将IFM划分k个区域，在这些区域上同时进行k个小块的计算(区域之间不连边)。
                - ![image](https://cdn.nlark.com/yuque/__latex/3628fbcf2a7f5d4b17fda0c471aab0a0.svg)
        * SDK
            + 考虑权重复制![image](https://cdn.nlark.com/yuque/__latex/f5233f3e9901abf71f6490d535d21372.svg)，则可在一个周期内完成一个并行窗口(尺寸![image](https://cdn.nlark.com/yuque/__latex/945a64cf861de4314473f5b25b6aa8e8.svg))的卷积计算，计算一个小块需要![image](https://cdn.nlark.com/yuque/__latex/5e07fa45f9180eed1f8551aa31a47c9a.svg)次计算。
            + 直接将小块向之后计算的小块连边。
    - 拓扑排序
+ 结果分析
    - 在VGG13上的结果
    - 网络描述

```plain
# 输入特征图尺寸 (宽 高)
input_shape: 224 224
# 数据分区尺寸 (宽 高)
div_size: 3 3
# 循环展开因子
wdup: 4
# SDK模式 (1=启用, 0=禁用)
sdk: 1
# CLSA模式 (1=启用, 0=禁用)
clsa: 1
# ============================================
# VGG13 网络层定义
# 格式: 层名 步长宽 步长高 填充宽 填充高 核宽 核高
# ============================================
layers:
# Block 1
Conv1 1 1 1 1 3 3
Conv2 1 1 1 1 3 3
MaxPool1 2 2 0 0 2 2

# Block 2
Conv3 1 1 1 1 3 3
Conv4 1 1 1 1 3 3
MaxPool2 2 2 0 0 2 2

# Block 3
Conv5 1 1 1 1 3 3
Conv6 1 1 1 1 3 3
MaxPool3 2 2 0 0 2 2

# Block 4
Conv7 1 1 1 1 3 3
Conv8 1 1 1 1 3 3
MaxPool4 2 2 0 0 2 2

# Block 5
Conv9 1 1 1 1 3 3
Conv10 1 1 1 1 3 3
MaxPool5 2 2 0 0 2 2

```

    - ![image](https://cdn.nlark.com/yuque/__latex/3e321f2154f2f9c38bee9ea0e410676a.svg), 无CLSA, 85245cycles
        * ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765705933466-98dbb7e1-6735-4104-a45a-853dc2c3e8b9.png)
    - ![image](https://cdn.nlark.com/yuque/__latex/3e321f2154f2f9c38bee9ea0e410676a.svg), CLSA, 29335cycles, 2.9x提升相比无CLSA, 1.8x提升相比![image](https://cdn.nlark.com/yuque/__latex/b614307ebb06a1c4c11d4d502ba47bb2.svg)
        * ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765706143869-4b02449f-3b1b-4acf-ab2a-7c790672c32e.png)
    - ![image](https://cdn.nlark.com/yuque/__latex/b614307ebb06a1c4c11d4d502ba47bb2.svg), CLSA, 52803cycles
        * ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765706233890-70313c0b-24ce-4a6a-8d5a-1ab75e125fc2.png)
    - ![image](https://cdn.nlark.com/yuque/__latex/70e72348c188da76fdf4c140f66732ab.svg), CLSA, 17601cycles
        * ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765706413759-7b0bbea8-96bb-4287-9d2a-dd6d1e790307.png)
+ 生成的调度表
    - ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765706744700-1d014035-49a0-4cb3-89d2-ca781e5d1b78.png)
    - ![](https://cdn.nlark.com/yuque/0/2025/png/45388661/1765706771789-8d851cd8-54b8-4aa5-ad98-53f0d64bd1cb.png)

