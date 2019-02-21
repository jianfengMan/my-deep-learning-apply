**在深度学习中，Attention 机制从本质上讲和人类的选择性视觉注意力机制类似，核心目标也是从众多信息中选择出对当前任务目标更关键的信息。**

Attention 输出的结果通常为一个 One-Hot 或者 Soft 的软分布，也就对应到了 Soft-Attention 或者 Hard-Attention。这个输出分布表示了上下文信息的关联度，也就是信息的选择性。



### 卷积神经网的 Attention 机制及实现方法

实际上，在计算机视觉任务中，Attention 机制除了用于显著性目标检测，同样可以用来辅助卷积神经网的学习，并将 Attention 作为可微部分，直接添加到网络模型中。通俗来讲，主要包括两种类型的实现，分别为：

- **学习权重分布**：输入数据或者 Feature Map 上的不同位置，对应不同的专注度，也就是不同的权值分布。这个权值分布，会作为权重的方式，乘加到原先的 Feature Map 或者通道等不同位置上。 
- **任务聚焦**：通过将任务分解，设计不同的网络结构（或分支）专注于不同的子任务，重新分配网络的学习能力，从而降低原始任务的难度，使网络更加容易训练。



#### 关于 Seq2Seq+Attention 实现问题

这里我们通常会采用 TensorFlow 作为相应的框架进行模型的搭建和训练测试。而在 TensorFlow 中，也提供了相应的 Attention 机制定义，如下代码所示。

代码一：

```python
# tf.contrib.seq2seq.LuongAttention

    init(
    num_units,
    memory,
    memory_sequence_length=None,
    scale=False,
    probability_fn=None,
    score_mask_value=float('-inf'),
    name='LuongAttention'
    )
```

代码二：

```python
# tf.contrib.seq2seq.BahdanauAttention

    init(
    num_units,
    memory,
    memory_sequence_length=None,
    normalize=False,
    probability_fn=None,
    score_mask_value=float('-inf'),
    name='BahdanauAttention'
    )
```

我们对上面代码中的参数，做以下说明：

- `num_units`：在 Encoder 阶段产生了多个特征向量，它表示每个特征向量的大小；
- memory：一个 Batch 里，Encoder 阶段产生的所有的特征向量，在 RNN Encoder 中，维数为 `[batch_size,max_time,num_units]`，即 Encoder 阶段产生了 `max_time` 个大小为 `num_units` 的特征向量；
- `memory_sequence_length`：记录 memory 中的特征向量的长度，维数是 `[batch_size,]`，令 memory 中超过 `memory_sequence_length` 的值为0；
- scale：是否进行尺度变化；
- normalize：是否进行 Weight Normalization；
- `probability_fn`：将打分函数直接转成概率，默认为 Softmax；
- `score_mask_value`：在将分数传到 `probability_fn` 函数之前的掩码值，有 `Probability_fn` 函数的情况下才会用。

在实际使用的时候，`tf.contrib.seq2seq.BahdanauAttention/tf.contrib.seq2seq.LuongAttention` 会配合 `tf.contrib.seq2seq.AttentionWrapper` 使用。