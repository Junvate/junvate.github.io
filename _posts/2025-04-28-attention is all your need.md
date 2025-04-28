---
layout:     post
title:      attention is all your need练习
subtitle:   attention is all your need
date:       2025-04-26
author:     Junvate
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - 保研复习
---
# 文献参考
https://arxiv.org/pdf/1706.03762

# 补充的知识点
## BLUE分数是什么？
- BLUE分数是一种采用n-gram重叠度来衡量机器翻译中和人工参考翻译的相似度的指标，它衡量的是与参考翻译的字面相似度，并不能很好地捕捉语义的准确性、流畅性或语法的正确性。
- N-gram 精确率 (N-gram Precision)：计算机器翻译中 n-grams 出现在任一参考翻译中的比率
- 简洁性惩罚 (Brevity Penalty, BP):如果机器翻译的长度显著短于参考翻译的平均长度，BLEU 会施加一个惩罚因子。这是因为单纯追求高精确率可能会导致翻译出非常短但词语都正确的片段，而丢失了原文信息




# 翻译练习
## 摘要原文


### abstract
```
The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder.   
The best performing models also connect the encoder and decoder through an attention mechanism.   
We propose a new simple network architecture, the Transformer,based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely.    

之前主要的序列转换模型都是基于复杂的递归和包含编码器和解码器的卷积神经网络
表现最好的模型还通过注意力机制连接编码器和解码器  
我们发现一个新的简单网络架构，transformer, 完全依赖于注意力机制，无需递归和卷积神经网络

Experiments on two machine translation tasks show these models to
be superior in quality while being more parallelizable and requiring significantly less time to train. 

Our model achieves 28.4 BLEU on the WMT 2014 English to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. 

在两项机器翻译的任务上的实验表明，出这些模型在质量上更优而且可并行化程度更高和所需要的训练时间也大大减小。
我们的模型在MT 2014 英语翻译为德语的翻译任务上达到了28.4的布鲁分数，超过了现在的最佳成绩，包括ensembles超出了2布鲁分数以上，

On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. 

在WMT 2014英语翻译到德语的翻译任务上，我们的模型通过在八个gpu三天半的训练，达到了一个单模型41.8的布鲁分数，达到了世界的一流水平。这只是文献中最佳模型训练成本的一小部分

We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.
我们还通过把它成功应用在英语选区解析的任务上，在大的和有限的训练数据，证明Transformer在其他任务的泛化能力很好。
```

### introduction
```
递归式的神经网络，尤其是长短期记忆和门控式递归神经网络，已经被牢固证明是解决在语言建模和机器翻译等序列模型和转译难题的最先进方法。
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks
in particular, have been firmly established as state of the art approaches in sequence modeling and
transduction problems such as language modeling and machine translation [35, 2, 5]. 

自从那以后，许多人付出巨大的努力推进了递归式模型和编码解码器架构的发展。递归式模型通常沿着输入输出序列的符号位置进行计算。
将位置与计算时间的步长对齐后，他们能产生一系列的隐藏状态ht,作为之前隐藏层状态ht-1和位置t的输入的函数
Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].
Recurrent models typically factor computation along the symbol positions of the input and output sequences. 
Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht-1 and the input for position t. 

这种固有的序列性在训练示例上排除了并行性，这一点在更长的序列长度上变得更加重要，因为内存约束限制了跨示例的批处理
This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.
最近的工作通过参数分解技巧和条件计算在计算效率上取得了巨大的进步，也在后者同时提升了模型表现。
Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. 

然而，顺序计算的基础约束仍然存在，注意力机制在多种任务中已经变成了令人信服的序列模型和翻译模型的整合部分，可在不考虑到她们输入输出序列之间的距离对依赖关系进行建模
The fundamental constraint of sequential computation, however, remains.Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in
the input or output sequences [2, 19]. 


In all but a few cases [27], however, such attention mechanisms 
are used in conjunction with a recurrent network.
In this work we propose the Transformer, a model architecture eschewing recurrence and instead
relying entirely on an attention mechanism to draw global dependencies between input and output.
The Transformer allows for significantly more parallelization and can reach a new state of the art in
translation quality after being trained for as littie as twelve hours on eight P100 GPUs.
```
## 词汇
- dominant 主要的，主导的
- transduction 转换 翻译 转译
- complex recurrent 复杂的递归
- solely 只，仅仅
- propose 提出，建议
- parallelizable 可并行化的
- significantly 极其 显著
- existing best results 现有的最佳成绩
- state-of-the-art 最先进的
- fraction 部分，分数
- literature 文献，文学
- constituency parsing 选区解析
- generalizes 泛化
- in particular 特别是
- firmly 牢固地
- approaches 方法 靠近
- push the boundary of 推动什么的发展 突破什么的边界
- have since 从那以后
- typically 通常 经常
- factor computation 进行计算
- along the symbol positions of 沿着符号边界
- as 作为
- precludes 排除
- inherently sequential nature  固有的序列属性（固有的 本质上的）
- critical 重要的 争议的
- memory constraints 内存约束
- factorization tricks 参数分解技巧
- compelling 令人信服的 引人注目的
- without regard to 不需要考虑