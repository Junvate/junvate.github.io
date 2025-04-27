---
layout:     post
title:      attention is all your need练习
subtitle:   attention is all your need
date:       2025-04-26
author:     Junvate
header-img: img/post-bg-cook.jpg
catalog: true
tags:
    - 期末复习
---

https://arxiv.org/pdf/1706.03762
# 翻译练习
## 摘要原文


- 原文
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
## 词汇
- dominant 主要的，主导的
- transduction 转换 翻译
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