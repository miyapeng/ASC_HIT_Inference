# LLM

## 介绍

大语言模型LLM简单说来就是一个神经网络构成的语言(文本输入、文本输出)模型

1. 具体了解可以根据复旦这本书的前两章初步浏览了解：https://intro-llm.github.io/

2. google的科普教程：https://developer.aliyun.com/article/1222998

注：第三题中的LLAMA2-70B就是meta公司推出的开源大模型。可以先简单了解，然后上手一下一些简单的模型试试

## 要学习的工具

1. 机器学习和深度学习基础：[动手学习深度学习](https://zh.d2l.ai/chapter_introduction/index.html)快速浏览了解基本知识，李沐有对应的课

   相信大家都有这方面的基础，如果稍微欠缺，请和大家交流

2. NLP领域的常规模型：CNN, RNN, LSTM, Transformer, Bert, GPT

   a. 了解CNN,RNN,LSTM等常规模型的原理，熟悉一下nlp

   b. 了解预训练的范式：https://zh.d2l.ai/chapter_natural-language-processing-pretraining/index.html

   Transformer和Bert精讲推荐两个：

   1. 月来客栈的原理讲解：https://www.ylkz.life/deeplearning/p10553832/

      ​	                              https://www.ylkz.life/deeplearning/p10631450/

   2. GPT和LLAMA:

      gpt2:https://zhuanlan.zhihu.com/p/57251615(因为3之后都是闭源)

      LLAMA & LLAMA2:https://zhuanlan.zhihu.com/p/653303123             **本项目的重点**



可以根据自己的理解情况再查漏补缺，主要目的是快速对nlp的预训练范式有了解，然后快速掌握LLAMA2的原理，然后就是具体能上手推理（因为单卡远远不狗模型的大小，而且加上数据得传输容易引起显存爆炸，所以得需要大模型推理的优化参照第四点）

3. pytorch框架

   pytorch框架是目前学术界最主流，在大模型方面也最主流的框架

   官方文档：https://pytorch-cn.readthedocs.io/zh/latest/

   合工大老师的pytorch实践课：https://www.bilibili.com/video/BV1Y7411d7Ys/?vd_source=1758ada68448c5de0514bddfdc10b9c9

   （也讲了原理，对小白友好）

4. **大模型训练和推理的加速库或框架**

   accelerate ：https://huggingface.co/blog/accelerate-library

​                               https://github.com/huggingface/accelerate

​       Deepspeed    https://www.deepspeed.ai/getting-started/

​                               https://huggingface.co/docs/transformers/main_classes/deepspeed

​        megatron     https://github.com/NVIDIA/Megatron-LM

​                               https://huggingface.co/docs/accelerate/usage_guides/megatron_lm

## 初步想法

先找一个现成的框架通过模型切片把LLAMA2-70B跑起来，然后再调研有哪些方法能减少模型推理的消耗和延迟

## 学习资料网站

祖传的入门资料，可以参考：https://drive.google.com/file/d/1z32q29kgRd_xo9C-cYxf02hP7MVY1RSW/view?usp=sharing