## Mini-Chinese-GPT
Recreate a mini GPT model from 0 to 1, 能够在医学领域具备较强的领域性
## 🤖预训练

1. **分词器（Tokenizer）**

   * LLM分词器的构建方式有两种：一种是自己构造词表并训练一个分词器[custom tokenizers](https://github.com/karpathy/llama2.c)，另一种是选择开源模型训练好的分词器，例如ChatGLM2-6B，Llama2等。**关于小模型训练是否需要采用大模型的分词器有一定争论** 本项目选择[ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)的分词器，该词表大小为64793，值得注意的是：这是一个很妙的数字，因为它刚好在uint16的表示范围（0～65535的无符号整数），每一个token只需要两个字节即可表示，当我们的语料较大时候，相比常用的int32可以节省一半的存储空间。

2. **预训练语料（Corpus for pre-training ）**：从LLM技术革命以来，开源中文预训练语料越来越多。收集并处理了以下几个经典数据集：

   | 中文预训练语料                                               | 描述                                                         |
   | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | Wiki中文百科：[wikipedia-cn-20230720-filtered](https://huggingface.co/datasets/pleisto/wikipedia-cn-20230720-filtered) | 中文Wikipedia的数据                                          |
   | BaiduBaiKe：[百度网盘](https://pan.baidu.com/s/1jIpCHnWLTNYabftavo3DVw?pwd=bwvb) 提取码: bwvb | 中文BaiduBaiKe的数据                                         |
   | C4_zh：[百度网盘 part1](https://pan.baidu.com/s/18O2Tj_PPB718K8gnaWrWUQ) 提取码：zv4r；[百度网盘 part2](https://pan.baidu.com/s/11PTgtUfFXvpNkOige9Iw4w) 提取码：sb83；[百度网盘 part3](https://pan.baidu.com/s/1248QfTS8QHPojYW-0fd5jQ) 提取码：l89d | C4是可用的最大语言数据集之一，收集了来自互联网上超过3.65亿个域的超过1560亿个token。C4_zh是其中的一部分 |
   | WuDaoCorpora：[智源研究院BAAI：WuDaoCorpora Text文本预训练数据集](https://data.baai.ac.cn/details/WuDaoCorporaText) | 中文悟道开源的200G数据                                       |
   | shibing624/medical：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical/tree/main) | 源自shibing624的一部分医学领域的预训练数据                   |
   | SkyPile-150B：[SkyPile-150B](https://www.modelscope.cn/datasets/modelscope/SkyPile-150B/summary) | SkyPile-150B是一个全面、大规模的中文数据集，专门为大型语言模型的预训练而设计。它来源于广泛的可供公众访问的中国互联网网页。严格的过滤、广泛的重复数据消除和彻底的敏感数据过滤已被用于确保其质量。此外，我们还使用了诸如fastText和BERT之类的高级工具来过滤低质量的数据。SkyPile-150B数据集的公共访问部分包括大约2.33亿个独特的网页，每个网页平均包含1000多个汉字。总的来说，该数据集包括大约1500亿个令牌和620G的纯文本数据。 |
   | [Clinical Guidelines](https://huggingface.co/datasets/epfl-llm/guidelines) | 医学临床指南                                                 |
   | M[edsci  Guideline](https://www.medsci.cn/guideline/)（部分购买） | 梅斯临床指南                                                 |
   | PubMed and PubMed Central papers S2ORC数据集医学文献数据     | 医学文献库                                                   |
   | [cMeKG](https://github.com/king-yyf/CMeKG_tools)             | 知识图谱的数据                                               |
   | [cMedQA2](https://github.com/zhangsheng93/cMedQA2)           | 医学问答数据                                                 |
   | [中医药指令数据集](https://huggingface.co/datasets/michaelwzhu/ChatMed_TCM_Dataset) [ChatMed_TCM_Dataset](https://huggingface.co/datasets/michaelwzhu/ChatMed_TCM_Dataset) **[Chinese-medical-dialogue-data](https://github.com/Toyhom/Chinese-medical-dialogue-data)** | 医学问诊数据                                                 |

3.**多模态预训练语料（Corpus for pre-training ）**

* 收集vLM的数据集
* 模型结构，MiniGPT4/5框架、Qwenvl、CogvLM等MMLM结构

### 预训练语料预处理
数据预处理采取GPT的通用做法，对语料进行提前分词，对一个样本做完分词后在末尾加上一个结束符号`<eos>`，与下一个样本区分开。然后将所有的训练语料拼接成一个数组（np.uint16）以.bin二进制格式存储到磁盘上。如果语料过大，避免内存溢出，可以选择mmap格式。
```bash
#脚本里面每一个函数对应一个语料库的预处理，搭建新加语料可以自行扩展。
python data_process.py
#运行结束后，会在./data目录下产生pretrain_data.bin文件
```
### 预训练

#### 训练策略

**采用MiniCPM训练策略**

* 先使用普通的数据进行快速训练
* 精炼的通识数据进行稳定阶段的训练
* 使用精炼医学文献数据进行退火训练
* 医学临床指南数据+部分疾病文档数据在进行一次收敛阶段的训练

```bash
sh script/pretrain.sh
#运行结束后，预训练模型会保存在‘out/pretrain’文件夹中
```

## 💡SFT指令微调

1. **SFT微调数据**：LLM在垂直领域的适应已经是2023年的主格调，因此各个领域的SFT语料和微调模型层出不穷。目前已经有大佬整理并持续更新这方面的[最新进展](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)，

   **通识SFT数据**：

   | SFT语料      | 描述                                                         |
   | ------------ | ------------------------------------------------------------ |
   | GPT4-learned | GPT4 与shareGPT通识指令数据集                                |
   | bell         | 源自BelleGroup的一部分SFT数据。包含约100万条由BELLE项目生成的中文指令数据。 |

   **医学垂直领域SFT数据**：

   | SFT语料                                                      | 描述                                                         |
   | ------------------------------------------------------------ | ------------------------------------------------------------ |
   | shibing624/medical：[shibing624/medical](https://huggingface.co/datasets/shibing624/medical/tree/main) | 源自shibing624。该数据集不仅包含了预训练语料如上文所述，还包含一部分SFT数据。 |
   | HuatuoGPT-sft-data-v1：[HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT-sft-data-v1) | 源自HuatuoGPT的SFT数据                                       |
   | DISC-Med-SFT：[HuatuoGPT-sft-data-v1](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) | DISC-Med-SFT Dataset的子集                                   |
   | ChatMed_Consult-v0.3：[michaelwzhu/ChatMed_Consult-v0.3](https://huggingface.co/datasets/michaelwzhu/ChatMed_Consult_Dataset) | 本数据集, ChatMed-Dataset, 中的query(或者是prompt)来自于互联网上的医疗问诊问题(549,326)，反映了真实世界的不同用户/患者的医疗问诊需求。目前response都是由OpenAI GPT-3.5引擎回答的。 |

### SFT样本构建
因为SFT语料一般较小，我们没必要提前分词，而是在构建Dataloader的时候进行分词构建batch送给模型。所以自行参考dataset_sft.py即可！

基本逻辑如下：
- prompt和answer之间一定要有一个开始符`<bos>`隔开，然后answer后需要一个结束符`<eos>`。
- 计算loss的时候，对prompt部分的loss进行mask，只计算answer部分的loss即可。

```bash
#脚本里面针对alpaca-zh和bell两个SFT语料进行处理，搭建新加SFT语料可以自行扩展。
python sft_data_process.py
#运行结束后，会在./sft_data目录下产生sft_data.csv文件
```
### 全面微调（Full Fine-tuning）
```bash
#在该screen下执行微调代码
sh script/sft.py
#运行结束后，SFT模型会保存在‘out/sft’文件夹中
```