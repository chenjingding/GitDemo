# 邮件分类项目  

## 代码核心功能说明  

该仓库实现了一个基于多项式朴素贝叶斯分类器的邮件分类系统。该系统使用朴素贝叶斯方法对电子邮件进行分类，能够有效区分垃圾邮件和正常邮件。  

### 算法基础  

本项目采用多项式朴素贝叶斯分类器，基于以下核心假设和原理：  

1. **条件概率与特征独立性假设**：  
   - 朴素贝叶斯分类器假设给定类别时特征之间是条件独立的。这意味着特征的出现与其他特征无关，这大大简化了模型的构建与计算。  
   
2. **贝叶斯定理的应用**：  
   - 在邮件分类任务中，我们可以利用贝叶斯定理来计算某封邮件属于某一类别的概率。具体公式为：  

$$
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
$$

  其中， $C$ 表示类别， $X$ 表示特征向量。在邮件分类中， $P(X|C)$ 是基于特征的条件概率, $P(C)$ 是类别的先验概率，而 $P(X)$ 是特征的边际概率。  

### 数据处理流程  

在邮件分类的准备过程中，我们依赖以下数据处理步骤：  

1. **分词处理**：  
   - 将原始邮件文本切分为单词或词组，以形成特征向量。常用的分词工具有 NLTK 或 spaCy，这些工具可以帮助提取文本中有意义的单元。  

2. **停用词过滤**：  
   - 移除如 "和"、"的"、"了" 等频繁出现但对分类没有实际意义的词汇。停用词列表通常通过标准库获取，也可以根据特定任务进行自定义。  

### 特征构建过程  

邮件分类中的特征构建策略主要包括两种方法：  

1. **高频词特征选择**：  
   - 该方法通过选取训练集中高频出现的词作为特征。这种方法简单易实现，但可能会忽略信息量较小的稀有词。  

   - 数学表达为：选取前 $K$ 个频率最高的词。  

2. **TF-IDF 特征加权**：  
   - TF-IDF（Term Frequency-Inverse Document Frequency）不仅考虑词的频率，还综合考虑词在整个语料库中的重要性。通过对单词频率进行加权，能够更好地反映某个词在语料库中的相对重要性。  

   - 数学公式：  

$$
\text{TF-IDF}(w, d) = \text{TF}(w, d) \times \text{IDF}(w)  
$$

  其中,  $\text{TF}(w, d)$ 是词 $w$ 在文档 $d$ 中出现的频率， $\text{IDF}(w) = \log\left(\frac{N}{df(w)}\right)$ 是逆文档频率， $N$ 为总文档数， $df(w)$ 为包含词  $w$ 的文档数量。  

### 高频词/TF-IDF 两种特征模式的切换方法  

在本项目中，高频词特征选择和 TF-IDF 特征加权可以通过调整配置参数进行切换。具体方法如下：  

- **高频词特征**：  
   - 设置参数 `feature_selection = 'high_frequency'`，系统将使用高频词特征进行训练。  

- **TF-IDF 特征**：  
   - 设置参数 `feature_selection = 'tfidf'`，系统将启用 TF-IDF 加权。  



<img src="https://github.com/chenjingding/GitDemo/blob/master/6.png">                     
