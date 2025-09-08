# RankSearch: A Ranked Retrieval Search Engine

## 项目背景
本项目目标是实现一个基于倒排索引（inverted index with positional information）的搜索引擎。该搜索引擎能够根据用户输入的查询语句，在文档集合中检索并返回相关文档，并按照以下因素进行排序：
1. **查询词覆盖度**（包含多少查询词）
2. **匹配词的接近程度**（平均距离）
3. **查询词顺序保持情况**

排序公式如下：
\[
Score(d) = \alpha \times \frac{\#matched\_terms}{\#total\_query\_terms} + \beta \times \frac{1}{1 + avg\_pair\_distance} + \gamma \times ordered\_pairs
\]
其中默认参数为：  
- α = 1.0  
- β = 1.0  
- γ = 0.1  

---

## 功能实现
本项目实现了以下功能：

### 1. Indexer (index.py)
- 输入：文档集合目录
- 输出：索引文件目录
- 功能：构建带有位置信息的倒排索引，并输出统计信息：
  - 文档总数
  - 处理后的 token 数
  - term 数量
- 示例：
  ```bash
  $ python3 index.py ~/cs6714/Public/data ./MyTestIndex
  Total number of documents: 1000
  Total number of tokens: 268568
  Total number of terms: 268618

### 2. Search (search.py)
- 输入：索引文件目录
- 输出：标准输出（匹配文档 ID，按排名排序）
- 特性：
  - 支持普通查询（按 ID 排序输出）
  - 支持特殊查询：以 "> " 开头时，除了文档 ID 之外，还输出包含最优匹配词的文本行

- 示例：
  ```bash
  $ python3 search.py ./MyTestIndex
  australia technology
  3454
  10
  18
  105

### 3. 排序机制
- 覆盖度：匹配查询词比例越高，分数越高
- 接近度：匹配词间的平均距离越小，分数越高
- 顺序保持：若文档中词序与查询一致，得分提升

### 4. 特殊显示模式
当查询以 "> " 开头时，会输出文档中包含最优匹配词的文本行。例如：
    ```bash
      $ python3 search.py ./MyTestIndex
      > Apples
      > 1361
        The department said stocks of fresh apples in cold storage
---

## 运行指南 (Quick Start)
### 1. 环境要求
- Python 3.9+
- 允许使用的库：
  - Python 标准库
  - NLTK （需预先下载所需资源包）

### 2. 构建索引
    python3 index.py [folder-of-documents] [folder-of-indexes]

### 3. 执行搜索
    python3 search.py [folder-of-indexes]
