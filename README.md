# 文件结构如下


```
multilabel-twin-svm
├─ 📁__pycache__
│  ├─ 📄KernelMatrix.cpython-310.pyc
│  ├─ 📄MLTSVM.cpython-310.pyc
│  ├─ 📄MLTSVM_k.cpython-310.pyc
│  ├─ 📄MLTSVM_ova.cpython-310.pyc
│  ├─ 📄Read.cpython-310.pyc
│  ├─ 📄TsvmPlane1.cpython-310.pyc
│  ├─ 📄TsvmPlane2.cpython-310.pyc
│  └─ 📄TwinSvm.cpython-310.pyc
├─ 📁dataset
│  ├─ 📁birds
│  │  ├─ 📄.DS_Store
│  │  ├─ 📄birds-test.csv
│  │  └─ 📄birds-train.csv
│  ├─ 📁emotions
│  │  ├─ 📄.DS_Store
│  │  ├─ 📄emotions-test.csv
│  │  ├─ 📄emotions-train.csv
│  │  └─ 📄emotions.csv
│  ├─ 📁flags
│  │  ├─ 📄.DS_Store
│  │  ├─ 📄README.txt
│  │  ├─ 📄flags-test.csv
│  │  ├─ 📄flags-train.csv
│  │  └─ 📄flags.csv
│  ├─ 📁yeast
│  │  ├─ 📄.DS_Store
│  │  ├─ 📄yeast-test.csv
│  │  ├─ 📄yeast-train.csv
│  │  └─ 📄yeast.csv
│  └─ 📄.DS_Store
├─ 📁notebooks
│  ├─ 📄notebook.md
│  ├─ 📄test.py
│  ├─ 📄损失函数.ipynb
│  └─ 📄验证.ipynb
├─ 📁twsvmlib
│  ├─ 📁__pycache__
│  │  ├─ 📄TwinSvm.cpython-310.pyc
│  │  └─ 📄__init__.cpython-310.pyc
│  ├─ 📄KernelMatrix.py
│  ├─ 📄MLTSVM_k.py
│  ├─ 📄MLTSVM_ova.py
│  ├─ 📄TsvmPlane1.py
│  ├─ 📄TsvmPlane2.py
│  ├─ 📄TwinSvm.py
│  └─ 📄__init__.py
├─ 📁试验结果
│  ├─ 📄kMLTSVM.txt
│  └─ 📄ovaMLTSVM.txt
├─ 📄.DS_Store
├─ 📄Plot单标签.py
├─ 📄Plot多标签.py
├─ 📄README.md
├─ 📄Read.py
├─ 📄TestandPlot.py
└─ 📄性能测试pipline.py
```


# 开源数据集


![20250316153458](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250316153458.png)


### **1. 基础信息类指标**
| 指标名       | 含义                                                                 | 示例（假设数据集）                     |
|--------------|----------------------------------------------------------------------|----------------------------------------|
| `name`       | 数据集名称（如：MedicalImages）                                      | `scene`（经典多标签数据集）             |
| `domain`     | 数据集领域（如：图像、文本、生物信息）                               | `图像分类`                             |
| `instances`  | 样本总数（即行数）                                                   | 1000 个样本                            |


### **2. 特征描述类指标**
| 指标名     | 含义                                                                 | 多标签场景的特殊性                     |
|------------|----------------------------------------------------------------------|----------------------------------------|
| `nominal`  | 分类（标称）特征数量（如：颜色、类别名称）                           | 例如：文本数据中的词性特征（标称）      |
| `numeric`  | 数值型特征数量（如：身高、温度）                                     | 例如：图像的像素值（数值）             |


### **3. 标签核心指标（多标签专属）**
| 指标名       | 含义                                                                 | 计算方式                                  | 示例（10个标签）                |
|--------------|----------------------------------------------------------------------|-------------------------------------------|----------------------------------|
| `labels`     | 总标签数量（即列数）                                                 | 唯一标签的总数                            | 50 个标签（如：图像的多个类别） |
| `cardinality`| **平均每个样本的标签数**（核心指标）                                 | `Σ(每个样本的标签数) / 样本总数`          | 2.3（平均每个样本有2-3个标签）  |
| `density`    | **标签密度**（标准化的稀疏性指标）                                   | `cardinality / labels`                    | 0.046（即4.6%的标签被激活）     |
| `distinct`   | **唯一标签组合的数量**（反映标签组合的多样性）                       | 统计所有样本中不同的标签二值化向量的数量  | 800（1000样本中有200个重复组合）|


### **指标的实际意义**
1. **`cardinality`（基数）**  
   - 反映标签的“平均重叠程度”：值越高，样本越可能属于多个类别（如：新闻分类中`cardinality=3`表示平均每篇新闻属于3个主题）。  
   - 极端情况：`cardinality=1`退化为单标签数据，`cardinality=labels`为全标签样本。

2. **`density`（密度）**  
   - 衡量标签的稀疏性：值越低，标签越稀疏（如：图像标注中`density=0.02`表示仅2%的标签被使用）。  
   - 作用：帮助选择算法（如：稀疏数据适合`ML-KNN`，密集数据适合`BR`算法）。

3. **`distinct`（唯一组合数）**  
   - 体现数据的复杂度：若`distinct=instances`，说明每个样本标签组合唯一（无重复模式）；若`distinct`远小于样本数，说明存在高频标签组合（如：标签0和1常共现）。



