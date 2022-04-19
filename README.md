## 文件说明

`/SVD`文件夹下存放有`SVD.py`，`2019211420.txt`，分别为SVD的源代码与任务的输出结果

`/SGNS`文件夹下为SGNS方法的实现

* `SGNS_pytorch.py`，`2019211420.txt`分别为SGNS的pytorch版本源代码与输出结果；
* `words_for_visualization.txt`为用于可视化词向量的词语，可更改
* `SGNS_no_pytorch.py`为不使用pytorch版本的源代码

输出结果为使用pytorch框架计算所得，即`SGNS_pytorch.py`的输出结果。

`文档.pdf`为该项目的文档

`/img`文件夹下存储了文档内用到的图片

## 运行需求

* Python 3.5+
* Pytorch
* toolz
* tqdm
* numpy
* sklearn



> 输入文件应为utf-8编码的用空格分隔的已分词文本。内容应如：我 写完 作业 了。 真是 太 好 了。
