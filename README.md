# w2vembeddings
This package is main for translate word into vector for nlp embedding.
maybe chinese maybe english, any language you want to use is ok.

This is main for those who will use word2vec in local from txt files,will help them. In particular, this implementation primarily helps to construct a separate word vector matrix for embedding in local NLP tasks.

It's a efficient way for researcher to reuse in their scenes.

this project reference [embeddings](https://github.com/vzhong/embeddings)

## insatll
### github
### pip (recommend)
``` bash
pip install w2vembeddings
``` 
``` bash
git clone https://github.com/LG-1/w2vembeddings.git
cd w2vembeddings
python setup.py install
``` 

## init and manage emb db
```python
from w2vembeddings.managedb import ManageDB
md = ManageDB()
md.list_db()  # 查看有哪些db
md.add_file2db('test', '../data/test_corpos.txt', 10, 8)  # 导入db
md.delete_db('test', 10)  # 删除db
```
![image](images/managedb.PNG)


## Get word vector
```python
import numpy as np
from w2vembeddings.w2vemb import EMB
emb = EMB(name='tencent', dimensions=200)
np.array(emb.get_vector('三生有幸'))
```
![image](images/getvector.PNG)

This speed indicates that if you are in a task, need to build a vector matrix of tens of thousands of words, may be second-order.
## Advantages and disadvantages
### Advantages
    1- Do not need import txt file each time anymore.
    2- Does not take up your running memory(RAM, only need disk sapce, base on sqlite3).
    3- fast
### Disadvantages
    1- Unable to use global information.
   
If you have global information needs, maybe you should reference gensim or else package which load word2vec into RAM once.

## reference
### tencent word2vec
[Tencent Chinese Word2vec](https://ai.tencent.com/ailab/nlp/data/Tencent_AILab_ChineseEmbedding.tar.gz) you can download from here.
[More info](https://ai.tencent.com/ailab/nlp/embedding.html) you can reference.
```python
md.add_file2db('tencent', 'Tencent_AILab_ChineseEmbedding.txt', 200, 8824300)  # that's what i used in previous code for Chinese.
```
### othser embeddings
[Glove](https://nlp.stanford.edu/data/wordvecs/)

[embeddings](https://www.kaggle.com/c/quora-insincere-questions-classification/data)