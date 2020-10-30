# FaissDemo
这个项目用来做简单的faiss demo

# 遇到的问题
### 安装faiss
linux环境下  
避免用pip安装
```
> pip install faiss

> python demo.py
No module named '_swigfaiss'
```

尽量选择conda安装
```
#安装cpu版本
#更新conda
conda update conda

#先安装mkl
conda install mkl

#安装faiss-cpu
conda install faiss-cpu -c pytorch

#测试安装是否成功
python -c "import faiss"
```


