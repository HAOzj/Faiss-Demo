# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on OCT 29, 2020

@author: woshihaozhaojun@sina.com
"""
import faiss
import numpy as np

from utils import print_run_time

# 向量维度
d = 64

# 待索引向量数量
nv = 10000000

# 查询向量数量
nq = 10000

# 随机种子确定
np.random.seed(1234)

# 初始化带搜索向量和查询向量
xb = np.random.random((nv, d)).astype('float32')
xb[:, 0] += np.arange(nv) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


@print_run_time
def test_brutal_force(k=4, queries=xq):
    print("------开始测试暴力搜索------")

    @print_run_time
    def train(d=d):
        ind = faiss.IndexFlatL2(d)
        print(ind.is_trained)
        return ind

    index = train()

    @print_run_time
    def add():
        """索引中添加向量"""
        index.add(xb)

    add()
    print(f"ntotal={index.ntotal}")

    @print_run_time
    def search():
        index.search(queries, k)

    search()


@print_run_time
def test_ivf(nprobe_list, k=4, queries=xq):
    print("\n------开始测试IVF------")
    # 聚类的数量,论文中的k
    nlist = 100

    # 量化器索引
    quantizer = faiss.IndexFlatL2(d)

    # 指定用L2距离进行搜索，若不指定默认为內积
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    assert not index.is_trained

    @print_run_time
    def train():
        """索引训练"""
        index.train(xb)

    train()

    assert index.is_trained

    @print_run_time
    def add():
        index.add(xb)

    add()

    @print_run_time
    def search(nprobe):
        index.nprobe = nprobe
        print(f"nprobe={nprobe}")
        return index.search(queries, k)

    for nprobe in nprobe_list:
        _, _ = search(nprobe)


def main():
    test_brutal_force()
    test_ivf(nprobe_list=[1, 5, 10, 50, 100])


if __name__ == "__main__":
    main()
