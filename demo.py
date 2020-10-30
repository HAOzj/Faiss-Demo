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

# 聚类的数量,论文中的k
nlist = 100

# PQ的子空间的数量
m = 8

# 字节数量
cnt_bits = 8

# 待索引向量数量
ntotal = 1000000

# 查询向量数量
nq = 9999

# 随机种子确定
np.random.seed(1234)

# 初始化带搜索向量和查询向量
xb = np.random.random((ntotal, d)).astype('float32')
xb[:, 0] += np.arange(ntotal) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.
xq = np.concatenate((xb[:1], xq))

show_flag = True
print(f"ntotal={ntotal}, nq={xq.shape[0]}")


@print_run_time
def search(index, queries, k, nprobe=None):
    # 返回queries的k个最近邻
    if nprobe is not None:
        print(f"nprobe={nprobe}")
        index.nprobe = nprobe

    dis, ind = index.search(queries, k)
    if show_flag:
        print(dis[0])
        print(ind[0])


@print_run_time
def add(index, xb=xb):
    index.add(xb)


@print_run_time
def test_brutal_force(k=4, queries=xq):
    print("------开始测试暴力搜索------")

    @print_run_time
    def train(d=d):
        ind = faiss.IndexFlatL2(d)
        return ind

    index = train()
    add(index)
    search(index, queries, k)


@print_run_time
def test_ivf(nprobe_list, k=4, queries=xq):
    print("\n------开始测试IVF,也就是先用K-means对文件聚类,搜索时在聚类中找------")

    # 量化器索引
    quantizer = faiss.IndexFlatL2(d)

    # 指定用L2距离进行搜索，若不指定默认为內积faiss.METRIC_INNER_PRODUCT
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    assert not index.is_trained

    @print_run_time
    def train():
        """索引训练"""
        index.train(xb)

    train()

    assert index.is_trained
    add(index)

    for nprobe in nprobe_list:
        search(index, queries, k, nprobe)


@print_run_time
def test_ivf_pq(nprobe_list, k=4, queries=xq):
    print("\n------开始测试IVF+PQ------")

    # 量化器索引
    quantizer = faiss.IndexFlatL2(d)

    # 指定用L2距离进行搜索，若不指定默认为內积
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, cnt_bits)

    assert not index.is_trained

    @print_run_time
    def train():
        """索引训练"""
        index.train(xb)

    train()

    assert index.is_trained
    add(index)

    for nprobe in nprobe_list:
        search(index, queries, k, nprobe)


def main():
    nprobe_list = [1, 5, 10, 50, 100]
    if show_flag:
        nprobe_list = nprobe_list[:2]
    test_brutal_force()
    test_ivf(nprobe_list)
    test_ivf_pq(nprobe_list)


if __name__ == "__main__":
    main()
