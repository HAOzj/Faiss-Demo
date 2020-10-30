# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on OCT 29, 2020

@author: woshihaozhaojun@sina.com
"""
import time


def print_run_time(func):
    """ 计算时间的装饰器
    """
    def wrapper(*args, **kw):
        local_time = time.time()
        res = func(*args, **kw)
        print("Current function : {function}, time used : {temps}".format(
            function=func.__name__, temps=time.time() - local_time))
        return res
    return wrapper
