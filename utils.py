# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Created on OCT 29, 2020

@author: woshihaozhaojun@sina.com
"""
import time
import cProfile
import pstats
import os


def do_cprofile(filename):
    """
    Decorator for function profiling.
    """
    def wrapper(func):
        def profiled_func(*args, **kwargs):
            # Flag for do profiling or not.
            DO_PROF = os.getenv("PROFILING")
            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                # Sort stat by internal time.
                sortby = "tottime"
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result
        return profiled_func
    return wrapper


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
