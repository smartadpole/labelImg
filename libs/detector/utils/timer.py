#!/usr/bin/python3 python
# encoding: utf-8
'''
@author: 孙昊
@contact: smartadpole@163.com
@file: timer.py
@time: 2021/2/10 下午3:33
@desc:
'''

from time import time

class Timer():
    def __init__(self):
        self.time = time()

    def Timing(self, message:str):
        print("{}: {:.3f}s".format(message.rjust(20, " "), time()-self.time))
        self.Update()

    def Update(self):
        self.time = time()
