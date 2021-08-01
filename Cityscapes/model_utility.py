import os
import re
import sys
import time
import json
import numpy as np
from scipy import misc
from collections import Counter

import torch
import torch.nn as nn
import matplotlib.pyplot as plt



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
CITYSCAPES 데이터셋을 위한 함수
def removeline_city
def 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def read_lines(datapath):
    file = open(datapath, "r")
    lines = file.readlines()
    lines = [line.rstrip(" \n") for line in lines]
    return lines


def save_lines(filename, datapath: str):
    f = open(datapath, "w")
    for data in filename:
        f.write(data + "\n") 
    f.close()