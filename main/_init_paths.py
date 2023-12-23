# -*- coding:utf-8 -*-
"""
@Project: SHIKE
@File：_init_paths.py
@Author：Sihang Xie
@Time：2023/12/22 10:30
@Description：add lib to PYTHONPATH
"""

import sys
import os


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
        try:
            os.environ['PYTHONPATH'] = path + ":" + os.environ['PYTHONPATH']
        except KeyError:
            os.environ['PYTHONPATH'] = path


this_dir = os.path.dirname(__file__)

lib_path = os.path.join(this_dir, '..', 'lib')
add_path(lib_path)
