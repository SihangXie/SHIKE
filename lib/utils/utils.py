# -*- coding:utf-8 -*-
"""
@Project: SHIKE
@File：util.py
@Author：Sihang Xie
@Time：2023/12/22 9:22
@Description：useful auxiliary utils
"""

import logging
import os
from datetime import datetime


def get_logger(args, rank):
    log_dir = os.path.join(args.output_dir, args.cfg_name, 'logs')
    if not os.path.exists(log_dir) and rank == 0:
        os.makedirs(log_dir)
    time_str = datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_name = '{}-{}.log'.format(args.cfg_name, time_str)
    log_file = os.path.join(str(log_dir), log_name)
    # set up logger
    if rank == 0:
        print('====> Creating log file in {}'.format(log_file))
    log_format = '%(asctime)-15s  %(message)s'
    logging.basicConfig(filename=str(log_file), level=logging.INFO if rank == 0 else logging.WARN, format=log_format)
    logger = logging.getLogger()
    if rank > 0:
        return logger
    logging.getLogger().addHandler(logging.StreamHandler())

    logger.info('---------------------Cfg is set as follow--------------------')
    logger.info(args.__dict__)
    logger.info('-------------------------------------------------------------')
    return logger
