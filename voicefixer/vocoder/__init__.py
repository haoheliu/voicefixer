#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/14/21 1:00 AM   Haohe Liu      1.0         None
"""

import os
from voicefixer.vocoder.config import Config
import urllib.request

if not os.path.exists(Config.ckpt):
    os.makedirs(os.path.dirname(Config.ckpt), exist_ok=True)
    print("Downloading the weight of neural vocoder: TFGAN")
    urllib.request.urlretrieve(
        "https://zenodo.org/record/5469951/files/model.ckpt-1490000_trimed.pt?download=1",
        Config.ckpt,
    )
    print(
        "Weights downloaded in: {} Size: {}".format(
            Config.ckpt, os.path.getsize(Config.ckpt)
        )
    )
    # cmd = "wget https://zenodo.org/record/5469951/files/model.ckpt-1490000_trimed.pt?download=1 -O " + Config.ckpt
    # os.system(cmd)
