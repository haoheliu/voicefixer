#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __init__.py.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/14/21 12:31 AM   Haohe Liu      1.0         None
'''

import os

meta = {
    "voicefixer_fe": {
        "path": os.path.join(os.path.expanduser('~'), ".cache/voicefixer/analysis_module/checkpoints/epoch=15_trimed_bn.ckpt"),
        "url": "https://zenodo.org/record/5464142/files/epoch%3D15_trimed.ckpt?download=1",
    },
}

if (not os.path.exists(meta["voicefixer_fe"]['path'])):
    os.makedirs(os.path.dirname(meta["voicefixer_fe"]['path']), exist_ok=True)
    print("Downloading the weight of voicefixer_fe")
    cmd = "wget https://zenodo.org/record/5175846/files/epoch%3D33-val_loss%3D0.4293_trimed.ckpt?download=1 -O " + \
          meta["voicefixer_fe"]['path']
    os.system(cmd)
