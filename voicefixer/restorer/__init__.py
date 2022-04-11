#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   __init__.py.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/14/21 12:31 AM   Haohe Liu      1.0         None
"""

import os
import torch
import urllib.request

meta = {
    "voicefixer_fe": {
        "path": os.path.join(
            os.path.expanduser("~"),
            ".cache/voicefixer/analysis_module/checkpoints/vf.ckpt",
        ),
        "url": "https://zenodo.org/record/5600188/files/vf.ckpt?download=1",
    },
}

if not os.path.exists(meta["voicefixer_fe"]["path"]):
    os.makedirs(os.path.dirname(meta["voicefixer_fe"]["path"]), exist_ok=True)
    print("Downloading the main structure of voicefixer")

    urllib.request.urlretrieve(
        meta["voicefixer_fe"]["url"], meta["voicefixer_fe"]["path"]
    )
    print(
        "Weights downloaded in: {} Size: {}".format(
            meta["voicefixer_fe"]["path"],
            os.path.getsize(meta["voicefixer_fe"]["path"]),
        )
    )

    # cmd = "wget "+ meta["voicefixer_fe"]['url'] + " -O " + meta["voicefixer_fe"]['path']
    # os.system(cmd)
    # temp = torch.load(meta["voicefixer_fe"]['path'])
    # torch.save(temp['state_dict'], os.path.join(os.path.expanduser('~'), ".cache/voicefixer/analysis_module/checkpoints/vf.ckpt"))
