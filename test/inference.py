#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   inference.py    
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/6/21 3:08 PM   Haohe Liu      1.0         None
"""

from voicefixer import VoiceFixer
from voicefixer import Vocoder

from os.path import isdir, exists, basename, join
from argparse import ArgumentParser
from progressbar import *

parser = ArgumentParser()

parser.add_argument(
    "-i",
    "--input_file_path",
    default="/Users/liuhaohe/Desktop/test.wav",
    help="The .wav file or the audio folder to be processed",
)
parser.add_argument(
    "-o", "--output_path", default=".", help="The output dirpath for the results"
)
parser.add_argument("-m", "--models", default="voicefixer_fe")
parser.add_argument(
    "--cuda", type=bool, default=False, help="Whether use GPU acceleration."
)
args = parser.parse_args()

if __name__ == "__main__":
    voicefixer = VoiceFixer()

    if not isdir(args.output_path):
        raise ValueError("Error: output path need to be a directory, not a file name.")
    if not exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    if not isdir(args.input_file_path):
        assert (
            args.input_file_path[-3:] == "wav" or args.input_file_path[-4:] == "flac"
        ), (
            "Error: invalid file "
            + args.input_file_path
            + ", we only accept .wav and .flac file."
        )
        output_path = join(args.output_path, basename(args.input_file_path))
        print("Start Prediction.")
        voicefixer.restore(
            input=args.input_file_path, output=output_path, cuda=args.cuda
        )
    else:
        files = os.listdir(args.input_file_path)
        print("Found", len(files), "files in", args.input_file_path)
        widgets = [
            "Performing Resotartion",
            " [",
            Timer(),
            "] ",
            Bar(),
            " (",
            ETA(),
            ") ",
        ]
        pbar = ProgressBar(widgets=widgets).start()
        print("Start Prediction.")
        for i, file in enumerate(files):
            if not file[-3:] == "wav" and not file[-4:] == "flac":
                print(
                    "Ignore file",
                    file,
                    " unsupported file type. Please use wav or flac format.",
                )
                continue
            output_path = join(args.output_path, basename(file))
            voicefixer.restore(
                input=join(args.input_file_path, file),
                output=output_path,
                cuda=args.cuda,
            )
            pbar.update(int((i / (len(files))) * 100))
    print("Congratulations! Prediction Complete.")
