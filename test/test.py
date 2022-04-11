#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   test.py
@Contact :   haoheliu@gmail.com
@License :   (C)Copyright 2020-2100

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
9/14/21 11:02 AM   Haohe Liu      1.0         None
"""

import git
import os
import sys
import librosa
import numpy as np
import torch

git_root = git.Repo("", search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(git_root)
from voicefixer import VoiceFixer, Vocoder

os.makedirs(os.path.join(git_root, "test/utterance/output"), exist_ok=True)


def check(fname):
    """
    check if the output is normal
    """
    output = os.path.join(git_root, "test/utterance/output", fname)
    target = os.path.join(git_root, "test/utterance/target", fname)
    output, _ = librosa.load(output, sr=44100)
    target, _ = librosa.load(target, sr=44100)
    assert np.mean(np.abs(output - target)) < 0.01


# TEST VOICEFIXER
## Initialize a voicefixer
print("Initializing VoiceFixer...")
voicefixer = VoiceFixer()
# Mode 0: Original Model (suggested by default)
# Mode 1: Add preprocessing module (remove higher frequency)
# Mode 2: Train mode (might work sometimes on seriously degraded real speech)
for mode in [0, 1, 2]:
    print("Test voicefixer mode", mode, end=", ")
    print("Using CPU:")
    voicefixer.restore(
        input=os.path.join(
            git_root, "test/utterance/original/original.flac"
        ),  # low quality .wav/.flac file
        output=os.path.join(
            git_root, "test/utterance/output/output_mode_" + str(mode) + ".flac"
        ),  # save file path
        cuda=False,  # GPU acceleration
        mode=mode,
    )
    if mode != 2:
        check("output_mode_" + str(mode) + ".flac")

    if torch.cuda.is_available():
        print("Using GPU:")
        voicefixer.restore(
            input=os.path.join(git_root, "test/utterance/original/original.flac"),
            # low quality .wav/.flac file
            output=os.path.join(
                git_root, "test/utterance/output/output_mode_" + str(mode) + ".flac"
            ),
            # save file path
            cuda=True,  # GPU acceleration
            mode=mode,
        )
    if mode != 2:
        check("output_mode_" + str(mode) + ".flac")
    print("Pass")

# TEST VOCODER
## Initialize a vocoder
print("Initializing 44.1kHz speech vocoder...")
vocoder = Vocoder(sample_rate=44100)

### read wave (fpath) -> mel spectrogram -> vocoder -> wave -> save wave (out_path)
print("Test vocoder using groundtruth mel spectrogram...")
print("Using CPU:")
vocoder.oracle(
    fpath=os.path.join(git_root, "test/utterance/original/p360_001_mic1.flac"),
    out_path=os.path.join(git_root, "test/utterance/output/oracle.flac"),
    cuda=False,
)  # GPU acceleration

check("oracle.flac")

if torch.cuda.is_available():
    print("Using GPU:")
    vocoder.oracle(
        fpath=os.path.join(git_root, "test/utterance/original/p360_001_mic1.flac"),
        out_path=os.path.join(git_root, "test/utterance/output/oracle.flac"),
        cuda=True,
    )  # GPU acceleration
# Another interface
# vocoder.forward(mel=mel)
check("oracle.flac")

print("Pass")
