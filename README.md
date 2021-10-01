[![arXiv](https://img.shields.io/badge/arXiv-2109.13731-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2109.13731) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HYYUepIsl2aXsdET6P_AmNVXuWP1MCMf?usp=sharing) [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://haoheliu.github.io/demopage-voicefixer)
 
- [VoiceFixer](#voicefixer)
  * [Demo](#demo)
  * [Usage](#usage)
    + [Desktop App](#desktop-app)
    + [Python interface](#python-interface)
    + [Others Features](#others-features)
  * [Materials](#materials)
   
# VoiceFixer

This package provides: 
- A pretrained 44.1k universal speaker-independent neural vocoder.
- A pretrained *Voicefixer*, which is build based on neural vocoder.

*Voicefixer* aims at the restoration of human speech regardless how serious its degraded. It can handle noise, reveberation, low resolution (2kHz~44.1kHz) and clipping (0.1-1.0 threshold) effect within one model.

[![46dAq1.png](https://z3.ax1x.com/2021/09/26/46dAq1.png)](https://imgtu.com/i/46dAq1)

## Demo

Please visit [demo page](https://haoheliu.github.io/demopage-voicefixer/) to view what voicefixer can do.

## Usage

Install voicefixer first:
```shell script
pip install voicefixer
```

### Desktop App

You can test audio samples on your desktop by running website (powered by [streamlit](https://streamlit.io/))
```shell script
# Install additional web package
pip install streamlit
# Run streamlit 
streamlit run test/streamlit.py
```
**Important:** When you run the above command for the first time, the web page may leave blank for several minutes for downloading models. You can checkout the terminal for downloading progresses.  
 

### Python interface

Basic example: 

```python
# Will automatically download model parameters.
from voicefixer import VoiceFixer
from voicefixer import Vocoder

# Initialize model
voicefixer = VoiceFixer()
# Speech restoration

# Mode 0: Original Model (suggested by default)
voicefixer.restore(input="", # input wav file path
                   output="", # output wav file path
                   cuda=False, # whether to use gpu acceleration
                   mode = 0) # You can try out mode 0, 1, 2 to find out the best result
# Mode 1: Add preprocessing module (remove higher frequency)
voicefixer.restore(input="", # input wav file path
                   output="", # output wav file path
                   cuda=False, # whether to use gpu acceleration
                   mode = 1) # You can try out mode 0, 1, 2 to find out the best result
# Mode 2: Train mode (might work sometimes on seriously degraded real speech)
voicefixer.restore(input="", # input wav file path
                   output="", # output wav file path
                   cuda=False, # whether to use gpu acceleration
                   mode = 2) # You can try out mode 0, 1, 2 to find out the best result

# Another similar function
# voicefixer.restore_inmem()

# Universal speaker independent vocoder
vocoder = Vocoder(sample_rate=44100) # Only 44100 sampling rate is supported.

# Convert mel spectrogram to waveform
wave = vocoder.forward(mel=mel_spec) # This forward function is used in the following oracle function.

# Test vocoder using the mel spectrogram of 'fpath', save output to file out_path
vocoder.oracle(fpath="", # input wav file path
               out_path="") # output wav file path
```

### Others Features

- How to use your own vocoder, like pre-trained HiFi-Gan?

First you need to write a following helper function with your model. Similar to the helper function in this repo: https://github.com/haoheliu/voicefixer/blob/main/voicefixer/vocoder/base.py#L35

```shell script
    def convert_mel_to_wav(mel):
        """
        :param non normalized mel spectrogram: [batchsize, 1, t-steps, n_mel]
        :return: [batchsize, 1, samples]
        """
        return wav
```

Then pass this function to *voicefixer.restore*, for example:
```
voicefixer.restore(input="", # input wav file path
                   output="", # output wav file path
                   cuda=False, # whether to use gpu acceleration
                   mode = 0,
                   your_vocoder_func = convert_mel_to_wav)
```

Note: 
- For compatibility, your vocoder should working on 44.1kHz wave with mel frequency bins 128. 
- The input mel spectrogram to the helper function should not be normalized by the width of each mel filter. 

## Materials
- Voicefixer training: https://github.com/haoheliu/voicefixer_main.git
- Demo page: https://haoheliu.github.io/demopage-voicefixer/ 
- If you found this repo helpful, please consider citing

```bib
 @misc{liu2021voicefixer,   
     title={VoiceFixer: Toward General Speech Restoration With Neural Vocoder},   
     author={Haohe Liu and Qiuqiang Kong and Qiao Tian and Yan Zhao and DeLiang Wang and Chuanzeng Huang and Yuxuan Wang},  
     year={2021},  
     eprint={2109.13731},  
     archivePrefix={arXiv},  
     primaryClass={cs.SD}  
 }
```

[![46dnPO.png](https://z3.ax1x.com/2021/09/26/46dnPO.png)](https://imgtu.com/i/46dnPO)
[![46dMxH.png](https://z3.ax1x.com/2021/09/26/46dMxH.png)](https://imgtu.com/i/46dMxH)










