[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HYYUepIsl2aXsdET6P_AmNVXuWP1MCMf?usp=sharing) [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer)

# VoiceFixer

This package provides: 
- A pretrained 44.1k universal speaker-independent neural vocoder.
- A pretrained *Voicefixer*, which is build based on neural vocoder.

*Voicefixer* aims at the restoration of human speech regardless how serious its degraded. It can handle noise, reveberation, low resolution (2kHz~44.1kHz) and clipping (0.1-1.0 threshold) effect within one model.

[![46dAq1.png](https://z3.ax1x.com/2021/09/26/46dAq1.png)](https://imgtu.com/i/46dAq1)

## Demo

Please visit [demo page](https://haoheliu.github.io/demopage-voicefixer/) to view what voicefixer can do.

## Usage

- Basic example:

```python
# Will automatically download model parameters.
from voicefixer import VoiceFixer
from voicefixer import Vocoder

# Initialize model
voicefixer = VoiceFixer()
# Speech restoration
voicefixer.restore(input="", # input wav file path
                   output="", # output wav file path
                   cuda=False, # whether to use gpu acceleration
                   mode = 0) # You can try out mode 0, 1 to find out the best result

# Universal speaker independent vocoder
vocoder = Vocoder(sample_rate=44100) # Only 44100 sampling rate is supported.

# Convert mel spectrogram to waveform
wave = vocoder.forward(mel=mel_spec) # This forward function is used in the following oracle function.

# Test vocoder using the mel spectrogram of 'fpath', save output to file out_path
vocoder.oracle(fpath="", # input wav file path
               out_path="") # output wav file path
```
## Materials
- Voicefixer training: https://github.com/haoheliu/voicefixer_main.git
- Demo page: https://haoheliu.github.io/demopage-voicefixer/ 
- If you found this repo helpful, please consider citing

>   @misc{liu2021voicefixer,   
>       title={VoiceFixer: Toward General Speech Restoration With Neural Vocoder},   
>       author={Haohe Liu and Qiuqiang Kong and Qiao Tian and Yan Zhao and DeLiang Wang and Chuanzeng Huang and Yuxuan Wang},  
>       year={2021},  
>       eprint={2109.13731},  
>       archivePrefix={arXiv},  
>       primaryClass={cs.SD}  
>   }

[![46dnPO.png](https://z3.ax1x.com/2021/09/26/46dnPO.png)](https://imgtu.com/i/46dnPO)
[![46dMxH.png](https://z3.ax1x.com/2021/09/26/46dMxH.png)](https://imgtu.com/i/46dMxH)










