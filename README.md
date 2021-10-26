[![arXiv](https://img.shields.io/badge/arXiv-2109.13731-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2109.13731) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HYYUepIsl2aXsdET6P_AmNVXuWP1MCMf?usp=sharing) [![PyPI version](https://badge.fury.io/py/voicefixer.svg)](https://badge.fury.io/py/voicefixer) [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://haoheliu.github.io/demopage-voicefixer)
 
- [VoiceFixer](#voicefixer)
  - [Demo](#demo)
  - [Usage](#usage)
    - [Desktop App](#desktop-app)
    - [Python Examples](#python-examples)
    - [Others Features](#others-features)
  - [Materials](#materials)
  
# VoiceFixer

*Voicefixer* aims at the restoration of human speech regardless how serious its degraded. It can handle noise, reveberation, low resolution (2kHz~44.1kHz) and clipping (0.1-1.0 threshold) effect within one model.

This package provides: 
- A pretrained *Voicefixer*, which is build based on neural vocoder.
- A pretrained 44.1k universal speaker-independent neural vocoder.

![main](test/figure.png)

## Demo

Please visit [demo page](https://haoheliu.github.io/demopage-voicefixer/) to view what voicefixer can do.

## Usage
### Desktop App

First, install voicefixer via pip:
```shell script
pip install voicefixer==0.0.16
```

You can test audio samples on your desktop by running website (powered by [streamlit](https://streamlit.io/))

1. Clone the repo first.
```shell script
git clone https://github.com/haoheliu/voicefixer.git
cd voicefixer
```
2. Initialize and start web page.
```shell script
# Install additional web package
pip install streamlit
# Run streamlit 
streamlit run test/streamlit.py
```
**Important:** When you run the above command for the first time, the web page may leave blank for several minutes for downloading models. You can checkout the terminal for downloading progresses.  
 

### Python Examples 

First, install voicefixer via pip:
```shell script
pip install voicefixer==0.0.16
```

Then run the following scripts for a test run:

```shell script
git clone https://github.com/haoheliu/voicefixer.git; cd voicefixer
python3 test/test.py # test script
```
We expect it will give you the following output:
```shell script
Initializing VoiceFixer...
Test voicefixer mode 0, Pass
Test voicefixer mode 1, Pass
Test voicefixer mode 2, Pass
Initializing 44.1kHz speech vocoder...
Test vocoder using groundtruth mel spectrogram...
Pass
```
*test/test.py* mainly contains the test of the following two APIs:
- voicefixer.restore
- vocoder.oracle

```python
...

# TEST VOICEFIXER
## Initialize a voicefixer
print("Initializing VoiceFixer...")
voicefixer = VoiceFixer()
# Mode 0: Original Model (suggested by default)
# Mode 1: Add preprocessing module (remove higher frequency)
# Mode 2: Train mode (might work sometimes on seriously degraded real speech)
for mode in [0,1,2]:
    print("Testing mode",mode)
    voicefixer.restore(input=os.path.join(git_root,"test/utterance/original/original.flac"), # low quality .wav/.flac file
                       output=os.path.join(git_root,"test/utterance/output/output_mode_"+str(mode)+".flac"), # save file path
                       cuda=False, # GPU acceleration
                       mode=mode)
    if(mode != 2):
        check("output_mode_"+str(mode)+".flac")
    print("Pass")

# TEST VOCODER
## Initialize a vocoder
print("Initializing 44.1kHz speech vocoder...")
vocoder = Vocoder(sample_rate=44100)

### read wave (fpath) -> mel spectrogram -> vocoder -> wave -> save wave (out_path)
print("Test vocoder using groundtruth mel spectrogram...")
vocoder.oracle(fpath=os.path.join(git_root,"test/utterance/original/p360_001_mic1.flac"),
               out_path=os.path.join(git_root,"test/utterance/output/oracle.flac"),
               cuda=False) # GPU acceleration

...
```

You can clone this repo and try to run test.py inside the *test* folder.

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










