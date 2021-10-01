import os
import argparse
import time
import librosa
import soundfile
import streamlit as st
import torch
from io import BytesIO
from voicefixer import VoiceFixer


@st.experimental_singleton
def init_voicefixer():
    return VoiceFixer()


# init with global shared singleton instance
voice_fixer = init_voicefixer()


sample_rate = 44100  # Must be 44100 when using the downloaded checkpoints.


st.write('Wav player')


w = st.file_uploader('Upload a wav file', type='wav')


if w:
    st.write('Inference : ')
    
    # choose options
    mode = st.radio('Voice fixer mode (0: rm high frequency, 1: none, 2: train fixer)', [0, 1, 2])
    if torch.cuda.is_available():
        is_cuda = st.radio('Turn on GPU', [True, False])

    t1 = time.time()
    
    # Load audio from binary
    audio, _ = librosa.load(w, sr=sample_rate, mono=False)

    # Separate.
    sep_wav = voice_fixer.restore_inmem(audio, mode=mode, cuda=is_cuda)

    sep_time = time.time() - t1


    # original audio
    st.write('Original Audio : ')
    
    st.audio(w)

    # predicted audio
    st.write('Predicted Audio : ')

    # make buffer
    with BytesIO() as buffer:
        soundfile.write(buffer, sep_wav.T, samplerate=sample_rate, format='WAV')
        st.write("Time: {:.3f}".format(sep_time))
        st.audio(buffer.getvalue(), format='audio/wav')
