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


sample_rate = 44100


st.write("Wav player")


w = st.file_uploader("Upload a wav file", type="wav")


if w:
    st.write("Inference : ")

    # choose options
    mode = st.radio(
        "Voice fixer modes (0: original mode, 1: Add preprocessing module 2: Train mode (may work sometimes on seriously degraded speech))",
        [0, 1, 2],
    )
    if torch.cuda.is_available():
        is_cuda = st.radio("Turn on GPU", [True, False])
        if is_cuda != list(voice_fixer._model.parameters())[0].is_cuda:
            device = "cuda" if is_cuda else "cpu"
            voice_fixer._model = voice_fixer._model.to(device)
    else:
        is_cuda = False

    t1 = time.time()

    # Load audio from binary
    audio, _ = librosa.load(w, sr=sample_rate, mono=True)

    # Inference
    pred_wav = voice_fixer.restore_inmem(audio, mode=mode, cuda=is_cuda)

    pred_time = time.time() - t1

    # original audio
    st.write("Original Audio : ")

    st.audio(w)

    # predicted audio
    st.write("Predicted Audio : ")

    # make buffer
    with BytesIO() as buffer:
        soundfile.write(buffer, pred_wav.T, samplerate=sample_rate, format="WAV")
        st.write("Time: {:.3f}s".format(pred_time))
        st.audio(buffer.getvalue(), format="audio/wav")
