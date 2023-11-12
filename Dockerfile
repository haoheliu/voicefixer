FROM python:3.11-bookworm

ARG USERNAME=voicefixer
ARG USER_UID=1000
ARG USER_GID=1000
ARG WORKDIR_PATH=/opt/voicefixer
ENV PYTHONUNBUFFERED=1

RUN pip install numpy==1.26.1 librosa==0.10.1 pytz progressbar mpmath zipp watchdog validators tzlocal \
    tzdata tornado toolz toml tenacity sympy smmap six rpds-py pyyaml pyparsing pygments pyarrow protobuf \
    pillow nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-nccl-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-cupti-cu12 nvidia-cublas-cu12 networkx mdurl \
    MarkupSafe kiwisolver fsspec fonttools filelock cycler contourpy click cachetools blinker attrs triton \
    referencing python-dateutil nvidia-cusparse-cu12 nvidia-cudnn-cu12 markdown-it-py jinja2 importlib-metadata \
    gitdb rich pydeck pandas nvidia-cusolver-cu12 matplotlib jsonschema-specifications GitPython torchlibrosa \
    torch jsonschema altair streamlit

RUN mkdir -p ${WORKDIR_PATH}

ADD . $WORKDIR_PATH
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -d ${WORKDIR_PATH} $USERNAME \
    && chown -R $USERNAME:$USERNAME ${WORKDIR_PATH}

WORKDIR ${WORKDIR_PATH}
USER $USERNAME
ENV PATH="${PATH}:${WORKDIR_PATH}/.local/bin"

RUN pip install .
RUN voicefixer --weight_prepare

ENTRYPOINT ["voicefixer"]