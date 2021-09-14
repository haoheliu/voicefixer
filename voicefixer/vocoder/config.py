import torch
import numpy as np
import os
from voicefixer.tools.path import root_path
class Config:
    @classmethod
    def refresh(cls, sr):
        if(sr == 44100):
            Config.ckpt = os.path.join(os.path.expanduser('~'),".cache/voicefixer/synthesis_module/44100/model.ckpt-1490000_trimed.pt")
            Config.cond_channels = 512
            Config.m_channels = 768
            Config.resstack_depth = [8,8,8,8]
            Config.channels = 1024
            Config.cin_channels=128
            Config.upsample_scales = [7, 7, 3, 3]
            Config.num_mels = 128
            Config.n_fft = 2048
            Config.hop_length = 441
            Config.sample_rate=44100
            Config.fmax=22000
            Config.mel_win = 128
            Config.local_condition_dim = 128
        else:
            raise RuntimeError("Error: Vocoder currently only support 44100 samplerate.")

    ckpt = os.path.join(os.path.expanduser('~'),".cache/voicefixer/synthesis_module/44100/model.ckpt-1490000_trimed.pt")
    m_channels = 384
    bits = 10
    opt = "Ralamb"
    cond_channels=256
    clip = 0.5
    num_bands = 1
    cin_channels = 128
    upsample_scales = [7, 7, 3, 3]
    filterbands = "test/filterbanks_4bands.dat"
    ##For inference
    tag = ""
    min_db=-115
    num_mels = 128
    n_fft = 2048
    hop_length = 441
    win_size = None
    sample_rate = 44100
    frame_shift_ms = None

    trim_fft_size = 512
    trim_hop_size = 128
    trim_top_db = 23

    signal_normalization = True
    allow_clipping_in_normalization = True
    symmetric_mels = True
    max_abs_value = 4.

    preemphasis = 0.85
    min_level_db = -100
    ref_level_db = 20
    fmin = 50
    fmax = 22000
    power = 1.5
    griffin_lim_iters = 60
    rescale = False
    rescaling_max = 0.95
    trim_silence = False
    clip_mels_length = True
    max_mel_frames = 2000

    mel_win = 128
    batch_size = 24
    g_learning_rate = 0.001
    d_learning_rate = 0.001
    warmup_steps = 100000
    decay_learning_rate = 0.5
    exponential_moving_average = True
    ema_decay = 0.99

    reset_opt = False
    reset_g_opt = False
    reset_d_opt = False

    local_condition_dim = 128
    lambda_update_G = 1
    multiscale_D = 3

    lambda_adv = 4.0
    lambda_fm_loss = 0.0
    lambda_sc_loss = 5.0
    lambda_mag_loss = 5.0
    lambda_mel_loss = 50.0
    use_mle_loss = False
    lambda_mle_loss = 5.0

    lambda_freq_loss = 2.0
    lambda_energy_loss = 100.0
    lambda_t_loss = 200.0
    lambda_phase_loss = 100.0
    lambda_f0_loss = 1.0
    use_elu = False
    de_preem = False  # train
    up_org = False
    use_one = True
    use_small_D = False
    use_condnet = True
    use_depreem = False  # inference
    use_msd = False
    model_type = "tfgan"  # or bytewave, frame level vocoder using istft
    use_hjcud = False
    no_skip = False
    out_channels = 1
    use_postnet = False  # wn in postnet
    use_wn = False  # wn in resstack
    up_type = "transpose"
    use_smooth = False
    use_drop = False
    use_shift_scale = False
    use_gcnn = False
    resstack_depth = [6, 6, 6, 6]
    kernel_size = [3, 3, 3, 3]
    channels = 512
    use_f0_loss = False
    use_sine = False
    use_cond_rnn = False
    use_rnn = False

    f0_step = 120
    use_lowfreq_loss = False
    lambda_lowfreq_loss = 1.0
    use_film = False
    use_mb_mr_gan = False

    use_mssl = False
    use_ml_gan = False
    use_mb_gan = True
    use_mpd = False
    use_spec_gan = True
    use_rwd = False
    use_mr_gan = True
    use_pqmf_rwd = False
    no_sine = False
    use_frame_mask = False

    lambda_var_loss = 0.0
    discriminator_train_start_steps = 40000  # 80k
    aux_d_train_start_steps = 40000  # 100k
    rescale_out = 0.40
    use_dist = True
    dist_backend = "nccl"
    dist_url = "tcp://localhost:12345"
    world_size = 1

    mel_weight_torch = torch.tensor([19.40951426, 19.94047336, 20.4859038, 21.04629067,
                               21.62194148, 22.21335214, 22.8210215, 23.44529231,
                               24.08660962, 24.74541882, 25.42234287, 26.11770576,
                               26.83212784, 27.56615283, 28.32007747, 29.0947679,
                               29.89060111, 30.70832636, 31.54828121, 32.41121487,
                               33.29780773, 34.20865341, 35.14437675, 36.1056621,
                               37.09332763, 38.10795802, 39.15039691, 40.22119881,
                               41.32154931, 42.45172373, 43.61293329, 44.80609379,
                               46.031602, 47.29070223, 48.58427549, 49.91327905,
                               51.27863232, 52.68119708, 54.1222372, 55.60274206,
                               57.12364703, 58.68617876, 60.29148652, 61.94081306,
                               63.63501986, 65.37562658, 67.16408954, 69.00109084,
                               70.88850318, 72.82736101, 74.81985537, 76.86654792,
                               78.96885475, 81.12900906, 83.34840929, 85.62810662,
                               87.97005418, 90.37689804, 92.84887686, 95.38872881,
                               97.99777002, 100.67862715, 103.43232942, 106.26140638,
                               109.16827015, 112.15470471, 115.22184756, 118.37439245,
                               121.6122689, 124.93877158, 128.35661454, 131.86761321,
                               135.47417938, 139.18059494, 142.98713744, 146.89771854,
                               150.91684347, 155.0446638, 159.28614648, 163.64270198,
                               168.12035831, 172.71749158, 177.44220154, 182.29556933,
                               187.28286676, 192.40502126, 197.6682721, 203.07516896,
                               208.63088733, 214.33770931, 220.19910108, 226.22363072,
                               232.41087124, 238.76803591, 245.30079083, 252.01064464,
                               258.90261676, 265.98474, 273.26010248, 280.73496362,
                               288.41440094, 296.30489752, 304.41180337, 312.7377183,
                               321.28877878, 330.07870237, 339.10812951, 348.38276173,
                               357.91393924, 367.70513992, 377.76413924, 388.09467408,
                               398.70920178, 409.61813793, 420.81980127, 432.33215467,
                               444.16083117, 456.30919947, 468.78589276, 481.61325588,
                               494.78824596, 508.31969844, 522.2238331, 536.51163441,
                               551.18859414, 566.26142988, 581.75006061, 597.66210737])

    x_orig = np.linspace(1, mel_weight_torch.shape[0], num=mel_weight_torch.shape[0])

    x_orig_torch = torch.linspace(1, mel_weight_torch.shape[0], steps=mel_weight_torch.shape[0])

    @classmethod
    def get_mel_weight(cls, percent = 1, a=18.8927416350036,b=0.0269863588184314):
        b = percent * b
        def func(a, b, x):
            return a * np.exp(b * x)
        return func(a,b,Config.x_orig)

    @classmethod
    def get_mel_weight_torch(cls, percent = 1, a=18.8927416350036,b=0.0269863588184314):
        b = percent * b
        def func(a, b, x):
            return a * torch.exp(b * x)
        return func(a,b,Config.x_orig_torch)



