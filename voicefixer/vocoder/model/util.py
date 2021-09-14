from voicefixer.vocoder.config import Config
import torch
import librosa
import numpy as np

def tr_normalize(S):
    if Config.allow_clipping_in_normalization:
        if Config.symmetric_mels:
            return torch.clip((2 * Config.max_abs_value) * ((S - Config.min_db) / (-Config.min_db)) - Config.max_abs_value, -Config.max_abs_value,
                           Config.max_abs_value)
        else:
            return torch.clip(
                Config.max_abs_value * ((S - Config.min_db) /
                                         (-Config.min_db)), 0,
                Config.max_abs_value)

    assert S.max() <= 0 and S.min() - Config.min_db >= 0
    if Config.symmetric_mels:
        return ((2 * Config.max_abs_value) * ((S - Config.min_db) /
                                               (-Config.min_db)) -
                Config.max_abs_value)
    else:
        return (Config.max_abs_value * ((S - Config.min_db) /
                                         (-Config.min_db)))

def tr_amp_to_db(x):
    min_level = torch.exp(Config.min_level_db / 20 * torch.log(torch.tensor(10.0)))
    min_level = min_level.type_as(x)
    return 20 * torch.log10(torch.maximum(min_level, x))

def normalize(S):
    if Config.allow_clipping_in_normalization:
        if Config.symmetric_mels:
            return np.clip((2 * Config.max_abs_value) * ((S - Config.min_db) /
                                                          (-Config.min_db)) -
                           Config.max_abs_value, -Config.max_abs_value,
                           Config.max_abs_value)
        else:
            return np.clip(
                Config.max_abs_value * ((S - Config.min_db) /
                                         (-Config.min_db)), 0,
                Config.max_abs_value)

    assert S.max() <= 0 and S.min() - Config.min_db >= 0
    if Config.symmetric_mels:
        return ((2 * Config.max_abs_value) * ((S - Config.min_db) /
                                               (-Config.min_db)) -
                Config.max_abs_value)
    else:
        return (Config.max_abs_value * ((S - Config.min_db) /
                                         (-Config.min_db)))

def amp_to_db(x):
    min_level = np.exp(Config.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))

def tr_pre(npy):
    # conditions = torch.FloatTensor(npy).type_as(npy) # to(device)
    conditions = npy.transpose(1, 2)
    l = conditions.size(-1)
    pad_tail = l % 2 + 4
    zeros = torch.zeros([conditions.size()[0], Config.num_mels, pad_tail]).type_as(conditions) + -4.0
    return torch.cat([conditions, zeros], dim=-1)

def pre(npy, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    conditions = npy
    ## padding tail
    if(type(conditions) == np.ndarray):
        conditions = torch.FloatTensor(conditions).unsqueeze(0).to(device)
    else:
        conditions = torch.FloatTensor(conditions.float()).unsqueeze(0).to(device)
    conditions = conditions.transpose(1, 2)
    l = conditions.size(-1)
    pad_tail = l % 2 + 4
    zeros = torch.zeros([1, Config.num_mels, pad_tail]).to(device) + -4.0
    return torch.cat([conditions, zeros], dim=-1)

def load_try(state, model):
    model_dict = model.state_dict()
    try:
        model_dict.update(state)
        model.load_state_dict(model_dict)
    except RuntimeError as e:
        print(str(e))
        model_dict = model.state_dict()
        for k, v in state.items():
            model_dict[k] = v
            model.load_state_dict(model_dict)

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def build_mel_basis():
    return librosa.filters.mel(
        Config.sample_rate,
        Config.n_fft,
        htk=True,
        n_mels=Config.num_mels,
        fmin=0,
        fmax=int(Config.sample_rate // 2))


def linear_to_mel(spectogram):
    _mel_basis = build_mel_basis()
    return np.dot(_mel_basis, spectogram)


if __name__ == "__main__":
    data = torch.randn((3,5,100))
    b = normalize(amp_to_db(data.numpy()))
    a = tr_normalize(tr_amp_to_db(data)).numpy()
    print(a-b)
