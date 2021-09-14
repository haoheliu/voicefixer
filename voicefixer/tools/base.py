import math

import numpy as np
import torch
import os
import torch.fft
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_window(window_size, window_type, square_root_window=True):
    """Return the window"""
    window = {
        'hamming': torch.hamming_window(window_size),
        'hanning': torch.hann_window(window_size),
    }[window_type]
    if square_root_window:
        window = torch.sqrt(window)
    return window


def fft_point(dim):
    assert dim > 0
    num = math.log(dim, 2)
    num_point = 2 ** (math.ceil(num))
    return num_point


def pre_emphasis(signal, coefficient=0.97):
    """Pre-emphasis original signal
    y(n) = x(n) - a*x(n-1)
    """
    return np.append(signal[0], signal[1:] - coefficient * signal[:-1])


def de_emphasis(signal, coefficient=0.97):
    """De-emphasis original signal
    y(n) = x(n) + a*x(n-1)
    """
    length = signal.shape[0]
    for i in range(1, length):
        signal[i] = signal[i] + coefficient * signal[i - 1]
    return signal


def seperate_magnitude(magnitude, phase):
    real = torch.cos(phase) * magnitude
    imagine = torch.sin(phase) * magnitude
    expand_dim = len(list(real.size()))
    return torch.stack((real, imagine), expand_dim)



def stft_single(signal,
         sample_rate=44100,
         frame_length=46,
         frame_shift=10,
         window_type="hanning",
         device=torch.device("cuda"),
         square_root_window=True):
    """Compute the Short Time Fourier Transform.

    Args:
        signal: input speech signal,
        sample_rate: waveform datas sample frequency (Hz)
        frame_length: frame length in milliseconds
        frame_shift: frame shift in milliseconds
        window_type: type of window
        square_root_window: square root window
    Return:
        fft: (n/2)+1 dim complex STFT restults
    """
    hop_length = int(sample_rate * frame_shift / 1000)  # The greater sample_rate, the greater hop_length
    win_length = int(sample_rate * frame_length / 1000)
    # num_point = fft_point(win_length)
    num_point = win_length
    window = get_window(num_point, window_type, square_root_window)
    if ('cuda' in str(device)):
        window = window.cuda(device)
    feat = torch.stft(signal, n_fft=num_point, hop_length=hop_length,
                      win_length=window.shape[0], window=window)
    real, imag = feat[...,0],feat[...,1]
    return real.permute(0,2,1).unsqueeze(1), imag.permute(0,2,1).unsqueeze(1)

def istft(real,imag,length,
          sample_rate=44100,
          frame_length=46,
          frame_shift=10,
          window_type="hanning",
          preemphasis=0.0,
          device=torch.device('cuda'),
          square_root_window=True):
    """Convert frames to signal using overlap-and-add systhesis.
    Args:
        spectrum: magnitude spectrum [batchsize,x,y,2]
        signal: wave signal to supply phase information
    Return:
        wav: synthesied output waveform
    """
    real = real.permute(0,3,2,1)
    imag = imag.permute(0,3,2,1)
    spectrum = torch.cat([real,imag],dim=-1)

    hop_length = int(sample_rate * frame_shift / 1000)
    win_length = int(sample_rate * frame_length / 1000)

    # num_point = fft_point(win_length)
    num_point = win_length
    if ('cuda' in str(device)):
        window = get_window(num_point, window_type, square_root_window).cuda(device)
    else:
        window = get_window(num_point, window_type, square_root_window)

    wav = torch_istft(spectrum, num_point, hop_length=hop_length,
                      win_length=window.shape[0], window=window)
    return wav[...,:length]


def torch_istft(stft_matrix,  # type: Tensor
                n_fft,  # type: int
                hop_length=None,  # type: Optional[int]
                win_length=None,  # type: Optional[int]
                window=None,  # type: Optional[Tensor]
                center=True,  # type: bool
                pad_mode='reflect',  # type: str
                normalized=False,  # type: bool
                onesided=True,  # type: bool
                length=None  # type: Optional[int]
                ):
    # type: (...) -> Tensor

    stft_matrix_dim = stft_matrix.dim()
    assert 3 <= stft_matrix_dim <= 4, ('Incorrect stft dimension: %d' % (stft_matrix_dim))

    if stft_matrix_dim == 3:
        # add a channel dimension
        stft_matrix = stft_matrix.unsqueeze(0)

    dtype = stft_matrix.dtype
    device = stft_matrix.device
    fft_size = stft_matrix.size(1)
    assert (onesided and n_fft // 2 + 1 == fft_size) or (not onesided and n_fft == fft_size), (
            'one_sided implies that n_fft // 2 + 1 == fft_size and not one_sided implies n_fft == fft_size. ' +
            'Given values were onesided: %s, n_fft: %d, fft_size: %d' % (
            'True' if onesided else False, n_fft, fft_size))

    # use stft defaults for Optionals
    if win_length is None:
        win_length = n_fft

    if hop_length is None:
        hop_length = int(win_length // 4)

    # There must be overlap
    assert 0 < hop_length <= win_length
    assert 0 < win_length <= n_fft

    if window is None:
        window = torch.ones(win_length, requires_grad=False, device=device, dtype=dtype)

    assert window.dim() == 1 and window.size(0) == win_length

    if win_length != n_fft:
        # center window with pad left and right zeros
        left = (n_fft - win_length) // 2
        window = torch.nn.functional.pad(window, (left, n_fft - win_length - left))
        assert window.size(0) == n_fft
    # win_length and n_fft are synonymous from here on

    stft_matrix = stft_matrix.transpose(1, 2)  # size (channel, n_frames, fft_size, 2)
    stft_matrix = torch.irfft(stft_matrix, 1, normalized,
                              onesided, signal_sizes=(n_fft,))  # size (channel, n_frames, n_fft)

    assert stft_matrix.size(2) == n_fft
    n_frames = stft_matrix.size(1)

    ytmp = stft_matrix * window.view(1, 1, n_fft)  # size (channel, n_frames, n_fft)
    # each column of a channel is a frame which needs to be overlap added at the right place
    ytmp = ytmp.transpose(1, 2)  # size (channel, n_fft, n_frames)

    eye = torch.eye(n_fft, requires_grad=False,
                    device=device, dtype=dtype).unsqueeze(1)  # size (n_fft, 1, n_fft)

    # this does overlap add where the frames of ytmp are added such that the i'th frame of
    # ytmp is added starting at i*hop_length in the output
    y = torch.nn.functional.conv_transpose1d(
        ytmp, eye, stride=hop_length, padding=0)  # size (channel, 1, expected_signal_len)

    # do the same for the window function
    window_sq = window.pow(2).view(n_fft, 1).repeat((1, n_frames)).unsqueeze(0)  # size (1, n_fft, n_frames)
    window_envelop = torch.nn.functional.conv_transpose1d(
        window_sq, eye, stride=hop_length, padding=0)  # size (1, 1, expected_signal_len)

    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    assert y.size(2) == expected_signal_len
    assert window_envelop.size(2) == expected_signal_len

    half_n_fft = n_fft // 2
    # we need to trim the front padding away if center
    start = half_n_fft if center else 0
    end = -half_n_fft if length is None else start + length

    y = y[:, :, start:end]
    window_envelop = window_envelop[:, :, start:end]

    # check NOLA non-zero overlap condition
    window_envelop_lowest = window_envelop.abs().min()
    assert window_envelop_lowest > 1e-11, ('window overlap add min: %f' % (window_envelop_lowest))

    y = (y / window_envelop).squeeze(1)  # size (channel, expected_signal_len)

    if stft_matrix_dim == 3:  # remove the channel dimension
        y = y.squeeze(0)
    return y
