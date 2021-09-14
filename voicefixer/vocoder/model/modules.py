import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from voicefixer.vocoder.config import Config

# From xin wang of nii
class SineGen(torch.nn.Module):
    """ Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(self, samp_rate=24000, harmonic_num=0,
                 sine_amp=0.1, noise_std=0.003,
                 voiced_threshold=0,
                 flag_for_pulse=False):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """ f0_values: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(f0_values.shape[0], f0_values.shape[2], \
                              device=f0_values.device)
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x *2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] -
                                tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(torch.cumsum(rad_values + cumsum_shift, dim=1) \
                              * 2 * np.pi)
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """ sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """

        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, \
                                 device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class LowpassBlur(nn.Module):
    ''' perform low pass filter after upsampling for anti-aliasing'''

    def __init__(self, channels=128, filt_size=3, pad_type='reflect', pad_off=0):
        super(LowpassBlur, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.off = 0
        self.channels = channels

        if (self.filt_size == 1):
            a = np.array([1., ])
        elif (self.filt_size == 2):
            a = np.array([1., 1.])
        elif (self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif (self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif (self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif (self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif (self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            return inp
        return F.conv1d(self.pad(inp), self.filt, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if (pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif (pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif (pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class MovingAverageSmooth(torch.nn.Conv1d):
    def __init__(self, channels, window_len=3):
        """Initialize Conv1d module."""
        super(MovingAverageSmooth, self).__init__(in_channels=channels, out_channels=channels, kernel_size=1,
                                                  groups=channels, bias=False)

        torch.nn.init.constant_(self.weight, 1.0 / window_len)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, data):
        return super(MovingAverageSmooth, self).forward(data)


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode="nearest"):
        """Initialize Stretch2d module.
        Args:
            x_scale (int): X scaling factor (Time axis in spectrogram).
            y_scale (int): Y scaling factor (Frequency axis in spectrogram).
            mode (str): Interpolation mode.
        """
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input tensor (B, C, F, T).
        Returns:
            Tensor: Interpolated tensor (B, C, F * y_scale, T * x_scale),
        """
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode)


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        self.weight.data.fill_(1. / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(self,
                 upsample_scales,
                 nonlinear_activation=None,
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1,
                 use_causal_conv=False,
                 ):
        """Initialize upsampling network module.
        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            interpolate_mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
        """
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            # interpolation layer
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            # conv layer
            assert (freq_axis_kernel_size - 1) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            # nonlinear
            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, C, T).
        Returns:
            Tensor: Upsampled tensor (B, C, T'), where T' = T * prod(upsample_scales).
        """
        c = c.unsqueeze(1)  # (B, 1, C, T)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., :c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)  # (B, C, T')


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(self,
                 upsample_scales=[3, 4, 5, 5],
                 nonlinear_activation="ReLU",
                 nonlinear_activation_params={},
                 interpolate_mode="nearest",
                 freq_axis_kernel_size=1,
                 aux_channels=80,
                 aux_context_window=0,
                 use_causal_conv=False
                 ):
        """Initialize convolution + upsampling network module.
        Args:
            upsample_scales (list): List of upsampling scales.
            nonlinear_activation (str): Activation function name.
            nonlinear_activation_params (dict): Arguments for specified activation function.
            mode (str): Interpolation mode.
            freq_axis_kernel_size (int): Kernel size in the direction of frequency axis.
            aux_channels (int): Number of channels of pre-convolutional layer.
            aux_context_window (int): Context window size of the pre-convolutional layer.
            use_causal_conv (bool): Whether to use causal structure.
        """
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        # To capture wide-context information in conditional features
        kernel_size = aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        # NOTE(kan-bayashi): Here do not use padding because the input is already padded
        self.conv_in = Conv1d(aux_channels, aux_channels, kernel_size=kernel_size, bias=False)
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c : Input tensor (B, C, T').
        Returns:
            Tensor: Upsampled tensor (B, C, T),
                where T = (T' - aux_context_window * 2) * prod(upsample_scales).
        Note:
            The length of inputs considers the context window size.
        """
        c_ = self.conv_in(c)
        c = c_[:, :, :-self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class DownsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor,
                 hp=None,
                 index=0):
        super(DownsampleNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor
        self.skip_conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.index = index
        layer = nn.Conv1d(input_size,
                          output_size,
                          kernel_size=upsample_factor * 2,
                          stride=upsample_factor,
                          padding=upsample_factor // 2 + upsample_factor % 2)

        self.layer = nn.utils.weight_norm(layer)

    def forward(self, inputs):
        B, C, T = inputs.size()
        res = inputs[:, :, ::self.upsample_factor]
        skip = self.skip_conv(res)

        outputs = self.layer(inputs)
        outputs = outputs + skip

        return outputs


class UpsampleNet(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 upsample_factor,
                 hp=None,
                 index=0):

        super(UpsampleNet, self).__init__()
        self.up_type = Config.up_type
        self.use_smooth = Config.use_smooth
        self.use_drop = Config.use_drop
        self.input_size = input_size
        self.output_size = output_size
        self.upsample_factor = upsample_factor
        self.skip_conv = nn.Conv1d(input_size, output_size, kernel_size=1)
        self.index = index
        if self.use_smooth:
            window_lens = [5, 5, 4, 3]
            self.window_len = window_lens[index]

        if self.up_type != "pn" or self.index < 3:
            # if self.up_type != "pn":
            layer = nn.ConvTranspose1d(input_size, output_size, upsample_factor * 2,
                                       upsample_factor,
                                       padding=upsample_factor // 2 + upsample_factor % 2,
                                       output_padding=upsample_factor % 2)
            self.layer = nn.utils.weight_norm(layer)
        else:
            self.layer = nn.Sequential(
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(nn.Conv1d(input_size, output_size * upsample_factor, kernel_size=3)),
                nn.LeakyReLU(),
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(
                    nn.Conv1d(output_size * upsample_factor, output_size * upsample_factor, kernel_size=3)),
                nn.LeakyReLU(),
                nn.ReflectionPad1d(1),
                nn.utils.weight_norm(
                    nn.Conv1d(output_size * upsample_factor, output_size * upsample_factor, kernel_size=3)),
                nn.LeakyReLU(),
            )

        if hp is not None:
            self.org = Config.up_org
            self.no_skip = Config.no_skip
        else:
            self.org = False
            self.no_skip = True

        if self.use_smooth:
            self.mas = nn.Sequential(
                # LowpassBlur(output_size, self.window_len),
                MovingAverageSmooth(output_size, self.window_len),
                # MovingAverageSmooth(output_size, self.window_len),
            )

    def forward(self, inputs):

        if not self.org:
            inputs = inputs + torch.sin(inputs)
            B, C, T = inputs.size()
            res = inputs.repeat(1, self.upsample_factor, 1).view(B, C, -1)
            skip = self.skip_conv(res)
            if self.up_type == "repeat":
                return skip

        outputs = self.layer(inputs)
        if self.up_type == "pn" and self.index > 2:
            B, c, l = outputs.size()
            outputs = outputs.view(B, -1, l * self.upsample_factor)

        if self.no_skip:
            return outputs

        if not self.org:
            outputs = outputs + skip

        if self.use_smooth:
            outputs = self.mas(outputs)

        if self.use_drop:
            outputs = F.dropout(outputs, p=0.05)

        return outputs


class ResStack(nn.Module):
    def __init__(self, channel, kernel_size=3, resstack_depth=4, hp=None):
        super(ResStack, self).__init__()

        self.use_wn = Config.use_wn
        self.use_shift_scale = Config.use_shift_scale
        self.channel = channel

        def get_padding(kernel_size, dilation=1):
            return int((kernel_size * dilation - dilation) / 2)

        if self.use_shift_scale:
            self.scale_conv = nn.utils.weight_norm(
                nn.Conv1d(channel, 2 * channel, kernel_size=kernel_size, dilation=1, padding=1))

        if not self.use_wn:
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(nn.Conv1d(channel, channel,
                                                   kernel_size=kernel_size, dilation=3 ** (i % 10),
                                                   padding=get_padding(kernel_size, 3 ** (i % 10)))),
                    nn.LeakyReLU(),
                    nn.utils.weight_norm(nn.Conv1d(channel, channel,
                                                   kernel_size=kernel_size, dilation=1,
                                                   padding=get_padding(kernel_size, 1))),
                )
                for i in range(resstack_depth)
            ])
        else:
            self.wn = WaveNet(
                in_channels=channel,
                out_channels=channel,
                cin_channels=-1,
                num_layers=resstack_depth,
                residual_channels=channel,
                gate_channels=channel,
                skip_channels=channel,
                # kernel_size=5,
                # dilation_rate=3,
                causal=False,
                use_downup=False,
            )

    def forward(self, x):
        if not self.use_wn:
            for layer in self.layers:
                x = x + layer(x)
        else:
            x = self.wn(x)

        if self.use_shift_scale:
            m_s = self.scale_conv(x)
            m_s = m_s[:, :, :-1]

            m, s = torch.split(m_s, self.channel, dim=1)
            s = F.softplus(s)

            x = m + s * x[:, :, 1:]  # key!!!
            x = F.pad(x, pad=(1, 0), mode='constant', value=0)

        return x


class WaveNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 num_layers=10,
                 residual_channels=64,
                 gate_channels=64,
                 skip_channels=64,
                 kernel_size=3,
                 dilation_rate=2,
                 cin_channels=80,
                 hp=None,
                 causal=False,
                 use_downup=False,
                 ):
        super(WaveNet, self).__init__()

        self.in_channels = in_channels
        self.causal = causal
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size
        self.use_downup = use_downup

        self.front_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=self.residual_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        if self.use_downup:
            self.downup_conv = nn.Sequential(
                nn.Conv1d(in_channels=self.residual_channels, out_channels=self.residual_channels, kernel_size=3,
                          stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.residual_channels, out_channels=self.residual_channels, kernel_size=3,
                          stride=2, padding=1),
                nn.ReLU(),
                UpsampleNet(self.residual_channels, self.residual_channels, 4, hp),
            )

        self.res_blocks = nn.ModuleList()
        for n in range(self.num_layers):
            self.res_blocks.append(ResBlock(self.residual_channels,
                                            self.gate_channels,
                                            self.skip_channels,
                                            self.kernel_size,
                                            dilation=dilation_rate ** n,
                                            cin_channels=self.cin_channels,
                                            local_conditioning=(self.cin_channels > 0),
                                            causal=self.causal,
                                            mode='SAME'))
        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            Conv(self.skip_channels, self.out_channels, 1, causal=self.causal),
        )

    def forward(self, x, c=None):
        return self.wavenet(x, c)

    def wavenet(self, tensor, c=None):

        h = self.front_conv(tensor)
        if self.use_downup:
            h = self.downup_conv(h)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [2 ** (i % self.num_layers) for i in range(self.num_layers)]
        return num_dir * (self.kernel_size - 1) * sum(dilations) + 1 + (self.front_channels - 1)

    def remove_weight_norm(self):
        for f in self.res_blocks:
            f.remove_weight_norm()


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, causal=False, mode='SAME'):
        super(Conv, self).__init__()

        self.causal = causal
        self.mode = mode
        if self.causal and self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1)
        elif self.mode == 'SAME':
            self.padding = dilation * (kernel_size - 1) // 2
        else:
            self.padding = 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, tensor):
        out = self.conv(tensor)
        if self.causal and self.padding is not 0:
            out = out[:, :, :-self.padding]
        return out

    def remove_weight_norm(self):
        nn.utils.remove_weight_norm(self.conv)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size, dilation,
                 cin_channels=None, local_conditioning=True, causal=False, mode='SAME'):
        super(ResBlock, self).__init__()
        self.causal = causal
        self.local_conditioning = local_conditioning
        self.cin_channels = cin_channels
        self.mode = mode

        self.filter_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal, mode)
        self.gate_conv = Conv(in_channels, out_channels, kernel_size, dilation, causal, mode)
        self.res_conv = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(out_channels, skip_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)

        if self.local_conditioning:
            self.filter_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.gate_conv_c = nn.Conv1d(cin_channels, out_channels, kernel_size=1)
            self.filter_conv_c = nn.utils.weight_norm(self.filter_conv_c)
            self.gate_conv_c = nn.utils.weight_norm(self.gate_conv_c)

    def forward(self, tensor, c=None):
        h_filter = self.filter_conv(tensor)
        h_gate = self.gate_conv(tensor)

        if self.local_conditioning:
            h_filter += self.filter_conv_c(c)
            h_gate += self.gate_conv_c(c)

        out = torch.tanh(h_filter) * torch.sigmoid(h_gate)

        res = self.res_conv(out)
        skip = self.skip_conv(out)
        if self.mode == 'SAME':
            return (tensor + res) * math.sqrt(0.5), skip
        else:
            return (tensor[:, :, 1:] + res) * math.sqrt(0.5), skip

    def remove_weight_norm(self):
        self.filter_conv.remove_weight_norm()
        self.gate_conv.remove_weight_norm()
        nn.utils.remove_weight_norm(self.res_conv)
        nn.utils.remove_weight_norm(self.skip_conv)
        nn.utils.remove_weight_norm(self.filter_conv_c)
        nn.utils.remove_weight_norm(self.gate_conv_c)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int])
    s_act = torch.sigmoid(in_act[:, n_channels_int:])
    acts = t_act * s_act
    return acts


@torch.jit.script
def fused_res_skip(tensor, res_skip, n_channels):
    n_channels_int = n_channels[0]
    res = res_skip[:, :n_channels_int]
    skip = res_skip[:, n_channels_int:]
    return (tensor + res), skip


class ResStack2D(nn.Module):
    def __init__(self, channels=16, kernel_size=3, resstack_depth=4, hp=None):
        super(ResStack2D, self).__init__()
        channels = 16
        kernel_size = 3
        resstack_depth = 2
        self.channels = channels

        def get_padding(kernel_size, dilation=1):
            return int((kernel_size * dilation - dilation) / 2)

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv2d(1, self.channels, kernel_size,
                                               dilation=(1, 3 ** (i)),
                                               padding=(1, get_padding(kernel_size, 3 ** (i))))),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv2d(self.channels, self.channels, kernel_size,
                                               dilation=(1, 3 ** (i)),
                                               padding=(1, get_padding(kernel_size, 3 ** (i))))),
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.Conv2d(self.channels, 1, kernel_size=1)))
            for i in range(resstack_depth)])

    def forward(self, tensor):
        x = tensor.unsqueeze(1)
        for layer in self.layers:
            x = x + layer(x)
        x = x.squeeze(1)

        return x


class FiLM(nn.Module):
    """
    feature-wise linear modulation
    """

    def __init__(self, input_dim, attribute_dim):
        super().__init__()
        self.input_dim = input_dim
        self.generator = nn.Conv1d(attribute_dim, input_dim * 2, kernel_size=3, padding=1)

    def forward(self, x, c):
        """
        x: (B, input_dim, seq)
        c: (B, attribute_dim, seq)
        """
        c = self.generator(c)
        m, s = torch.split(c, self.input_dim, dim=1)

        return x * s + m


class FiLMConv1d(nn.Module):
    """
    Conv1d with FiLMs in between
    """

    def __init__(self, in_size, out_size, attribute_dim, ins_norm=True, loop=1):
        super().__init__()
        self.loop = loop
        self.mlps = nn.ModuleList(
            [nn.Conv1d(in_size, out_size, kernel_size=3, padding=1)]
            + [nn.Conv1d(out_size, out_size, kernel_size=3, padding=1) for i in range(loop - 1)])
        self.films = nn.ModuleList([FiLM(out_size, attribute_dim) for i in range(loop)])
        self.ins_norm = ins_norm
        if self.ins_norm:
            self.norm = nn.InstanceNorm1d(attribute_dim)

    def forward(self, x, c):
        """
        x: (B, input_dim, seq)
        c: (B, attribute_dim, seq)
        """
        if self.ins_norm:
            c = self.norm(c)
        for i in range(self.loop):
            x = self.mlps[i](x)
            x = F.relu(x)
            x = self.films[i](x, c)

        return x
