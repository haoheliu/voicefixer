import torch
import torch.nn as nn
import numpy as np
from voicefixer.vocoder.model.modules import UpsampleNet, ResStack
from voicefixer.vocoder.config import Config
from voicefixer.vocoder.model.pqmf import PQMF
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=128,
        use_elu=False,
        use_gcnn=False,
        up_org=False,
        group=1,
        hp=None,
    ):
        super(Generator, self).__init__()
        self.hp = hp
        channels = Config.channels
        self.upsample_scales = Config.upsample_scales
        self.use_condnet = Config.use_condnet
        self.out_channels = Config.out_channels
        self.resstack_depth = Config.resstack_depth
        self.use_postnet = Config.use_postnet
        self.use_cond_rnn = Config.use_cond_rnn
        if self.use_condnet:
            cond_channels = Config.cond_channels
            self.condnet = nn.Sequential(
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(in_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(cond_channels, cond_channels, kernel_size=3, padding=1)
                ),
                nn.ELU(),
            )
            in_channels = cond_channels
        if self.use_cond_rnn:
            self.rnn = nn.GRU(
                cond_channels,
                cond_channels // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        if use_elu:
            act = nn.ELU()
        else:
            act = nn.LeakyReLU(0.2, True)

        kernel_size = Config.kernel_size

        if self.out_channels == 1:
            self.generator = nn.Sequential(
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, channels, kernel_size=7)),
                act,
                UpsampleNet(channels, channels // 2, self.upsample_scales[0], hp, 0),
                ResStack(channels // 2, kernel_size[0], self.resstack_depth[0], hp),
                act,
                UpsampleNet(
                    channels // 2, channels // 4, self.upsample_scales[1], hp, 1
                ),
                ResStack(channels // 4, kernel_size[1], self.resstack_depth[1], hp),
                act,
                UpsampleNet(
                    channels // 4, channels // 8, self.upsample_scales[2], hp, 2
                ),
                ResStack(channels // 8, kernel_size[2], self.resstack_depth[2], hp),
                act,
                UpsampleNet(
                    channels // 8, channels // 16, self.upsample_scales[3], hp, 3
                ),
                ResStack(channels // 16, kernel_size[3], self.resstack_depth[3], hp),
                act,
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels // 16, self.out_channels, kernel_size=7)
                ),
                nn.Tanh(),
            )
        else:
            channels = Config.m_channels
            self.generator = nn.Sequential(
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(nn.Conv1d(in_channels, channels, kernel_size=7)),
                act,
                UpsampleNet(channels, channels // 2, self.upsample_scales[0], hp),
                ResStack(channels // 2, kernel_size[0], self.resstack_depth[0], hp),
                act,
                UpsampleNet(channels // 2, channels // 4, self.upsample_scales[1], hp),
                ResStack(channels // 4, kernel_size[1], self.resstack_depth[1], hp),
                act,
                UpsampleNet(channels // 4, channels // 8, self.upsample_scales[3], hp),
                ResStack(channels // 8, kernel_size[3], self.resstack_depth[2], hp),
                act,
                nn.ReflectionPad1d(3),
                nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels // 8, self.out_channels, kernel_size=7)
                ),
                nn.Tanh(),
            )
        if self.out_channels > 1:
            self.pqmf = PQMF(4, 64)

        self.num_params()

    def forward(self, conditions, use_res=False, f0=None):
        res = conditions
        if self.use_condnet:
            conditions = self.condnet(conditions)
        if self.use_cond_rnn:
            conditions, _ = self.rnn(conditions.transpose(1, 2))
            conditions = conditions.transpose(1, 2)

        wav = self.generator(conditions)
        if self.out_channels > 1:
            B = wav.size(0)
            f_wav = (
                self.pqmf.synthesis(wav)
                .transpose(1, 2)
                .reshape(B, 1, -1)
                .clamp(-0.99, 0.99)
            )
            return f_wav, wav
        return wav

    def num_params(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        return parameters
        # print('Trainable Parameters: %.3f million' % parameters)

    def remove_weight_norm(self):
        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


if __name__ == "__main__":
    model = Generator(128)
    x = torch.randn(3, 128, 13)
    print(x.shape)
    y = model(x)
    print(y.shape)
