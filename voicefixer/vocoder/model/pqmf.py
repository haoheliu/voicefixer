import os
import sys
import torch
import torch.nn as nn
import numpy as np
import scipy.io.wavfile

class PQMF(nn.Module):
    def __init__(self, N, M, file_path="utils/pqmf_hk_4_64.dat"):
        super().__init__()
        self.N = N #nsubband
        self.M = M #nfilter
        self.ana_conv_filter = nn.Conv1d(1, out_channels=N, kernel_size=M, stride=N, bias=False)
        data=np.reshape(np.fromfile(file_path, dtype=np.float32), (N, M))
        data=np.flipud(data.T).T
        gk=data.copy()
        data=np.reshape(data, (N, 1, M)).copy()
        dict_new = self.ana_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(data)
        self.ana_pad = nn.ConstantPad1d((M-N,0), 0)
        self.ana_conv_filter.load_state_dict(dict_new)

        self.syn_pad = nn.ConstantPad1d((0, M//N-1), 0)
        self.syn_conv_filter = nn.Conv1d(N, out_channels=N, kernel_size=M//N, stride=1, bias=False)
        gk=np.transpose(np.reshape(gk,(4, 16, 4)), (1, 0, 2))*N
        gk=np.transpose(gk[::-1,:,:], (2, 1, 0)).copy()
        dict_new = self.syn_conv_filter.state_dict().copy()
        dict_new['weight'] = torch.from_numpy(gk)
        self.syn_conv_filter.load_state_dict(dict_new)

        for param in self.parameters():
            param.requires_grad = False

    def analysis(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))
    def synthesis(self, inputs):
        return self.syn_conv_filter(self.syn_pad(inputs))
    def forward(self, inputs):
        return self.ana_conv_filter(self.ana_pad(inputs))

if __name__ == "__main__":
    a=PQMF(4, 64)
    #x = np.load('data/train/audio/010000.npy')
    x = np.zeros([8, 24000], np.float32)
    x=np.reshape(x, (8, 1, -1))
    x = torch.from_numpy(x)
    b=a.analysis(x)
    c=a.synthesis(b)
    print(x.shape, b.shape, c.shape)
    b=(b * 32768).numpy()
    b=np.reshape(np.transpose(b, (0, 2, 1)), (-1, 1)).astype(np.int16)
    #b.tofile('1.pcm')
    #np.reshape(np.transpose(c.numpy()*32768, (0, 2, 1)), (-1,1)).astype(np.int16).tofile('2.pcm')
