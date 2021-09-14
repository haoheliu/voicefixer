from voicefixer.restorer.modules import *

from voicefixer.tools.pytorch_util import *

class UNetResComplex_100Mb(nn.Module):
    def __init__(self, channels, nsrc=1):
        super(UNetResComplex_100Mb, self).__init__()
        activation = 'relu'
        momentum = 0.01

        self.nsrc = nsrc
        self.channels = channels
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blcoks}

        self.encoder_block1 = EncoderBlockRes(in_channels=channels * nsrc, out_channels=32,
                                              downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block2 = EncoderBlockRes(in_channels=32, out_channels=64,
                                              downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block3 = EncoderBlockRes(in_channels=64, out_channels=128,
                                              downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block4 = EncoderBlockRes(in_channels=128, out_channels=256,
                                              downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block5 = EncoderBlockRes(in_channels=256, out_channels=384,
                                              downsample=(2, 2), activation=activation, momentum=momentum)
        self.encoder_block6 = EncoderBlockRes(in_channels=384, out_channels=384,
                                                 downsample=(2, 2), activation=activation, momentum=momentum)
        self.conv_block7 = ConvBlockRes(in_channels=384, out_channels=384,
                                           size=3, activation=activation, momentum=momentum)
        self.decoder_block1 = DecoderBlockRes(in_channels=384, out_channels=384,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block2 = DecoderBlockRes(in_channels=384, out_channels=384,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block3 = DecoderBlockRes(in_channels=384, out_channels=256,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block4 = DecoderBlockRes(in_channels=256, out_channels=128,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block5 = DecoderBlockRes(in_channels=128, out_channels=64,
                                              stride=(2, 2), activation=activation, momentum=momentum)
        self.decoder_block6 = DecoderBlockRes(in_channels=64, out_channels=32,
                                                 stride=(2, 2), activation=activation, momentum=momentum)

        self.after_conv_block1 = ConvBlockRes(in_channels=32, out_channels=32,
                                                 size=3, activation=activation, momentum=momentum)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=1,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, sp):
        """
        Args:
          input: (batch_size, channels_num, segment_samples)

        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        # Batch normalization
        x = sp

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0: x.shape[-1] - 1]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool)  # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool)  # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5)  # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0: origin_len, :]

        output_dict = {'mel': x}
        return output_dict

if __name__ == "__main__":
    model = UNetResComplex_100Mb(channels=1)
    print(model(torch.randn((1,1,101,128)))['mel'].size())
