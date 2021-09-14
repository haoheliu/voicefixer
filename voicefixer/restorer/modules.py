import torch.nn as nn
import torch
import torch.nn.functional as F
import math

class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, size, activation, momentum):
        super(ConvBlockRes, self).__init__()

        self.activation = activation
        if(type(size) == type((3,4))):
            pad = size[0] // 2
            size = size[0]
        else:
            pad = size // 2
            size = size

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(size, size), stride=(1, 1),
                               dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        # self.abn1 = InPlaceABN(num_features=in_channels, momentum=momentum, activation='leaky_relu')

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(size, size), stride=(1, 1),
                               dilation=(1, 1), padding=(pad, pad), bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        # self.abn2 = InPlaceABN(num_features=out_channels, momentum=momentum, activation='leaky_relu')

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.init_weights()

    def init_weights(self):
        init_bn(self.bn1)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, x):
        origin = x
        x = self.conv1(F.leaky_relu_(self.bn1(x), negative_slope=0.01))
        x = self.conv2(F.leaky_relu_(self.bn2(x), negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(origin) + x
        else:
            return origin + x

class EncoderBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, downsample, activation, momentum):
        super(EncoderBlockRes, self).__init__()
        size = 3

        self.conv_block1 = ConvBlockRes(in_channels, out_channels, size, activation, momentum)
        self.conv_block2 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.downsample = downsample

    def forward(self, x):
        encoder = self.conv_block1(x)
        encoder = self.conv_block2(encoder)
        encoder = self.conv_block3(encoder)
        encoder = self.conv_block4(encoder)
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder

class DecoderBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation, momentum):
        super(DecoderBlockRes, self).__init__()
        size = 3
        self.activation = activation

        self.conv1 = torch.nn.ConvTranspose2d(in_channels=in_channels,
            out_channels=out_channels, kernel_size=(size, size), stride=stride,
            padding=(0, 0), output_padding=(0, 0), bias=False, dilation=(1, 1))

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_block2 = ConvBlockRes(out_channels * 2, out_channels, size, activation, momentum)
        self.conv_block3 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block4 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)
        self.conv_block5 = ConvBlockRes(out_channels, out_channels, size, activation, momentum)

    def init_weights(self):
        init_layer(self.conv1)

    def prune(self, x, both=False):
        """Prune the shape of x after transpose convolution.
        """
        if(both): x = x[:, :, 0 : - 1, 0:-1]
        else: x = x[:, :, 0: - 1, :]
        return x

    def forward(self, input_tensor, concat_tensor,both=False):
        x = self.conv1(F.relu_(self.bn1(input_tensor)))
        x = self.prune(x,both=both)
        x = torch.cat((x, concat_tensor), dim=1)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        return x




def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_gru(rnn):
    """Initialize a GRU layer. """

    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)

        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in: (i + 1) * fan_in, :])

    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))

    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)

from torch.cuda import init

def act(x, activation):
    if activation == 'relu':
        return F.relu_(x)

    elif activation == 'leaky_relu':
        return F.leaky_relu_(x, negative_slope=0.2)

    elif activation == 'swish':
        return x * torch.sigmoid(x)

    else:
        raise Exception('Incorrect activation!')
