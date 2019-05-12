import torch
import torch.nn as nn
import torch.functional as F
import numpy as np


def conv_size(H_in, k_size, stride, padd, dil=1):
    H_out = np.floor((H_in + 2 * padd - dil * (k_size - 1) - 1) / stride + 1)
    return np.int(H_out)


class BatchFlatten(nn.Module):
    def __init__(self):
        super(BatchFlatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BatchReshape(nn.Module):
    def __init__(self, shape):
        super(BatchReshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze()


class DCGAN2_Encoder(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(DCGAN2_Encoder, self).__init__()

        H_conv_out = conv_size(input_shape[-1], 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 3, 1, 1)
        H_conv_out = conv_size(H_conv_out, 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 3, 1, 1)

        convnet_out = np.int(H_conv_out * H_conv_out * out_channels * 2)

        self.H_conv_out = H_conv_out

        self.encoder = nn.ModuleList([
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.Conv2d(1, out_channels, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels * 2, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels * 2),

            nn.Conv2d(out_channels * 2, out_channels * 2, 3, 1, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(out_channels * 2),

            BatchFlatten(),
            nn.Linear(convnet_out, encoder_size)
        ])

        self.encoder_mu = nn.Linear(encoder_size, latent_size)
        self.encoder_std = nn.Linear(encoder_size, latent_size)

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        mu = self.encoder_mu(x)
        std = self.encoder_std(x)
        std = torch.clamp(torch.sigmoid(std), min=0.01)
        return mu, std


class DCGAN2_Decoder(nn.Module):
    def __init__(self, H_conv_out, out_channels, decoder_size, latent_size):
        super(DCGAN2_Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, decoder_size),
            nn.ReLU(),

            nn.Linear(decoder_size, H_conv_out * H_conv_out * out_channels * 2),
            nn.ReLU(),
            BatchReshape((out_channels * 2, H_conv_out, H_conv_out, )),

            nn.ConvTranspose2d(out_channels * 2, out_channels, 4, 2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels, out_channels // 2, 4, 2, padding=1),
            nn.ReLU(),

            nn.ConvTranspose2d(out_channels // 2, 1, 3, 1, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x


class DCGAN_Encoder(nn.Module):
    def __init__(self, input_shape, out_channels, encoder_size, latent_size):
        super(DCGAN_Encoder, self).__init__()

        H_conv_out = conv_size(input_shape[-1], 4, 2, 1)
        H_conv_out = conv_size(H_conv_out, 4, 2, 1)
        convnet_out = np.int(H_conv_out * H_conv_out * out_channels * 2)
        self.H_conv_out = H_conv_out

        self.encoder = nn.ModuleList([
            # in_channels, out_channels, kernel_size, stride=1, padding=0
            nn.Conv2d(1, out_channels, 4, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels * 2, 4, 2, padding=1),
            nn.LeakyReLU(),
            BatchFlatten(),
            nn.Linear(convnet_out, encoder_size),
            nn.LeakyReLU(),
            nn.Linear(encoder_size, latent_size)
        ])

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class DCGAN_Decoder(nn.Module):
    def __init__(self, H_conv_out, out_channels, decoder_size, latent_size):
        super(DCGAN_Decoder, self).__init__()
        self.decoder = nn.ModuleList([
            nn.Linear(latent_size, decoder_size),
            nn.ReLU(),
            nn.Linear(decoder_size, H_conv_out * H_conv_out * out_channels * 2),
            nn.ReLU(),
            BatchReshape((out_channels * 2, H_conv_out, H_conv_out, )),
            nn.ConvTranspose2d(out_channels * 2, out_channels, 4, 2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(out_channels, 1, 4, 2, padding=1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x
