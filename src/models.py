import torch
import torch.nn as nn


class UNet2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.5):
        super(UNet2D, self).__init__()
        # Encoding layers.
        self.encoder_conv_1 = self.double_conv(in_channels, 16, 3, 1, 1)
        self.encoder_conv_2 = self.double_conv(16, 32, 3, 1, 1)
        self.encoder_conv_3 = self.double_conv(32, 64, 3, 1, 1)

        # Bottleneck layer.
        self.bottleneck = self.double_conv(64, 128, 3, 1, 1)

        # Decoding layers - input channels are doubled to account for
        # concatenation with the encoding layers.
        self.decoder_conv_1 = self.double_conv(64 * 2, 64, 3, 1, 1)
        self.decoder_conv_2 = self.double_conv(32 * 2, 32, 3, 1, 1)
        self.decoder_conv_3 = self.double_conv(16 * 2, 16, 3, 1, 1)

        # Down sampling layers same for all encoding layers.
        self.down_sample = self.maxpool_w_dropout(2, 2, 0, dropout_p)

        # Up sampling layers (TODO: Check performance with 2x2 kernel).
        self.up_sample_1 = self.up_sample(128, 64, 3, 2, 1, 1)
        self.up_sample_2 = self.up_sample(64, 32, 3, 2, 1, 1)
        self.up_sample_3 = self.up_sample(32, 16, 3, 2, 1, 1)

        # Final layer.
        self.final_layer = nn.Conv2d(16, out_channels, 1)

    def double_conv(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def up_sample(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        input_padding,
        output_padding,
    ):
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=input_padding,
            output_padding=output_padding,
        )

    def maxpool_w_dropout(self, kernel_size, stride, padding, dropout_p):
        return nn.Sequential(
            nn.MaxPool2d(kernel_size, stride, padding), nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        # Encoding layers.
        x0 = self.encoder_conv_1(x)
        x1 = self.down_sample(x0)
        x2 = self.encoder_conv_2(x1)
        x3 = self.down_sample(x2)
        x4 = self.encoder_conv_3(x3)
        x5 = self.down_sample(x4)

        # Bottleneck layer.
        x6 = self.bottleneck(x5)

        # Decoding layers.
        x7 = self.up_sample_1(x6)
        x8 = self.decoder_conv_1(torch.cat([x7, x4], dim=1))
        x9 = self.up_sample_2(x8)
        x10 = self.decoder_conv_2(torch.cat([x9, x2], dim=1))
        x11 = self.up_sample_3(x10)
        x12 = self.decoder_conv_3(torch.cat([x11, x0], dim=1))
        return self.final_layer(x12)
