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


class SimpleUNet(nn.Module):
    """
    UNet implementation that was written in the tutorial.
    Used for testing purposes since pre trained weights are available.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.conv_block(in_channels, 16, 3, 1, 1)
        self.maxpool1 = self.maxpool_block(2, 2, 0)
        self.conv2 = self.conv_block(16, 32, 3, 1, 1)
        self.maxpool2 = self.maxpool_block(2, 2, 0)
        self.conv3 = self.conv_block(32, 64, 3, 1, 1)
        self.maxpool3 = self.maxpool_block(2, 2, 0)

        self.middle = self.conv_block(64, 128, 3, 1, 1)

        self.upsample3 = self.transposed_block(128, 64, 3, 2, 1, 1)
        self.upconv3 = self.conv_block(128, 64, 3, 1, 1)
        self.upsample2 = self.transposed_block(64, 32, 3, 2, 1, 1)
        self.upconv2 = self.conv_block(64, 32, 3, 1, 1)

        self.upsample1 = self.transposed_block(32, 16, 3, 2, 1, 1)
        self.upconv1 = self.conv_block(32, 16, 3, 1, 1)

        self.final = self.final_layer(16, 1, 1, 1, 0)

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        convolution = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        return convolution

    def maxpool_block(self, kernel_size, stride, padding):
        maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout2d(0.5),
        )
        return maxpool

    def transposed_block(
        self, in_channels, out_channels, kernel_size, stride, padding, output_padding
    ):
        transposed = torch.nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        return transposed

    def final_layer(self, in_channels, out_channels, kernel_size, stride, padding):
        final = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        return final

    def forward(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        # middle part
        middle = self.middle(maxpool3)

        # upsampling part
        upsample3 = self.upsample3(middle)
        upconv3 = self.upconv3(torch.cat([upsample3, conv3], 1))
        upsample2 = self.upsample2(upconv3)
        upconv2 = self.upconv2(torch.cat([upsample2, conv2], 1))
        upsample1 = self.upsample1(upconv2)
        upconv1 = self.upconv1(torch.cat([upsample1, conv1], 1))

        final_layer = self.final(upconv1)

        return final_layer
