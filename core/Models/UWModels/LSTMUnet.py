import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet_emb_oneBranch_symmetry_noreflect(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, bias=False):
        super(UNet_emb_oneBranch_symmetry_noreflect, self).__init__()

        self.cond1 = nn.Conv2d(in_channels, 32, 3, 1, 1, bias=True)
        self.cond_add1 = nn.Conv2d(32, out_channels, 3, 1, 1, bias=True)

        self.condx = nn.Conv2d(32, 64, 3, 1, 1, bias=True)
        self.condy = nn.Conv2d(64, 32, 3, 1, 1, bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.ResidualBlock1 = ResidualBlock(32, 32)
        self.ResidualBlock2 = ResidualBlock(32, 32)
        self.ResidualBlock3 = ResidualBlock(64, 64)
        self.ResidualBlock4 = ResidualBlock(64, 64)
        self.ResidualBlock5 = ResidualBlock(32, 32)
        self.ResidualBlock6 = ResidualBlock(32, 32)
        self.degradation = DegradationAware(64, 64)

        self.PPM1 = PPM1(32, 8, bins=(1, 2, 3, 6))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight.data)
                m.weight.data.normal_(0.0, 0.02)
                # nn.init.zeros_(m.bias.data)

    def forward(self, x):

        light_conv1 = self.lrelu(self.cond1(x))
        res1 = self.ResidualBlock1(light_conv1)

        res2 = self.ResidualBlock2(res1)
        res2 = self.PPM1(res2)
        res2 = self.condx(res2)

        res3 = self.ResidualBlock3(res2)
        res4 = self.ResidualBlock4(res3)
        res4_1 = self.degradation(res4)

        res4 = self.condy(res4_1)
        res5 = self.ResidualBlock5(res4)

        res6 = self.ResidualBlock6(res5)

        light_map = self.relu(self.cond_add1(res6))

        return light_map


class PPM1(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM1, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim + reduction_dim * 4, in_dim, kernel_size=3, padding=1, bias=False),
            nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.lrelu(out)
        return out


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False, padding_mode='reflect')


class DegradationAware(nn.Module):
    def __init__(self, in_channels, out_channels, lstm_hidden_size=64, lstm_num_layers=1):
        super(DegradationAware, self).__init__()
        self.fc = nn.Linear(lstm_hidden_size, out_channels)
        self.lstm_hidden_size = lstm_hidden_size
        # LSTM for degradation awareness
        self.lstm = nn.LSTM(input_size=in_channels * 3 * 3, hidden_size=lstm_hidden_size,
                            num_layers=lstm_num_layers, batch_first=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1x1 = nn.Conv2d(self.lstm_hidden_size, self.in_channels, kernel_size=1)
    def forward(self, x):
        # print("Input x shape:", x.shape)  # Original input shape (batch_size, channels, height, width)
        out_channels = self.out_channels
        # 1. Padding to make height and width divisible by 3
        x_pad = F.pad(x, (1, 1, 1, 1))
        x_unfolded = x_pad.unfold(2, 3, 2).unfold(3, 3, 2)  # (batch_size, channels, num_windows_h, num_windows_w, 3, 3)
        # print("Unfolded features shape:", x_unfolded.shape)  # (batch_size, in_channels, num_windows_h, num_windows_w, 3, 3)

        # 3. Reshape to (batch_size, in_channels, num_windows, 9)
        batch_size, channels, h_unfold, w_unfold, window_h, window_w = x_unfolded.size()
        num_windows = h_unfold * w_unfold  # Total number of windows (256 * 256 = 65536)
        lstm_input = x_unfolded.contiguous().view(batch_size, num_windows, channels * window_h * window_w)
        # 5. Pass through LSTM
        lstm_out, _ = self.lstm(lstm_input)  # lstm_out: (batch_size, num_windows, lstm_hidden_size)
        # print("After LSTM output shape:", lstm_out.shape)  # (batch_size, num_windows, lstm_hidden_size)
        # lstm_out_1 = lstm_out.view(batch_size, h_unfold, w_unfold, self.lstm_hidden_size)
        # print("LSTM_out:",lstm_out.shape)
        # lstm_out_upsampled = F.interpolate(lstm_out_1.permute(0, 3, 1, 2), scale_factor=2, mode='bilinear',align_corners=False)


        # 6. Extract the last LSTM output for each window
        last_lstm_out = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_size)
        # print("Last LSTM output shape:", last_lstm_out.shape)  # (batch_size, lstm_hidden_size)

        # 7. Compute degradation score
        degradation_score = self.fc(last_lstm_out)  # (batch_size, out_channels)
        # print("Degradation score shape:", degradation_score.shape)  # (batch_size, out_channels)

        # 8. Reshape degradation_score to (batch_size, out_channels, 1, 1) for broadcasting
        degradation_score = degradation_score.view(batch_size, out_channels, 1, 1)  # (batch_size, out_channels, 1, 1)
        # print("Reshaped degradation score shape:", degradation_score.shape)  # (batch_size, out_channels, 1, 1)


        degradation_map = x * degradation_score  # (batch_size, out_channels, H_padded, W_padded)
        # print("Degradation map shape:", degradation_map.shape)  # (batch_size, out_channels, H_padded, W_padded)

        return degradation_map

