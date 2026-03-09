"""
PRF (Programmed Ribosomal Frameshift) prediction model based on SpliceAI architecture.

Adapted from SpliceAI models with 10k, 2k, 400nt and 80nt receptive fields.
Key change: output is a single binary channel (PRF site / not PRF site)
instead of 3-class (neither/donor/acceptor).

Architecture reference:
https://www.cell.com/cell/pdf/S0092-8674(18)31629-5.pdf
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block RB(N, W, D):
      - N: number of channels
      - W: kernel size
      - D: dilation
    Diagram:
      BatchNorm -> ReLU -> Conv(N, W, D)
      BatchNorm -> ReLU -> Conv(N, W, D)
      Skip connection adds input to output
    """

    def __init__(self, channels, kernel_size, dilation, dropout=0.1):
        super().__init__()

        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding="same",
            dilation=dilation,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
        return x + out


class PRForm_10k(nn.Module):
    """
    PRF prediction model with 10k receptive field.
    Output: single channel (logit) per position for binary PRF classification.
    """

    def __init__(self, in_channels=4, mid_channels=32, out_channels=1, dropout=0.1):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 1, dropout) for _ in range(4)]
        )
        self.conv11_4 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_4 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 4, dropout) for _ in range(4)]
        )
        self.conv21_10 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb21_10 = nn.Sequential(
            *[ResidualBlock(mid_channels, 21, 10, dropout) for _ in range(4)]
        )
        self.conv41_25 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb41_25 = nn.Sequential(
            *[ResidualBlock(mid_channels, 41, 25, dropout) for _ in range(4)]
        )
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        rf = 1
        rf += 4 * 2 * (11 - 1) * 1
        rf += 4 * 2 * (11 - 1) * 4
        rf += 4 * 2 * (21 - 1) * 10
        rf += 4 * 2 * (41 - 1) * 25
        # total rf = 1 + 80 + 320 + 1600 + 8000 = 10001
        self.receptive_field = rf
        self.crop = (rf - 1) // 2  # = 5000

    def forward(self, x):
        x = self.initial_conv(x)
        skip = self.conv11_1(x)
        x = self.rb11_1(x)
        skip += self.conv11_4(x)
        x = self.rb11_4(x)
        skip += self.conv21_10(x)
        x = self.rb21_10(x)
        skip += self.conv41_25(x)
        x = self.rb41_25(x)
        x = self.final_conv1(x) + skip
        x = self.dropout(x)
        x = self.final_conv2(x)
        # Output raw logits; BCEWithLogitsLoss handles sigmoid internally
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x


class PRForm_2k(nn.Module):
    """
    PRF prediction model with 2k receptive field.
    """

    def __init__(self, in_channels=4, mid_channels=32, out_channels=1, dropout=0.1):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 1, dropout) for _ in range(4)]
        )
        self.conv11_4 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_4 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 4, dropout) for _ in range(4)]
        )
        self.conv21_10 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb21_10 = nn.Sequential(
            *[ResidualBlock(mid_channels, 21, 10, dropout) for _ in range(4)]
        )
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        rf = 1
        rf += 4 * 2 * (11 - 1) * 1
        rf += 4 * 2 * (11 - 1) * 4
        rf += 4 * 2 * (21 - 1) * 10
        # total rf = 1 + 80 + 320 + 1600 = 2001
        self.receptive_field = rf
        self.crop = (rf - 1) // 2  # = 1000

    def forward(self, x):
        x = self.initial_conv(x)
        skip = self.conv11_1(x)
        x = self.rb11_1(x)
        skip += self.conv11_4(x)
        x = self.rb11_4(x)
        skip += self.conv21_10(x)
        x = self.rb21_10(x)
        x = self.final_conv1(x) + skip
        x = self.dropout(x)
        x = self.final_conv2(x)
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x


class PRForm_400nt(nn.Module):
    """
    PRF prediction model with 400nt receptive field.
    """

    def __init__(self, in_channels=4, mid_channels=32, out_channels=1, dropout=0.1):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 1, dropout) for _ in range(4)]
        )
        self.conv11_4 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_4 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 4, dropout) for _ in range(4)]
        )
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        rf = 1
        rf += 4 * 2 * (11 - 1) * 1
        rf += 4 * 2 * (11 - 1) * 4
        # total rf = 1 + 80 + 320 = 401
        self.receptive_field = rf
        self.crop = (rf - 1) // 2  # = 200

    def forward(self, x):
        x = self.initial_conv(x)
        skip = self.conv11_1(x)
        x = self.rb11_1(x)
        skip += self.conv11_4(x)
        x = self.rb11_4(x)
        x = self.final_conv1(x) + skip
        x = self.dropout(x)
        x = self.final_conv2(x)
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x


class PRForm_80nt(nn.Module):
    """
    PRF prediction model with 80nt receptive field.
    """

    def __init__(self, in_channels=4, mid_channels=32, out_channels=1, dropout=0.1):
        super().__init__()

        self.initial_conv = nn.Conv1d(in_channels, mid_channels, 1)
        self.conv11_1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.rb11_1 = nn.Sequential(
            *[ResidualBlock(mid_channels, 11, 1, dropout) for _ in range(4)]
        )
        self.final_conv1 = nn.Conv1d(mid_channels, mid_channels, 1)
        self.final_conv2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)

        rf = 1 + 4 * 2 * (11 - 1) * 1  # = 81
        self.receptive_field = rf
        self.crop = (rf - 1) // 2  # = 40

    def forward(self, x):
        x = self.initial_conv(x)
        skip = self.conv11_1(x)
        x = self.rb11_1(x)
        x = self.final_conv1(x) + skip
        x = self.dropout(x)
        x = self.final_conv2(x)
        if x.size(2) > 2 * self.crop:
            x = x[:, :, self.crop : -self.crop]
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test each model variant with different channel counts
    for in_ch in [4, 6, 8]:
        for ModelClass, flank, name in [
            (PRForm_10k, 5000, "10k"),
            (PRForm_2k, 1000, "2k"),
            (PRForm_400nt, 200, "400nt"),
            (PRForm_80nt, 40, "80nt"),
        ]:
            model = ModelClass(in_channels=in_ch, mid_channels=32, out_channels=1, dropout=0.1).to(device)
            block_len = 5000
            x = torch.randn(2, in_ch, block_len + 2 * flank).to(device)
            y = model(x)
            print(f"PRForm_{name} (in_channels={in_ch}): input {x.shape} → output {y.shape}")
            assert y.shape == (2, 1, block_len), f"Expected (2, 1, {block_len}), got {y.shape}"

    print("All model tests passed!")
