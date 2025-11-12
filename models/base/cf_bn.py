import torch.nn as nn


class ChannelFreqBatchNorm(nn.Module):
    """Channel-Freq-wise Batch Normalization (cfBN)"""

    def __init__(self, num_features):
        super(ChannelFreqBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        """
        Args:
            x: [B, C, T, F]
        Returns:
            y: [B, C, T, F]
        """
        x_i = x.permute(0, 1, 3, 2).contiguous().view(x.shape[0], -1, x.shape[2])  # [B, C*F, T]
        y = self.bn(x_i).view(x.shape[0], x.shape[1], x.shape[3], -1).permute(0, 1, 3, 2)  # [B, C, T, F]
        return y

