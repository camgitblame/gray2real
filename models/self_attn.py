# Adapted from https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
import torch
import torch.nn as nn


class SelfAttn(nn.Module):
    """🧠 Self-Attention layer for global context modeling in feature maps

    Parameters:
        in_dim (int): Number of input feature channels
        activation (nn.Module): Activation to apply after key/query/value projections
        forward_outputs_attention (bool): Whether to return (out, attn) or just out
        where_to_log_attention (dict, optional): Where to store attention maps externally
    """

    def __init__(
        self,
        in_dim,
        activation,
        forward_outputs_attention=True,
        where_to_log_attention=None,
    ):
        super(SelfAttn, self).__init__()
        self.channel_in = in_dim
        self.activation = activation
        self.forward_outputs_attention = forward_outputs_attention
        self.where_to_log_attention = where_to_log_attention

        # ✨ Projection layers for query, key, value
        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )

        # 🔧 Learnable scale parameter
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs:
            x (tensor): Input feature maps (B, C, W, H)
        returns:
            out: Self-attention enhanced output
            attention (optional): Attention map (B, N, N) where N = W * H
        """
        B, C, W, H = x.size()
        N = W * H

        # 📐 Generate queries, keys, and values
        proj_query = self.activation(
            self.query_conv(x).view(B, -1, N).permute(0, 2, 1)
        )  # (B, N, C//8)
        proj_key = self.activation(self.key_conv(x).view(B, -1, N))  # (B, C//8, N)
        energy = torch.bmm(proj_query, proj_key)  # (B, N, N)
        attention = self.softmax(energy)  # (B, N, N)

        proj_value = self.activation(self.value_conv(x).view(B, -1, N))  # (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)).view(
            B, C, W, H
        )  # (B, C, W, H)

        out = self.gamma * out + x  # 🧮 Residual connection

        if self.where_to_log_attention is not None:
            self.where_to_log_attention["attn"] = attention

        return (out, attention) if self.forward_outputs_attention else out
