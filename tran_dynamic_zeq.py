import torch
import torch.nn as nn
import torch.nn.functional as F

look_back = 5
T = 1

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(embed_dim, dense_dim)
        self.dense2 = nn.Linear(dense_dim, embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.mha(inputs, inputs, inputs)
        out1 = self.layernorm1(inputs + self.dropout1(attn_output))
        out2 = self.layernorm2(out1 + self.dropout2(self.dense2(self.dense1(out1))))
        return out2

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super().__init__()
        self.mha1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.ffn1 = nn.Linear(embed_dim, dense_dim)
        self.ffn2 = nn.Linear(dense_dim, embed_dim)

    def forward(self, inputs, enc_out):
        attn1, _ = self.mha1(inputs, inputs, inputs)
        x1 = self.norm1(inputs + self.dropout1(attn1))
        attn2, _ = self.mha2(x1, enc_out, enc_out)
        x2 = self.norm2(x1 + self.dropout2(attn2))
        x3 = self.norm3(x2 + self.dropout3(self.ffn2(self.ffn1(x2))))
        return x3

class CNN(nn.Module):
    def __init__(self, embed_dim, dense_dim, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, out_channels, kernel_size=1)
        self.fc1 = nn.Linear(out_channels, dense_dim)
        self.fc2 = nn.Linear(dense_dim, embed_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, F, T]
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [B, T, F]
        x = self.fc1(x)
        return self.fc2(x)

class Transformer(nn.Module):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate, num_blocks, output_sequence_length, conv_out_channels, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)
        ])
        self.cnn_blocks = nn.ModuleList([
            CNN(embed_dim, dense_dim, conv_out_channels) for _ in range(num_blocks)
        ])
        self.final_layer = nn.Linear(embed_dim * look_back, output_sequence_length)

    def forward(self, inputs):  # [B, T, input_dim]
        x = self.embedding(inputs)

        for enc in self.encoder_layers:
            x = enc(x)

        cnn = x
        for c in self.cnn_blocks:
            cnn = c(cnn)

        combined = x * cnn

        dec = x
        for d in self.decoder_layers:
            dec = d(dec, combined)

        dec = dec.view(dec.shape[0], -1)  # Flatten [B, T*F]
        return self.final_layer(dec)
