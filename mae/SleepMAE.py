import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
from .patch import PatchEmbedding
from .mask import MaskGenerator
from .position_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers
from .instance_norm import InstanceNorm


class SleepMAE(nn.Module):
    def __init__(self, patch_size, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, clip_num,
                 mask_ratio, encoder_depth, length, mode="pre-train"):
        super().__init__()
        assert mode in ["pre-train", "classify"], "Error mode."
        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token
        self.clip_num = clip_num
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio
        self.length = length
        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # encoder specifics
        # patchify & embedding
        self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        # masking
        self.mask = MaskGenerator(num_token, mask_ratio)
        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        # # decoder
        # self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)
        self.decoder = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim // 2, bias=True),
                                     nn.ReLU(),
                                     nn.Linear(self.embed_dim // 2, self.patch_size, bias=True))

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, patch_size)
        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        # mask token
        trunc_normal_(self.mask_token, std=.02)

    def encoding(self, long_term_history, mask=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        batch_size, num_nodes, _, _ = long_term_history.shape
        # patchify and embed input
        patches = self.patch_embedding(long_term_history)  # B, N, d, P
        patches = patches.transpose(-1, -2)  # B, N, P, d
        # positional embedding
        patches = self.positional_encoding(patches)

        # mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_unmasked = patches[:, :, unmasked_token_index, :]  # B, N, unmasked_P, d

        encoder_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), self.embed_dim),
            index=masked_token_index)
        encoder_input = torch.cat([encoder_unmasked, encoder_masked], dim=-2)  # B, N, P, d
        # encoding
        hidden_states = self.encoder(encoder_input)  # B, N, P, d
        hidden_states = self.encoder_norm(hidden_states).view(batch_size, num_nodes, -1, self.embed_dim)

        return hidden_states, unmasked_token_index, masked_token_index

    # def encoding(self, long_term_history, mask_index, unmask_index):
    #     batch_size, num_nodes, _, _ = long_term_history.shape
    #     # patchify and embed input
    #     patches = self.patch_embedding(long_term_history)  # B, N, d, P
    #     patches = patches.transpose(-1, -2)  # B, N, P, d
    #     # positional embedding
    #     patches = self.positional_encoding(patches)
    #
    #     unmasked_token_index, masked_token_index = unmask_index, mask_index
    #     encoder_unmasked = patches[:, :, unmasked_token_index, :]  # B, N, unmasked_P, d
    #
    #     encoder_masked = self.positional_encoding(
    #         self.mask_token.expand(batch_size, num_nodes, len(masked_token_index), self.embed_dim),
    #         index=masked_token_index)
    #     encoder_input = torch.cat([encoder_unmasked, encoder_masked], dim=-2)  # B, N, P, d
    #     # encoding
    #     hidden_states = self.encoder(encoder_input)  # B, N, P, d
    #     hidden_states = self.encoder_norm(hidden_states).view(batch_size, num_nodes, -1, self.embed_dim)
    #
    #     return hidden_states, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states, masked_token_index):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, num_nodes, _, _ = hidden_states.shape

        # decoding
        hidden_states_masked = hidden_states[:, :, self.num_token - len(masked_token_index):, :]
        reconstruction_masked = self.decoder(hidden_states_masked)

        return reconstruction_masked

    def get_reconstructed_masked_tokens(self, reconstruction_masked, real_value_full, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_masked (torch.Tensor): reconstructed masked tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        batch_size, num_nodes, _, _ = reconstruction_masked.shape  # B, N, r*P, L
        reconstruction_masked_tokens = reconstruction_masked.view(batch_size, num_nodes, -1).transpose(1,
                                                                                                       2)  # B, r*P*L, N

        label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :,
                     self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, :, masked_token_index, :].contiguous()  # B, N, r*P, L
        label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)  # B, r*P*L, N

        return reconstruction_masked_tokens, label_masked_tokens

    # def get_reconstructed_masked_tokens(self, reconstruction_masked, real_value_full, masked_token_index):
    #     """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.
    #
    #     Args:
    #         reconstruction_masked (torch.Tensor): reconstructed masked tokens.
    #         real_value_full (torch.Tensor): ground truth full tokens.
    #         masked_token_index (list): masked token index.
    #
    #     Returns:
    #         torch.Tensor: reconstructed masked tokens.
    #         torch.Tensor: ground truth masked tokens.
    #     """
    #     # get reconstructed masked tokens
    #     batch_size, num_nodes, _, _ = reconstruction_masked.shape  # B, N, r*P, L
    #     reconstruction_masked_tokens = reconstruction_masked.view(batch_size, num_nodes, -1).transpose(1,
    #                                                                                                    2)  # B, r*P*L, N
    #
    #     label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :,
    #                  self.selected_feature, :].transpose(1, 2)  # B, N, P, L
    #
    #     return reconstruction_masked_tokens, label_full

    def forward(self, history_data):
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
                :param history_data: input eeg signal
        """
        # reshape
        history_data = history_data.unsqueeze(1)
        history_data = history_data.permute(0, 1, 3, 2)  # B, N, 1, L * P
        history_data = history_data.view(history_data.size(0) * self.clip_num * self.length, history_data.size(1),
                                         history_data.size(2), history_data.size(3) // (self.clip_num * self.length))
        # print(history_data.shape)
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states, unmasked_token_index, masked_token_index = self.encoding(history_data)
            # decoding
            reconstruction_masked = self.decoding(hidden_states, masked_token_index)
            # for subsequent loss computing
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
                reconstruction_masked, history_data, masked_token_index)
            return reconstruction_masked_tokens, label_masked_tokens, hidden_states
        else:
            hidden_states, _, _ = self.encoding(history_data)
            return hidden_states

    # def forward(self, history_data, mask_index, unmask_index):
    #     # reshape
    #     history_data = history_data.unsqueeze(1)
    #     history_data = history_data.permute(0, 1, 3, 2)  # B, N, 1, L * P
    #     history_data = history_data.view(history_data.size(0) * self.clip_num * self.length, history_data.size(1),
    #                                      history_data.size(2), history_data.size(3) // (self.clip_num * self.length))
    #     hidden_states, _, _ = self.encoding(history_data, mask_index, unmask_index)
    #     return hidden_states

    # def forward(self, history_data):
    #     # reshape
    #     history_data = history_data.unsqueeze(1)
    #     history_data = history_data.permute(0, 1, 3, 2)  # B, N, 1, L * P
    #     history_data = history_data.view(history_data.size(0) * self.clip_num * self.length, history_data.size(1),
    #                                      history_data.size(2), history_data.size(3) // (self.clip_num * self.length))
    #     # print(history_data.shape)
    #     # feed forward
    #     if self.mode == "pre-train":
    #         # encoding
    #         hidden_states, unmasked_token_index, masked_token_index = self.encoding(history_data)
    #         # decoding
    #         reconstruction_masked = self.decoding(hidden_states, masked_token_index)
    #         # for subsequent loss computing
    #         reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(
    #             reconstruction_masked, history_data, masked_token_index)
    #         return reconstruction_masked_tokens, label_masked_tokens, masked_token_index
    #     else:
    #         hidden_states, _, _ = self.encoding(history_data)
    #         return hidden_states
