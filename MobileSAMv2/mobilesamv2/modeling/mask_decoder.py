# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Union

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(approximate='tanh'),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(approximate='tanh'),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )
        # self.mask_han=None

        self._export = False

    @torch.jit.ignore
    def _get_slice(self, flag):
        if flag:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        
        return mask_slice
    
    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: Union[bool, torch.Tensor] = False,
        simple_type=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        if not simple_type:
            masks, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )
        else:
            masks, iou_pred = self.predict_masks_simple(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
            )
        # if multimask_output:
        #     mask_slice = slice(1, None)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]
        
        # if isinstance(multimask_output, torch.Tensor):
        #     multimask_output.to(torch.bool)

        if self._export:
            mask_split = torch.split(masks, 1, 1)
            iou_split = torch.split(iou_pred, 1, 1)
            if multimask_output:
                masks = torch.cat(mask_split[1:], 1)
                iou_pred = torch.cat(iou_split[1:], 1)
            else:
                masks = mask_split[0]
                iou_pred = iou_split[0]
        else:
            if multimask_output:
                masks = masks[:, 1:, :, :]
                iou_pred = iou_pred[:, 1:]
            else:
                masks = masks[:, 0:1, :, :]
                iou_pred = iou_pred[:, 0:1]
            # masks = masks[:,0,:,:].unsqueeze(1)
            # iou_pred = iou_pred[:, 0].unsqueeze(1)

        # print(f"{masks.shape=}")

        # mask_slice = self._get_slice(multimask_output)
        # if multimask_output:
        #     mask_slice = slice(1, 4)
        # else:
        #     mask_slice = slice(0, 1)
        # masks = masks[:, mask_slice, :, :]
        # iou_pred = iou_pred[:, mask_slice]
        if self._export:
            # 1,1,256,256
            masks = F.interpolate(
                masks,
                (1024, 1024),
                mode="bilinear",
                align_corners=False,
            )
            return torch.permute(masks,(0,2,3,1))
        else:
            return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        if self._export:
            output_tokens = (torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)).unsqueeze(0) # (1, 1, 2, 256) 
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        
        print(f"{sparse_prompt_embeddings.shape=}")
        if self._export:
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1, -1)
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2).squeeze(1)
        else:
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        print(f"{image_embeddings.device=}")
        print(f"{tokens.device=}")
        print(f"{tokens.shape=}")
        # Expand per-image data in batch direction to be per-mask
        if tokens.shape[0] > 1:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if tokens.shape[0] > 1:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe

        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        if self._export:
            hs_split = torch.split(hs, 1, 1)
            iou_token_out = hs_split[0].squeeze(1)
            mask_tokens_out = torch.cat(hs_split[1: 1 + self.num_mask_tokens], 1)
        else:
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        if self._export:
            # src is (b n c)

            # src_list = torch.split(src, 1, 2) # each b n 1
            # new_src_list = []
            # for i in range(len(src_list)):
            #     new_src_list.append(src_list[i].squeeze(-1).reshape((b, h, w)))
            # src = torch.stack(new_src_list, 1)

            src = src.reshape((b, h, w, c))
            src = src.permute((0,3,1,2)) # THIS NEEDS TO NOT HAPPEN IN TFLITE CONVERSION
            # src = src.permute((0,2,1)).reshape((b, c, h, w))
        else:
            src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        # import pdb;pdb.set_trace()
        if self._export:
            mask_splits = torch.split(mask_tokens_out, 1, 1)
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_splits[i].squeeze(1)))
        else:
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        # masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.bmm(hyper_in, upscaled_embedding.reshape((b, c, -1)))
        masks = masks.reshape((b, -1, h, w))
        # import pdb;pdb.set_trace()
        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    def predict_masks_simple(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        src=image_embeddings

        src = src + dense_prompt_embeddings
        pos_src=image_pe
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]
        src = src.transpose(1, 2).view(b, c, h, w)
        
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
